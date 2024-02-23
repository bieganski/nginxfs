#!/usr/bin/env python3

import stat
import errno
import threading
import requests
import time
import math

try:
    import fuse
    from fuse import Fuse
except ImportError:
    print("ERROR: 'pip3 install fuse-python' (and probably 'apt-get install libfuse-dev' required!")
    exit(1)

if not hasattr(fuse, '__version__'):
    raise RuntimeError("your fuse-py doesn't know of fuse.__version__, probably it's too old.")

def get_caller_function_name():
    # or caller's caller?
    import inspect
    calling_frame = inspect.currentframe().f_back.f_back
    return calling_frame.f_code.co_name

fuse.fuse_python_api = (0, 2)

def notify_send(msg: str):
    print(msg)
    # run_shell(f"notify-send '{msg}'")

def run_shell(cmd: str) -> tuple[str, str]:
    import subprocess
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, universal_newlines=True)
    stdout, stderr = process.communicate()
    if (ecode := process.returncode):
        raise ValueError(f"Command <{cmd}> exited with {ecode}")
    return stdout, stderr


def _assert(cond: bool, msg: str = None):
    if not cond:
        caller = get_caller_function_name()
        run_shell(f"notify-send 'assert: {msg or caller}'")

def dload_worker(url: str, output: dict[int, bytes], o_lock: threading.Lock, chunk_size: int):
    notify_send(f"worker starts: {url.split('/')[-1]}")
    _assert(chunk_size == (_page_size := 4096))

    response = requests.get(url, stream=True)
    if not response.ok:
        notify_send(f"panic: HTTP {response.status_code}, {url.split('/')[-2:]}")
    
    # NOTE: len(data) is not necessarily 'chunk_size', can be lower or greater.
    last_written_chunk_id = -1
    for data in response.iter_content(chunk_size=chunk_size):
        
        # Split partial response into 'chunk_size' blocks.
        iters = math.ceil(len(data) / chunk_size)
        blocks = [data[i*chunk_size:(i+1)*chunk_size] for i in range(iters)]
        
        with o_lock:
            for b in blocks:
                cur = last_written_chunk_id + 1
                notify_send(f"worker: {len(b)} at {cur * chunk_size}")
                output[cur * chunk_size] = b
                last_written_chunk_id += 1
    notify_send(f"wd:{url[-7:]}, size={len(data)}")


class AtomicCounter:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._val = 0
    
    def incr(self) -> int:
        with self._lock:
            self._val += 1
            return self._val - 1

def i_was_first(ctr: AtomicCounter):
    return ctr.incr() == 0

def find_urls_in_index_html(content: str) -> list[str]:
    lines = content.splitlines()
    ret = []
    import re
    pattern = r'<a href="([^"]*)"'
    matches = re.findall(pattern, content)
    return [x for x in matches if not x.startswith('?')]
    # run_shell(f"")
    # for line in lines:
    #     line = line.strip()
    #     if line.startswith("<a href"):
    #         ret.append(line.split('"')[1])
    return ret

class NginxFS(Fuse):

    def __init__(self, root_url: str, **kwargs):
        super().__init__(**kwargs)

        self.root_url = root_url
        self.free_after_read = True

        # for anyone modifying any of dicts below (not not sub-dicts!)
        self.fs_lock = threading.Lock()

        self.outputs : dict[str, dict[int, bytes]] = dict()
        # modifying self.outputs[x] require acquiring self.locks[x]
        self.locks   : dict[str, threading.Lock] = dict()

        self.ctrs : dict[str, AtomicCounter] = dict()
        self.threads : dict[str, threading.Thread] = dict()
        

    def getattr(self, path: str):
        if not path.startswith("/"):
            notify_send("getattr: path doesn't start with /")
            return -errno.ENOENT

        # NOTE: (true for nginx) with Accept-Encoding != none, the response
        # does not necessarily contains Content-Length for regular files.
        response = requests.head(f"{self.root_url}/{path}", headers=
                                 {"Connection": 'close',
                                  'Accept-Encoding': 'none',
                                  })
        
        if not response.ok:
            # if all([x not in path for x in ["git", "autorun", "Trash", "xdg"]]):
            #     notify_send(f"getattr bad response: {path}")
            return -errno.ENOENT
        
        if (content_type := response.headers.get("Content-Type", None)) is None:
            notify_send("readdir: No 'Content-Type' in response.headers")
            return
        
        # TODO - it might be nginx-specific.
        is_dir = ("text/html" in content_type) and (response.headers.get("Last-Modified", None) is None)

        st = fuse.Stat()
        if is_dir:
            st.st_mode = stat.S_IFDIR | 0o755
            st.st_nlink = 2
        else:
            if (content_length := response.headers.get("Content-Length", None)) is None:
                notify_send(f"readdir {path}: No 'Content-Length' in response.headers")
                return
            content_length = int(content_length)
            
            st.st_mode = stat.S_IFREG | 0o666
            st.st_nlink = 1
            st.st_size = content_length
        return st

    def readdir(self, path, offset):
        yield fuse.Direntry(".")
        yield fuse.Direntry("..")

        try:
            url = f"{self.root_url}/{path or ''}"
            response = requests.get(url)
            
            if not response.ok:
                notify_send(f"readdir: bad response")
                return
            
            if (content_type := response.headers.get("Content-Type", None)) is None:
                notify_send("readdir: No 'Content-Type' in response.headers")
                return
            
            if "text/html" not in content_type:
                notify_send("No 'text/html' in 'Content-Type'")
                return
            
            text = response.content.decode("utf-8")
            urls = find_urls_in_index_html(text)

            for r in urls:
                if r.endswith("/"): r = r[:-1]
                yield fuse.Direntry(r)

        except Exception as e:
            notify_send(f"ERR: {e}")
            return

    def open(self, path, flags):
        _assert(path not in self.ctrs)
        with self.fs_lock:
            self.ctrs[path] = AtomicCounter()
    
    def release(self, path, flags):
        _assert(path in self.ctrs and path in self.threads)
        self.threads[path].join()
        
        # if self.free_after_read:
        #     with self.locks[path]:
        #         _assert(not len(self.outputs[path]))
        with self.fs_lock:
            del self.threads[path]
            del self.ctrs[path]
            del self.locks[path]
            del self.outputs[path]

    def read(self, path, size, offset):
        notify_send(f"r:{path[-5:]} size={size} off={offset} starts")
        _assert(0 == size % (_page_size := 4096), f"read size: {size}/4096 = {size % 4096}")
        _assert(self.ctrs.get(path, None) is not None, "self.ctrs not yet ready")
        
        if i_was_first(self.ctrs[path]):
            # i am the first thread to read from that path.
            # prepare structures and spawn dload_worker.
            with self.fs_lock:
                self.locks[path] = threading.Lock()
                self.outputs[path] = dict()
            
            x = threading.Thread(target=dload_worker, kwargs={
                'url': f"{self.root_url}/{path}",
                'output': self.outputs[path],
                'o_lock': self.locks[path],
                'chunk_size': 4096,
            })
            
            x.start()
            with self.fs_lock:
                self.threads[path] = x

        # wait for first thread to create self.outputs.
        # NOTE: it is important that the first thread creates 'lock' first, then 'output'.
        while (output := self.outputs.get(path, None)) is None: time.sleep(0.001)

        my_file_lock = self.locks[path]
        
        # wait for my data chunk to be issued by worker thread.
        while True:
            with my_file_lock:
                range_factory = lambda: range(offset, offset + size, _page_size)
                for off in range_factory():
                    if output.get(off, None) is None: break
                else:
                    # none of loop iterations 'break'ed - full data chunk is ready.
                    data = b''.join([output[off] for off in range_factory()])
                    break
            time.sleep(0.1)
        
        if self.free_after_read:
            with my_file_lock:
                for off in range_factory():
                    del output[off]
        
        notify_send(f"read: returning data of size={len(data)}, cksum={sum(data)}")
        return data

def main(root_url: str):
    
    server = NginxFS(
        # Application params.
        root_url=root_url,

        # FUSE params.
        version="%prog " + fuse.__version__,
        usage=Fuse.fusage,
        dash_s_do='setsingle',
    )

    server.parse(errex=1)
    server.main()


if __name__ == '__main__':
    import sys
    if len(sys.argv) != 3:
        print(f"usage: {__file__} <root url> <mountpoint dir>")
        exit(1)

    # NOTE: 'server.main' parses sys.argv on it's own - make sure all application params are popped.
    root_url = sys.argv.pop(1)

    main(root_url=root_url)


def _test():
    # FIXME
    #
    # server.open("inotifywait", None)
    # for off in (0, 4096, 2*4096):
    #     _ = server.read("inotifywait", size=4096, offset=off)
    pass