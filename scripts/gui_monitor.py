# /// script
# requires-python = ">=3.11"
# dependencies = ["pyobjc-framework-Quartz", "pillow"]
# ///
"""Watch the Complete Test GUI from another room: a view-only web page with the GUI's windows (macOS).

Run it beside the GUI, on the Mac that runs the test, and open the printed address in a browser on the same network:

    uv run scripts/gui_monitor.py

A process of its own: it takes a picture of the GUI's windows every --every seconds while someone is watching, and
never touches the GUI or REAPER. Nothing can be clicked or changed from the page. The address holds a random key (new
every start); without it the server answers 404. Plain http: anyone on the network who has the address can watch.
The windows may be behind others but not minimised, and the Mac must not be locked (the display is kept awake while
this runs). The terminal needs System Settings > Privacy & Security > Screen Recording.
"""
import argparse
import io
import os
import secrets
import socket
import subprocess
import tempfile
import threading
import time
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import Quartz
from PIL import Image

PAGE = """<!doctype html><meta name=viewport content="width=device-width,initial-scale=1"><title>Test monitor</title>
<style>body{margin:0;background:#111;color:#ddd;font:15px system-ui}#s{padding:8px 12px;position:sticky;top:0;background:#111}
.bad{color:#ff6b6b}img{display:block;width:100%;margin-bottom:8px}</style><div id=s>connecting</div><div id=w></div><script>
const s=document.getElementById('s'),w=document.getElementById('w');let last=0;
const say=(t,bad)=>{s.textContent=t;s.className=bad?'bad':''};
async function tick(){try{
  const r=await fetch('KEY/list',{signal:AbortSignal.timeout(8000)});if(!r.ok)throw 0;const d=await r.json();
  while(w.children.length>d.n)w.lastChild.remove();while(w.children.length<d.n)w.append(new Image());
  if(d.error){say(d.error,true)}else{
    for(const[i,m]of[...w.children].entries())if(m.dataset.v!=d.v[i]){   // only a window that changed, one at a time
      const b=await fetch('KEY/'+i+'.webp?'+d.v[i],{signal:AbortSignal.timeout(20000)});if(!b.ok)throw 0;
      const u=URL.createObjectURL(await b.blob());m.onload=()=>URL.revokeObjectURL(u);m.src=u;m.dataset.v=d.v[i]}
    last=Date.now();say('Live · picture taken '+new Date(d.t*1000).toLocaleTimeString()+(d.age>15?' (OLD)':''),d.age>15)}
}catch(e){say('No contact with the test Mac'+(last?' since '+new Date(last).toLocaleTimeString():''),true)}
setTimeout(tick,EVERY)}tick()</script>"""


def gui_windows(match: str) -> list[int]:
    """Window ids of the Python process whose command line holds `match`, largest first (never another app's)."""
    pids = {int(p) for p in subprocess.run(["pgrep", "-f", match], capture_output=True, text=True).stdout.split()}
    info = Quartz.CGWindowListCopyWindowInfo(
        Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListExcludeDesktopElements, Quartz.kCGNullWindowID)
    area = lambda d: d["kCGWindowBounds"]["Width"] * d["kCGWindowBounds"]["Height"]  # noqa: E731
    mine = [d for d in info if d["kCGWindowOwnerPID"] in pids
            and d.get("kCGWindowOwnerName", "").lower().startswith("python") and area(d) > 10_000]
    return [int(d["kCGWindowNumber"]) for d in sorted(mine, key=area, reverse=True)]


def grab(window: int, width: int) -> bytes:
    """The window as WebP at most `width` pixels wide: a Retina JPEG is ~800 KB, this ~90 (report entry 284)."""
    with tempfile.TemporaryDirectory() as tmp:
        f = os.path.join(tmp, "w.jpg")
        subprocess.run(["screencapture", "-x", "-o", "-t", "jpg", f"-l{window}", f], capture_output=True, timeout=10)
        if not os.path.exists(f):
            return b""
        im = Image.open(f).convert("RGB")
        im.thumbnail((width, 10_000), Image.LANCZOS)
        out = io.BytesIO()
        im.save(out, "WEBP", quality=50, method=2)
        return out.getvalue()


class State:
    pics: list[bytes] = []
    taken, asked, error = 0.0, 0.0, "waiting for the first picture"


def capture(every: float, match: str, width: int) -> None:
    while True:
        if time.time() - State.asked < max(10.0, 3 * every):     # nobody watching: no work during the test
            try:
                ids = gui_windows(match)
                pics = [j for j in (grab(i, width) for i in ids) if j]
                if pics:
                    State.pics, State.taken, State.error = pics, time.time(), ""
                else:
                    State.error = ("The GUI is not running, or its window is minimised" if not ids else
                                   "Could not photograph the GUI: Screen Recording permission, or the Mac is locked")
            except Exception as e:                                # the page must say so, not go silently stale
                State.error = f"Capture failed: {e}"
        time.sleep(every)


def handler(key: str, every: float):
    import json

    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            parts = self.path.split("?")[0].strip("/").split("/")
            if not secrets.compare_digest(parts[0].encode(), key.encode()):
                return self.send_error(404)
            State.asked = time.time()
            pics = State.pics
            if len(parts) == 1:
                body = PAGE.replace("KEY", "/" + key).replace("EVERY", str(int(every * 1000))).encode()
                kind = "text/html; charset=utf-8"
            elif parts[1] == "list":
                kind, body = "application/json", json.dumps(dict(
                    n=len(pics), v=[zlib.crc32(j) for j in pics], t=State.taken, age=time.time() - State.taken, error=State.error)).encode()
            elif parts[1].removesuffix(".webp").isdecimal() and int(parts[1].removesuffix(".webp")) < len(pics):
                kind, body = "image/webp", pics[int(parts[1].removesuffix(".webp"))]
            else:
                return self.send_error(404)
            try:
                self.send_response(200)
                self.send_header("Content-Type", kind)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):       # the browser gave up on a slow picture
                pass

        def log_message(self, *a):
            pass

    return H


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", type=int, default=8090)
    p.add_argument("--host", default="0.0.0.0", help="0.0.0.0: the network; 127.0.0.1: this Mac only")
    p.add_argument("--every", type=float, default=1.0, help="seconds between pictures")
    p.add_argument("--width", type=int, default=1600, help="picture width in pixels: lower is faster, e.g. 1200")
    p.add_argument("--match", default="Complete_Test_Gui.py", help="the GUI is the process with this in its command line")
    a = p.parse_args()
    key = secrets.token_urlsafe(9)
    server = ThreadingHTTPServer((a.host, a.port), handler(key, a.every))
    subprocess.Popen(["caffeinate", "-di", "-w", str(os.getpid())])   # a sleeping display has no picture
    if not Quartz.CGPreflightScreenCaptureAccess():
        print("Screen Recording is not allowed for this terminal: System Settings > Privacy & Security > Screen Recording")
    threading.Thread(target=capture, args=(a.every, a.match, a.width), daemon=True).start()
    ips = [a.host] if a.host != "0.0.0.0" else [i for i in (subprocess.run(
        ["ipconfig", "getifaddr", n], capture_output=True, text=True).stdout.strip() for n in ("en0", "en1", "en5", "en6", "en7")) if i]
    names = ips + ([socket.gethostname()] if a.host == "0.0.0.0" else [])
    print("Open on a device on the same network (Ctrl+C stops it):", flush=True)
    for n in names:
        print(f"  http://{n}:{a.port}/{key}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
