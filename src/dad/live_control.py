"""Start and end recorded sections of a running live detector from another script.

The live GUI (scripts/live_gui.py) listens on 127.0.0.1, port 47800 by default. With
its sections set to "on start/end calls", another program on the same computer marks
what to save:

    from dad.live_control import start_section, end_section

    n = start_section()      # saving starts here; returns the section's number
    ...                      # fly the drone past, play the sound, ...
    end_section()            # saving stops; the CSV now holds section n's columns

From a shell, or a script in another language, the same as a command:

    uv run dad section start
    uv run dad section end
    uv run dad section status

A start or end that succeeded prints a line saying so (`report=None` silences it, or
pass your own function). To repeat sections of a fixed length with a gap between them:

    from dad.live_control import run_sections

    run_sections(20, gap_s=5, count=10)              # or: uv run dad section cycle --length 20 --gap 5

Each section becomes its own WAV and its own group of columns in the session's one CSV
(dad.live.CalledSectionRecorder). A call fails with LiveControlError, and changes
nothing, when no detector is listening, when it is not detecting, when its sections are
not set to start/end calls, when start comes with a section already open, or end with
none open.

The protocol, for other languages: connect over TCP, send one JSON object on one line,
{"cmd": "start"}, "end" or "status", and read one JSON line back: {"ok": true, ...}, or
{"ok": false, "error": "..."}. Only connections from this computer are accepted.
"""
from __future__ import annotations

import json
import socket
import socketserver
import threading
import time
from typing import Callable

DEFAULT_PORT = 47800
HOST = "127.0.0.1"                     # this computer only
_POLL_S = 0.25                         # how often run_sections asks; each ask is one short-lived connection
Report = Callable[[str, dict], None]   # report(cmd, reply), after each start or end that succeeded


class LiveControlError(RuntimeError):
    pass


# --------------------------------------------------------------------- client

def send(cmd: str, port: int = DEFAULT_PORT, timeout: float = 5.0, **fields) -> dict:
    """One request, one reply. Raises LiveControlError if nothing answers or the reply is not ok."""
    try:
        with socket.create_connection((HOST, port), timeout=timeout) as s:
            s.sendall((json.dumps(dict(cmd=cmd, **fields)) + "\n").encode())
            buf = b""
            while not buf.endswith(b"\n"):
                chunk = s.recv(4096)
                if not chunk:
                    break
                buf += chunk
    except (ConnectionRefusedError, socket.timeout, OSError) as e:
        raise LiveControlError(f"no live detector answered on {HOST}:{port} ({e}). Is scripts/live_gui.py "
                               "running, with this control port?") from e
    try:
        reply = json.loads(buf.decode())
    except ValueError as e:
        raise LiveControlError(f"unreadable reply from {HOST}:{port}: {buf[:200]!r}") from e
    if not reply.get("ok"):
        raise LiveControlError(reply.get("error", "refused"))
    return reply


def reply_message(cmd: str, reply: dict) -> str:
    """The log line for a start or end call that succeeded."""
    t = reply.get("t_s")
    at = "" if t is None else f" at t = {t:.2f} s"
    return f"section {reply['section']} {'started' if cmd == 'start' else 'ended'}{at}"


def print_reply(cmd: str, reply: dict) -> None:
    print(f"{time.strftime('%H:%M:%S')}  {reply_message(cmd, reply)}", flush=True)


def _call(cmd: str, port: int, timeout: float, report: Report | None) -> dict:
    reply = send(cmd, port, timeout)
    if report is not None:
        report(cmd, reply)
    return reply


def start_section(port: int = DEFAULT_PORT, timeout: float = 5.0, report: Report | None = print_reply) -> int:
    """Start saving a new section; returns its number (1, 2, ...)."""
    return int(_call("start", port, timeout, report)["section"])


def end_section(port: int = DEFAULT_PORT, timeout: float = 5.0, report: Report | None = print_reply) -> int:
    """End the open section; returns its number. Its data is in the CSV when this returns."""
    return int(_call("end", port, timeout, report)["section"])


def status(port: int = DEFAULT_PORT, timeout: float = 5.0) -> dict:
    """{"detecting": bool, "sections": "calls" | seconds | None, "open": section or None, "t_s": ...}"""
    return send("status", port, timeout)


def _wait_audio(until_s: float, port: int, timeout: float, stop: threading.Event, section: int | None = None) -> None:
    """Wait until the detector's audio clock reaches `until_s`, or `stop` is set. `section`: it must stay open."""
    while not stop.is_set():
        s = status(port, timeout)
        if not s["detecting"]:
            raise LiveControlError("the detector stopped detecting")
        if section is not None and s["open"] != section:
            raise LiveControlError(f"section {section} was ended by another call")
        left = until_s - s["t_s"]
        if left <= 0:
            return
        stop.wait(min(left, _POLL_S))


def run_sections(length_s: float, gap_s: float = 0.0, count: int | None = None, port: int = DEFAULT_PORT,
                 timeout: float = 5.0, stop: threading.Event | None = None,
                 report: Report | None = print_reply) -> int:
    """Start and end sections over and over: `length_s` saved, `gap_s` not saved, repeated.

    Blocks until `count` sections are done (None: until `stop` is set; run it in a thread
    to do something else meanwhile). Setting `stop` ends a section that is open, so none
    is left half done. Returns the number of sections finished. Raises LiveControlError
    if a call is refused or the detector stops; the section is then not ended here.

    Length and gap are in the detector's audio seconds, not the PC's clock. A section
    comes out up to a few audio blocks (some 10-60 ms) longer than `length_s`.
    """
    if not length_s > 0 or not gap_s >= 0 or (count is not None and count < 1):
        raise ValueError(f"need length_s > 0, gap_s >= 0, count >= 1 or None; got {length_s}, {gap_s}, {count}")
    stop = stop or threading.Event()
    done = 0
    while not stop.is_set() and (count is None or done < count):
        opened = _call("start", port, timeout, report)
        _wait_audio(opened["t_s"] + length_s, port, timeout, stop, opened["section"])
        closed = _call("end", port, timeout, report)
        done += 1
        if gap_s > 0 and (count is None or done < count):
            _wait_audio(closed["t_s"] + gap_s, port, timeout, stop)
    return done


# --------------------------------------------------------------------- server

class ControlServer:
    """Answers requests in a background thread with `handler(request) -> reply`."""

    def __init__(self, handler: Callable[[dict], dict], port: int = DEFAULT_PORT) -> None:
        outer = self

        class Handle(socketserver.StreamRequestHandler):
            def handle(self):
                line = self.rfile.readline(65536)
                try:
                    reply = outer.handler(json.loads(line.decode()))
                except Exception as e:                       # a bad request gets an answer, not silence
                    reply = dict(ok=False, error=f"{type(e).__name__}: {e}")
                self.wfile.write((json.dumps(reply) + "\n").encode())

        class Server(socketserver.ThreadingTCPServer):
            daemon_threads = True
            allow_reuse_address = False                      # Windows: a second detector must not share the port

        self.handler = handler
        self.server = Server((HOST, port), Handle)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def start(self) -> "ControlServer":
        self.thread.start()
        return self

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
