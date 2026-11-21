"""Mistri server.

Caller speech -> text (the browser's speech recognition, or Twilio ConversationRelay)
  -> /relay websocket -> understand (answer words or Claude) -> engine (Bayes, info gain, caller memory) -> reply text
Every step, including Mistri's thinking, is pushed to /events, which the dashboard at / draws.
Callers are remembered in data/learners.json, keyed by caller ID (phone number, or a browser id).
"""
import asyncio
import json
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from html import escape
from pathlib import Path

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response

from .engine import Call, load_kb, new_learner
from .understand import claude, understand

ROOT = Path(__file__).resolve().parent.parent
KB = load_kb(ROOT / os.environ.get("MISTRI_KB", "kb/printer.json"))
STORE = Path(os.environ.get("MISTRI_DATA", ROOT / "data" / "learners.json"))
_lock = threading.Lock()


def learners() -> dict:
    try:
        return json.loads(STORE.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_learner(caller: str, learner: dict):
    with _lock:
        all_ = learners()
        all_[caller] = learner
        STORE.parent.mkdir(parents=True, exist_ok=True)
        STORE.write_text(json.dumps(all_, indent=1))


@asynccontextmanager
async def lifespan(app):
    if claude.enabled:  # the first Claude request of a session is slow; pay for it before anyone calls
        asyncio.create_task(warm())
    yield


async def warm():
    try:
        t0 = time.perf_counter()
        await claude.ask("Caller said: 'hello'. Reply with intent 'other'.")
        print(f"claude warm in {time.perf_counter() - t0:.1f}s")
    except Exception as e:
        print("claude warm-up failed:", repr(e))


app = FastAPI(title="Mistri", lifespan=lifespan)


class Hub:
    """Fans events out to every open dashboard; keeps the current call so a refreshed page catches up."""

    def __init__(self):
        self.subs: set[WebSocket] = set()
        self.history: list[dict] = []
        self.current = None

    async def publish(self, ev: dict):
        if ev["type"] == "call_start":
            self.history, self.current = [], ev["call_id"]
        elif ev.get("call_id") != self.current:
            return  # a stale call hanging up must not touch the new one
        self.history.append(ev)
        print("event", json.dumps(ev, ensure_ascii=False), flush=True)
        for ws in list(self.subs):
            try:
                await ws.send_text(json.dumps(ev))
            except Exception:
                self.subs.discard(ws)


hub = Hub()


class Session:
    def __init__(self, call_id: str, source: str, caller: str):
        self.id, self.source, self.caller = call_id, source, caller
        self.t0 = time.time()
        self.learner = learners().get(caller) or new_learner()
        self.call = Call(KB, self.learner, call_no=len(self.learner["calls"]) + 1)

    async def ev(self, type_, **kw):
        await hub.publish({"type": type_, "call_id": self.id, "t": round(time.time() - self.t0, 2), **kw})

    async def flush(self):
        for e in self.call.drain():
            t = e.pop("type")
            if t == "summary":  # time the call and remember it
                e["secs"] = round(time.time() - self.t0)
                self.learner["calls"].append(e)
                save_learner(self.caller, self.learner)
            await self.ev(t, **e)

    async def start(self):
        await self.ev("call_start", call=f"Call {self.call.call_no}", source=self.source, returning=self.call.returning)
        await self.ev("history", calls=self.learner["calls"])
        greeting = self.call.start()
        await self.flush()
        await self.ev("turn", speaker="ai", text=greeting, phase=self.call.phase)

    async def on_user(self, text: str) -> str:
        t0 = time.perf_counter()
        await self.ev("turn", speaker="caller", text=text)
        u = await understand(self.call, text, on_thinking=lambda: self.ev("thinking"))
        await self.ev("understood", **u)
        reply = self.call.handle(u)
        await self.flush()
        save_learner(self.caller, self.learner)
        await self.ev("turn", speaker="ai", text=reply, phase=self.call.phase,
                      server_ms=int((time.perf_counter() - t0) * 1000), understood_by=u["by"])
        return reply


@app.api_route("/twiml", methods=["GET", "POST"])
async def twiml(request: Request):
    """Point a Twilio number's 'A call comes in' webhook here."""
    host = os.environ.get("PUBLIC_HOST") or request.headers.get("x-forwarded-host") or request.headers["host"]
    attrs = {
        "url": f"wss://{host}/relay",
        "welcomeGreeting": KB["greeting"],
        "language": os.environ.get("LANGUAGE", "en-US"),
        "hints": KB.get("hints", ""),
        "ttsProvider": os.environ.get("TTS_PROVIDER", ""),
        "voice": os.environ.get("TTS_VOICE", ""),
        "transcriptionProvider": os.environ.get("STT_PROVIDER", ""),
    }
    a = " ".join(f'{k}="{escape(v)}"' for k, v in attrs.items() if v)
    xml = f'<?xml version="1.0" encoding="UTF-8"?><Response><Connect><ConversationRelay {a} /></Connect></Response>'
    return Response(xml, media_type="text/xml")


@app.websocket("/relay")
async def relay(ws: WebSocket):
    """Speaks Twilio ConversationRelay's websocket protocol. The dashboard's Start call uses it too."""
    await ws.accept()
    s = None
    try:
        while True:
            msg = json.loads(await ws.receive_text())
            kind = msg.get("type")
            if kind == "setup":
                sid = msg.get("callSid")
                s = Session(sid or f"web-{uuid.uuid4().hex[:6]}", "phone" if sid else "browser",
                            caller=msg.get("from") or "anonymous")
                await s.start()
            elif kind == "prompt" and s and msg.get("last", True):
                reply = await s.on_user(msg.get("voicePrompt", ""))
                await ws.send_text(json.dumps({"type": "text", "token": reply, "last": True}))
                if s.call.phase in ("done", "handoff") and s.source == "phone":
                    await asyncio.sleep(6)  # let Twilio finish speaking, then hang up
                    await ws.send_text(json.dumps({"type": "end"}))
            elif kind == "interrupt" and s:
                await s.ev("interrupt", heard=msg.get("utteranceUntilInterrupt", ""))
            elif kind == "error":
                print("twilio error:", msg)
    except WebSocketDisconnect:
        pass
    finally:
        if s:
            await s.ev("call_end")


@app.websocket("/events")
async def events(ws: WebSocket):
    await ws.accept()
    for ev in hub.history:
        await ws.send_text(json.dumps({**ev, "replay": True}))
    hub.subs.add(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        hub.subs.discard(ws)


@app.post("/forget")
async def forget(caller: str):
    """Wipe one caller's memory, to run the two-call demo again."""
    with _lock:
        all_ = learners()
        all_.pop(caller, None)
        STORE.parent.mkdir(parents=True, exist_ok=True)
        STORE.write_text(json.dumps(all_, indent=1))
    return {"ok": True}


@app.get("/")
async def dashboard():
    return FileResponse(ROOT / "static" / "dashboard.html")


@app.get("/health")
async def health():
    return {"ok": True, "claude": claude.bin if claude.enabled else None, "model": os.environ.get("MISTRI_MODEL", "haiku"),
            "faults": len(KB["faults"]), "tests": len(KB["tests"])}
