"""Turn one caller line into {intent, observations, device, model, gloss} for the engine.

1. Vocabulary: an instant check against the knowledge base's answer words, used only
   when the line answers the open question unambiguously (one answer per test, no negation).
2. Claude (headless Claude Code, `claude -p`): everything else, including all "why" answers.
   No API key needed; it uses your logged-in Claude Code. One process is kept warm.
"""
import asyncio
import json
import os
import re
import shutil
import tempfile
import time

TIMEOUT = float(os.environ.get("MISTRI_CLAUDE_TIMEOUT", "15"))
MODEL = os.environ.get("MISTRI_MODEL", "haiku")
NEG = {"no", "not", "dont", "isnt", "wont", "cant", "never", "nothing", "didnt", "doesnt", "without", "aint", "arent"}
REPEAT = ["what", "sorry", "say again", "repeat", "pardon", "come again", "huh"]
DONT_KNOW = ["don't know", "dont know", "not sure", "no idea", "can't tell", "cant tell"]
FIXED = ["printing", "it prints", "it's working", "its working", "works now", "working now", "came out"]
NOT_FIXED = ["still not", "not printing", "didn't work", "didnt work", "still won't", "still wont", "nothing happened", "still orange", "still blinking"]
DONE = ["done", "did it", "lit", "on now", "pressed", "finished", "got it", "closed it", "it's in", "its in", "clicked", "swapped", "changed it", "pulled it"]
GENERIC = {"yes", "yeah", "yep", "no", "nope", "on", "off", "lit", "none", "nothing", "dark", "up", "down", "open", "closed", "clear", "stuck"}
INTENTS =["answer", "done", "stuck", "fixed", "not_fixed", "dont_know", "repeat", "other"]
SYSTEM = ("You read one line from a phone caller who is being coached through a printer repair. The line comes from "
          "speech-to-text and may contain errors; read it charitably. Reply with one JSON object and nothing else: "
          '{"intent": one of ' + "|".join(INTENTS) + ', "observations": [{"test": id, "answer": id}], '
          '"device": true|false, "model": true|false, "gloss": "one short sentence on how you read the line"}. '
          "Only use test and answer ids listed in the prompt. Only report what the caller actually said; "
          "never infer an answer they didn't give. Write the gloss like a technician's note, e.g. "
          "'“The bottom one” of A4, LTR, 4 by 6 means 4 by 6' or '“Normal” means plain A4 sheets'.")


def norm(text: str) -> str:
    t = text.lower().replace("×", "x").replace("’", "'")
    t = re.sub(r"[^a-z0-9' ]+", " ", t).replace("'", "")
    return re.sub(r"\s+", " ", t).strip()


def hit(text: str, kw: str, neg_ok=False) -> bool:
    """Whole-word phrase match, rejected if a negation sits in the 3 words before it."""
    w, k = norm(text).split(), norm(kw).split()
    for i in range(len(w) - len(k) + 1):
        if w[i:i + len(k)] == k and (neg_ok or k[0] in NEG or not NEG & set(w[max(0, i - 3):i])):
            return True
    return False


def vocabulary(call, text: str):
    """Instant, offline understanding. Returns None when not sure, so Claude decides."""
    if len(norm(text).split()) <= 4 and any(hit(text, k, True) for k in REPEAT):
        return {"intent": "repeat"}
    if call.phase == "device":
        return {"intent": "answer", "device": True} if any(hit(text, k) for k in call.kb["device_kind"]["keywords"]) else None
    if call.phase == "model":
        return {"intent": "answer", "model": True} if any(hit(text, k, True) for k in call.kb["model_keywords"]) else None
    if call.phase in ("diagnose", "lead"):
        asked = [call.T[call.q]] if call.q and call.q not in call.ev else []
        tests = asked + [t for t in call.kb["tests"] if t["id"] not in call.ev and t not in asked
                         and (t.get("volunteer") or call.phase == "lead")]
        obs = []
        for t in tests:
            # a bare "yes"/"on" only answers a question that asked about exactly one thing
            generic_ok = asked == [t]
            matched = [a["id"] for a in t["answers"]
                       if any(hit(text, k) and (generic_ok or norm(k) not in GENERIC) for k in a["keywords"])]
            if len(matched) == 1:
                obs.append({"test": t["id"], "answer": matched[0]})
        if obs and (call.phase == "diagnose" or len(norm(text).split()) <= 4):
            return {"intent": "answer", "observations": obs}
        if call.phase == "lead":
            return None
        if any(hit(text, k, True) for k in DONT_KNOW):
            return {"intent": "dont_know"}
        return None
    if call.phase == "repair":
        if any(hit(text, k, True) for k in NOT_FIXED):
            return {"intent": "not_fixed"}
        if any(hit(text, k) for k in FIXED):
            return {"intent": "fixed"}
        if "?" not in text and not re.search(r"\b(which|where|how|what|two|both)\b", norm(text)) and any(hit(text, k) for k in DONE):
            return {"intent": "done"}
    return None


def prompt(call, text: str) -> str:
    p = [f"Coach just said: {call.last!r}", f"Caller said: {text!r}", ""]
    if call.phase == "device":
        p.append("Set device=true and intent 'answer' if the caller says the thing to fix is a printer. Otherwise intent 'other'.")
    elif call.phase == "model":
        p.append(f"The only model on file is the {call.kb['device']} (also said as 'G3010', 'Pixma', 'G three thousand ten'). "
                 "Set model=true and intent 'answer' if the caller names that model, even garbled by speech-to-text. "
                 "Otherwise intent 'other'.")
    elif call.phase in ("diagnose", "lead"):
        p.append("Tests the caller might be answering (test_id: question -> answer ids):")
        for t in call.kb["tests"]:
            if t["id"] not in call.ev:
                p.append(f"- {t['id']}: {t['ask']} -> " + ", ".join(f"{a['id']} ({a['label']})" for a in t["answers"]))
        p.append("Known so far: " + (", ".join(f"{k}={v}" for k, v in call.ev.items() if k in call.T) or "nothing"))
        p.append("Return every test the caller clearly answered, including things they volunteered, as observations, "
                 "with intent 'answer'. If they say they don't know or can't check, intent 'dont_know'. "
                 "If they ask to hear it again, 'repeat'. Otherwise 'other' with no observations.")
    elif call.phase == "repair":
        f = call.F[call.fault]
        step = f["steps"][min(call.step, len(f["steps"]) - 1)]["say"]
        p.append(f"The caller is fixing: {f['name']}. Current instruction: {call.fmt(step)!r}")
        p.append("Intent: 'done' if they did the instruction; 'stuck' if they are confused or ask which/where/how; "
                 "'fixed' if they say the printer is printing or working; 'not_fixed' if they did it but it still "
                 "doesn't print; 'repeat' if they want it again; otherwise 'other'.")
    return "\n".join(p)


class Claude:
    """One warm `claude -p` process per request (stream-json in/out), respawned after each use."""

    def __init__(self):
        self.bin = shutil.which("claude") or next((p for p in [os.path.expanduser("~/.local/bin/claude")] if os.path.exists(p)), None)
        self.enabled = bool(self.bin) and os.environ.get("MISTRI_CLAUDE", "on") != "off"
        self.proc = None
        self.lock = asyncio.Lock()

    async def spawn(self):
        self.proc = await asyncio.create_subprocess_exec(
            self.bin, "-p", "--model", MODEL, "--effort", "low", "--input-format", "stream-json",
            "--output-format", "stream-json", "--verbose",
            "--system-prompt", SYSTEM, "--tools", "", "--setting-sources", "", "--strict-mcp-config",
            "--no-session-persistence", cwd=tempfile.gettempdir(), limit=2**24,
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)

    async def ask(self, text: str) -> dict:
        async with self.lock:
            if not self.proc or self.proc.returncode is not None:
                await self.spawn()
            p, self.proc = self.proc, None
            try:
                p.stdin.write((json.dumps({"type": "user", "message": {"role": "user", "content": text}}) + "\n").encode())
                await p.stdin.drain()
                return await asyncio.wait_for(self._result(p), TIMEOUT)
            finally:
                try:
                    p.kill()
                except ProcessLookupError:
                    pass
                await self.spawn()  # next one warms up while the caller is listening

    @staticmethod
    async def _result(p):
        while line := await p.stdout.readline():
            d = json.loads(line)
            # take the answer from the assistant message: the final "result" line comes ~6 s later,
            # after Claude Code writes a post-turn summary we don't need
            for c in d.get("message", {}).get("content", []) if d.get("type") == "assistant" else []:
                if c.get("type") == "text" and "{" in c.get("text", ""):
                    return json.loads(re.search(r"\{.*\}", c["text"], re.S).group(0))
            if d.get("type") == "result":
                if d.get("is_error"):
                    raise RuntimeError(d.get("result"))
                return json.loads(re.search(r"\{.*\}", d["result"], re.S).group(0))
        raise RuntimeError("claude exited without a result")


claude = Claude()


async def understand(call, text: str, on_thinking=None) -> dict:
    t0 = time.perf_counter()
    ms = lambda: int((time.perf_counter() - t0) * 1000)
    u = vocabulary(call, text)
    if u:
        return {**u, "by": "vocabulary", "ms": ms()}
    if claude.enabled:
        if on_thinking:
            await on_thinking()
        try:
            u = await claude.ask(prompt(call, text))
            if u.get("intent") not in INTENTS:
                u = {"intent": "other"}
            return {**u, "by": "claude", "ms": ms()}
        except Exception as e:
            print("claude error:", repr(e))
    return {"intent": "other", "by": "none", "ms": ms()}
