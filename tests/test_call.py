"""Engine and websocket tests. Claude is off, so only the instant answer-word path runs."""
import json, os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["MISTRI_CLAUDE"] = "off"
os.environ["MISTRI_DATA"] = os.path.join(tempfile.mkdtemp(), "learners.json")
import pytest
from fastapi.testclient import TestClient
from mistri.engine import Call, lik
from mistri.server import app, KB


def new_call():
    c = Call(KB); c.start()
    c.handle({"intent": "answer", "device": True})
    c.handle({"intent": "answer", "model": True})
    return c


@pytest.mark.parametrize("true", [f["id"] for f in KB["faults"]])
def test_truthful_caller_reaches_the_right_fix(true):
    """A simulated caller answers every question the way the true fault makes most likely."""
    c = new_call()
    c.handle({"intent": "answer", "observations": [{"test": "warning_light",
              "answer": max(["orange", "none"], key=lambda a: lik(c.T["warning_light"], a, true))}]})
    for _ in range(10):
        if c.phase != "diagnose":
            break
        t = c.T[c.q]
        a = max([x["id"] for x in t["answers"]], key=lambda a: lik(t, c.outcome(t, a), true))
        c.handle({"intent": "answer", "observations": [{"test": t["id"], "answer": a}]})
    assert c.phase == "repair" and c.fault == true


def test_failed_fix_goes_back_to_diagnosis():
    c = new_call()
    c.handle({"intent": "answer", "observations": [{"test": "warning_light", "answer": "orange"}, {"test": "paper_stuck", "answer": "yes"}]})
    assert c.fault == "paper_jam"
    r = c.handle({"intent": "not_fixed"})
    assert c.phase == "diagnose" and "paper_jam" in c.ruled_out and r.startswith("Okay, so it's not that.")


def run(lines, caller):
    with TestClient(app).websocket_connect("/relay") as ws:
        ws.send_text(json.dumps({"type": "setup", "from": caller}))
        out = []
        for l in lines:
            ws.send_text(json.dumps({"type": "prompt", "voicePrompt": l, "last": True}))
            out.append(json.loads(ws.receive_text())["token"])
    return out


def test_demo_two_calls():
    """The demo: first call fully guided; second call the caller leads and the skill is proven."""
    r = run(["I need help fixing a printer.", "Canon PIXMA G3010.", "It won't print. There's an orange light on.",
             "Uh, the bottom one?", "Normal.", "It's printing."], caller="test-demo")
    assert r[0].startswith("Sure.") and r[1] == "Got it. What's it doing?" and "Which one's lit?" in r[2]
    assert r[3].startswith("That's 4 by 6") and r[4].startswith("Found it.") and r[5].startswith("Nice. Remember this one")
    r = run(["Hi, it's the printer again. Orange light.", "LTR", "It's printing."], caller="test-demo")
    assert r[0] == "Same light as last time. What do you think it is?"


def test_twiml():
    x = TestClient(app).post("/twiml", headers={"host": "demo.example.com"}).text
    assert 'url="wss://demo.example.com/relay"' in x and "ConversationRelay" in x
