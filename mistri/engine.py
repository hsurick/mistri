"""The diagnosis engine. No script: every number, question and step is computed here.

- Fault odds: Bayes' rule over the likelihood table in the knowledge base (kb/*.json).
- Next question: highest expected information gain per unit of effort for the caller.
- Stops asking once one fault passes the threshold, then gives the fix.
- Remembers the caller (a "learner" dict the server persists). A skill taught last call
  but not yet proven means help fades: Mistri asks what they think first and lets them lead.
  A skill turns proven only when the caller diagnoses and fixes it without instructions.

Every turn also emits "thought" lines: what Mistri noticed and why it chose its next move.
The engine never reads free text; it takes an understanding dict from mistri/understand.py.
"""
import json
import math
from collections import defaultdict
from pathlib import Path

# how much one event moves a skill estimate p toward 1
RATE = {"read_guided": 0.4, "taught": 0.25, "fixed_guided": 0.27, "read_alone": 0.73, "fixed_alone": 0.84}


def load_kb(path) -> dict:
    kb = json.loads(Path(path).read_text())
    for t in kb["tests"]:
        for f in kb["faults"]:
            total = sum(lik(t, o, f["id"]) for o in outcomes(t))
            assert abs(total - 1) < 1e-6, f"{t['id']}/{f['id']} likelihoods sum to {total}"
    return kb


def outcomes(t):
    return t.get("outcomes") or [a["id"] for a in t["answers"]]


def lik(t, outcome, fault):
    return t["p"].get(fault, t["p"].get("*"))[outcome]


def entropy(ps):
    return -sum(p * math.log2(p) for p in ps if p > 0)


def normalise(probs):
    s = sum(probs.values())
    return {k: v / s for k, v in probs.items()}


def pct(p):
    return f"{round(p * 100)}%"


def new_learner():
    return {"device": None, "skills": {}, "taught": [], "calls": [], "last_evidence": {}}


class Call:
    def __init__(self, kb: dict, learner=None, call_no: int = 1):
        self.kb = kb
        self.L = learner if learner is not None else new_learner()
        self.call_no = call_no
        self.F = {f["id"]: f for f in kb["faults"]}
        self.T = {t["id"]: t for t in kb["tests"]}
        self.S = {s["id"]: s for s in kb["skills"]}
        self.threshold = kb.get("threshold", 0.9)
        self.probs = normalise({f["id"]: f["prior"] for f in kb["faults"]})
        self.ev = {}             # test id -> answer id
        self.skipped = set()
        self.ruled_out = []
        self.returning = bool(self.L.get("device"))
        self.phase = "diagnose" if self.returning else "device"
        self.lead = False        # caller leads: no questions, no instructions
        self.lead_offered = False
        self.q = None
        self.fault = None
        self.step = 0
        self.questions = 0
        self.instructions = 0
        self.touched = set()     # skills used this call
        self.last = kb["greeting_returning"] if self.returning else kb["greeting"]
        self.out = []

    # ---- plumbing
    def emit(self, type_, **kw):
        self.out.append({"type": type_, **kw})

    def think(self, text):
        self.emit("thought", text=text)

    def title(self, text):
        self.emit("think_title", title=text)

    def drain(self):
        out, self.out = self.out, []
        return out

    def say(self, text):
        self.last = text
        return text

    def start(self):
        if not self.returning:
            self.title("Picking up")
            self.think("Unknown number, no history on file")
            self.think("No device or problem known yet: ask an open question")
        if self.returning:
            self.title("Recognising the caller")
            self.think("Caller ID matches someone who called before")
            self.load_graph()
        return self.last

    def load_graph(self):
        kb = self.kb
        self.emit("device", device=kb["device"], graph=kb["graph_id"])
        self.think(f"{'Printer on file' if self.returning else 'Model'}: {kb['device']}. Loading its fault graph: "
                   f"{len(kb['faults'])} causes, {len([t for t in kb['tests'] if not t.get('symptom')])} tests")
        self.emit("graph",
                  faults=[{"id": f["id"], "name": f["name"], "prior": f["prior"]} for f in kb["faults"]],
                  tests=[{"id": t["id"], "label": t["label"], "links": [k for k in t["p"] if k != "*"]}
                         for t in kb["tests"] if not t.get("symptom")],
                  skills=kb["skills"], skill_edges=kb["skill_edges"], threshold=self.threshold,
                  symptom_title=kb.get("symptom_title", "Symptom"))
        for sid, s in self.L["skills"].items():
            self.emit("skill", skill=sid, p=s["p"], proven=s["proven"])
        if not self.returning:
            self.think("New caller, no skills on record: give full guidance")
            self.emit("learner", caller="new", support="full guidance")
        self.L["device"] = kb["device"]

    # ---- maths
    def outcome(self, t, answer):
        c = t.get("compare")
        if not c or answer in c.get("passthrough", []):
            return answer
        return "match" if self.ev.get(c["test"]) in c["same"][answer] else "mismatch"

    def available(self):
        return [t for t in self.kb["tests"] if t["id"] not in self.ev and t["id"] not in self.skipped
                and not t.get("symptom") and (not t.get("requires") or t["requires"] in self.ev)]

    def gain(self, t, probs=None, ev=None, lookahead=True):
        """Expected entropy drop (bits). A check that unlocks another (size light -> paper type)
        also counts the best follow-up it makes possible: one step of lookahead."""
        probs = probs or self.probs
        ev = ev if ev is not None else self.ev
        unlocks = [u for u in self.kb["tests"] if u.get("requires") == t["id"] and u["id"] not in ev] if lookahead else []
        exp, follow = 0.0, 0.0
        for a in [x["id"] for x in t["answers"]] if unlocks else outcomes(t):
            o = self.outcome(t, a) if unlocks else a
            joint = {f: p * lik(t, o, f) for f, p in probs.items()}
            po = sum(joint.values())
            if po > 0:
                post = {f: v / po for f, v in joint.items()}
                exp += po * entropy(post.values())
                nev = {**ev, t["id"]: a}
                follow += po * max(self.gain_given(u, post, nev) for u in unlocks) if unlocks else 0
        return entropy(probs.values()) - exp + follow

    def gain_given(self, u, probs, ev):
        saved, self.ev = self.ev, ev
        try:
            return self.gain(u, probs, ev, lookahead=False)
        finally:
            self.ev = saved

    def answer(self, t, a):
        return next(x for x in t["answers"] if x["id"] == a)

    def observe(self, t, a, alone=False):
        o = self.outcome(t, a)
        self.probs = normalise({f: p * lik(t, o, f) for f, p in self.probs.items()})
        self.ev[t["id"]] = a
        ans = self.answer(t, a)
        self.emit("evidence", test=t["id"], result=a, outcome=o, label=ans["label"])
        if t.get("compare") and o in ("match", "mismatch"):
            size = self.answer(self.T[t["compare"]["test"]], self.ev[t["compare"]["test"]])["label"]
            self.think(f"Setting says {size}, tray holds {ans['label']}: they {'match' if o == 'match' else 'don’t match'}")
        elif ans.get("note"):
            self.think(ans["note"])
        if t.get("skill"):
            self.bump(t["skill"], "read_alone" if alone else "read_guided", proven=alone)

    def bump(self, skill, kind, proven=False):
        if skill not in self.S or self.S[skill].get("expert"):
            return None
        s = self.L["skills"].setdefault(skill, {"p": 0.0, "proven": False})
        s["p"] = round(s["p"] + (1 - s["p"]) * RATE[kind], 2)
        s["proven"] = s["proven"] or proven
        self.touched.add(skill)
        self.emit("skill", skill=skill, p=s["p"], proven=s["proven"])
        return s

    def fmt(self, text):
        size = self.ev.get("size_light")
        paper = self.ev.get("paper")
        sz = self.answer(self.T["size_light"], size) if size else {}
        pa = self.answer(self.T["paper"], paper) if paper else {}
        return text.format_map(defaultdict(str, size_desc=sz.get("desc", "a different size"),
                                           paper_desc=pa.get("desc", "other paper"), paper=pa.get("paper", "the right size")))

    def odds_line(self, before):
        top = max(self.probs, key=self.probs.get)
        if top == before[0] and len(self.ev) > 1:
            return f"{self.F[top]['name']}: {pct(before[1])} → {pct(self.probs[top])}"
        second = sorted(self.probs, key=self.probs.get)[-2]
        return f"Updated the odds: {self.F[top]['name'].lower()} leads at {pct(self.probs[top])}, {self.F[second]['name'].lower()} {pct(self.probs[second])}"

    # ---- conversation
    def handle(self, u: dict) -> str:
        if u.get("intent") == "repeat":
            self.title("Repeating")
            self.think("Caller asked to hear it again")
            return self.say(self.last)
        return getattr(self, "on_" + self.phase)(u)

    def on_device(self, u):
        self.title("Finding the device")
        if u.get("device"):
            self.think(f"Device: a {self.kb['device_kind']['name']}. Model unknown")
            self.think("Every printer has a different panel, so ask for the model before giving any steps")
            self.phase = "model"
            return self.say("Sure. What's the name on the front of it?")
        self.think("No device named yet")
        return self.say("Hi there. What do you need help fixing?")

    def on_model(self, u):
        self.title("Loading the printer")
        if u.get("model"):
            self.load_graph()
            self.phase = "diagnose"
            return self.say("Got it. What's it doing?")
        self.think("Didn't recognise a model name")
        return self.say("Which model is it? The name's on the front, like PIXMA G3010.")

    def apply(self, u, alone=False):
        """Take the observations in knowledge-base order (size light before paper). Returns how many were new."""
        order = [t["id"] for t in self.kb["tests"]]
        obs = [o for o in sorted(u.get("observations") or [], key=lambda o: order.index(o["test"]) if o.get("test") in order else 99)
               if o.get("test") in self.T and o["test"] not in self.ev and o.get("answer") in [a["id"] for a in self.T[o["test"]]["answers"]]]
        ok = set()
        for o in obs:  # paper type counts only once the size light is known (it may arrive in the same line)
            req = self.T[o["test"]].get("requires")
            if not req or req in self.ev or req in ok:
                ok.add(o["test"])
        got = [(self.T[o["test"]], o["answer"]) for o in obs if o["test"] in ok]
        if got:
            self.think(u.get("gloss") or "Heard: " + ", ".join(self.answer(t, a)["label"].lower() if t.get("symptom") else f"{t['label'].rstrip('?').lower()} {self.answer(t, a)['label']}" for t, a in got))
        for t, a in got:
            self.observe(t, a, alone)
        return got

    def on_diagnose(self, u):
        before = max(self.probs.items(), key=lambda x: x[1])
        self.title("Recognising the caller" if self.returning and not self.lead_offered and not self.ev else
                   "Working out what's wrong" if not self.q else f"Reading the {self.T[self.q]['label'].rstrip('?').lower()}")
        got = self.apply(u)
        if not got and self.q and u.get("intent") == "dont_know":
            self.title("Trying another check")
            self.skipped.add(self.q)
            self.think("Caller can't check that: skip it")
            return self.next_move("No problem, let's try something else. ")
        if not got and self.q:
            self.title("Didn't catch that")
            self.think(u.get("gloss") or "Couldn't match that to the question")
            return self.say(f"{self.kb['fallback']} {self.T[self.q]['ask']}")
        if got:
            self.emit("faults", probs={k: round(v, 3) for k, v in self.probs.items()})
        # returning caller with a skill taught but not proven: fade the help, let them lead
        pending = [s for s in self.L["taught"] if not self.L["skills"].get(self.F[s]["teach"], {}).get("proven")]
        if self.returning and got and pending and not self.lead_offered:
            self.lead_offered = True
            self.phase, self.lead = "lead", True
            skill = self.S[self.F[pending[-1]]["teach"]]["name"]
            same = all(self.L["last_evidence"].get(t["id"]) == a for t, a in got)
            self.think(f"{', '.join(self.answer(t, a)['label'] for t, a in got)} again: same starting odds as last time" if same else "Different start from last time")
            self.think(f"{skill} was taught last call, not proven. Fade the help: ask them first")
            self.emit("learner", caller="returning", support="let them lead")
            return self.say(("Same light as last time. " if same else "") + "What do you think it is?")
        if got and max(self.probs.values()) < self.threshold:
            self.think(self.odds_line(before))
        ack = self.answer(*got[-1]).get("ack", "") if len(got) == 1 else ""
        return self.next_move((ack + " ") if ack else "", before)

    def on_lead(self, u):
        self.title("Letting the caller lead")
        before = max(self.probs.items(), key=lambda x: x[1])
        got = self.apply(u, alone=True)
        if got:
            self.think("Caller read the panel on their own: " + ", ".join(f"{self.T[t['id']]['label'].rstrip('?').lower()} {self.answer(t, a)['label']}" for t, a in got))
            self.emit("faults", probs={k: round(v, 3) for k, v in self.probs.items()})
            if any(self.L["last_evidence"].get(t["id"]) not in (None, a) for t, a in got):
                self.think("Different setting from last time, same rule: they understood it, not just remembered a button")
        top = max(self.probs, key=self.probs.get)
        if self.probs[top] >= self.threshold:
            return self.begin_repair(top, before)
        self.lead = False
        self.phase = "diagnose"
        self.emit("learner", caller="returning", support="guided")
        self.think("Not enough to go on yet: back to guided questions")
        return self.next_move("Okay, let's check. ")

    def next_move(self, prefix="", before=None):
        top = max(self.probs, key=self.probs.get)
        if self.probs[top] >= self.threshold:
            return self.begin_repair(top, before or (top, self.probs[top]))
        scored = [(self.gain(t), t) for t in self.available()]
        if not scored or max(g for g, _ in scored) < 0.01:
            self.phase = "handoff"
            self.emit("handoff")
            self.think("No check left that would narrow it down: hand off to a technician")
            return self.say(prefix + self.kb["handoff"])
        by_gain = sorted(scored, key=lambda x: -x[0])
        by_value = sorted(scored, key=lambda x: -x[0] / x[1].get("effort", 1))
        q = by_value[0][1]
        self.q = q["id"]
        self.questions += 1
        self.think("Scored the next questions: " + ", ".join(f"{t['label'].rstrip('?').lower()} {g:.2f} bits" for g, t in by_gain[:3]))
        if q is not by_gain[0][1] or q.get("why_first"):
            self.think(q.get("why_first") or f"{q['label'].rstrip('?')} is quicker for the caller to check: ask that first")
        self.emit("next_question", chosen=self.q,
                  candidates=[{"test": t["id"], "ask": t["label"], "gain_bits": round(g, 3)} for g, t in by_gain[:3]])
        return self.say(prefix + q["ask"])

    def begin_repair(self, fault, before):
        self.phase, self.fault, self.step, self.q = "repair", fault, 0, None
        f = self.F[fault]
        p = self.probs[fault]
        self.think(f"{f['name']} {pct(before[1]) + ' → ' if before[0] == fault else ''}{pct(p)}, past the {pct(self.threshold)} threshold: stop asking")
        self.emit("stop", fault=fault, p=round(p, 3), threshold=self.threshold)
        if self.lead:
            self.think("They have the fix. Stay out of the way")
            return self.say("Go for it.")
        self.emit("repair_step", step=0, text=self.fmt(f["steps"][0]["say"]))
        self.think("Explain the cause in one sentence, then give the fix")
        self.bump(f["teach"], "taught")
        self.instructions += 1
        return self.say(f"Found it. {self.fmt(f['explain'])} {self.fmt(f['steps'][0]['say'])}")

    def on_repair(self, u):
        steps = self.F[self.fault]["steps"]
        cur = steps[min(self.step, len(steps) - 1)]
        i = u.get("intent")
        self.title(("Checking they did it alone" if self.lead else "Confirming the fix") if i == "fixed" else "Guiding the fix")
        if i == "fixed":
            return self.confirm()
        if i == "not_fixed":
            return self.rule_out()
        if i == "stuck" and not self.lead:
            self.think(u.get("gloss") or "Caller is stuck on the step: describe it more exactly")
            return self.say(cur["help"])
        if i == "done" and not self.lead:
            self.step += 1
            if self.step < len(steps):
                self.instructions += 1
                self.think("Step done: give the next one")
                self.emit("repair_step", step=self.step, text=self.fmt(steps[self.step]["say"]))
                return self.say("Good. " + self.fmt(steps[self.step]["say"]))
            self.think("All steps done: check it worked")
            return self.say("Is it printing now?")
        if self.lead:
            self.think(u.get("gloss") or "Still working on it: wait")
            return self.say("Take your time. Tell me when it prints.")
        return self.say(f"{self.kb['fallback']} {self.fmt(cur['say'])}")

    def confirm(self):
        f = self.F[self.fault]
        self.emit("repair_confirmed", fault=self.fault, step=self.step)
        if self.lead:
            self.think("Repair confirmed, with no instructions from Mistri")
            s = self.bump(f["teach"], "fixed_alone", proven=True)
            self.think(f"Diagnosed and fixed it alone: {self.S[f['teach']]['name'].lower()} proven, p {s['p']:.2f}")
            reply = "Perfect. You didn't need me this time."
        else:
            self.think(f"Repair confirmed: {f['name'].lower()} fixed")
            parts = [f"{self.S[k]['name'].lower()} p {self.bump(k, 'fixed_guided')['p']:.2f}" for k in sorted(self.touched, key=lambda k: k != f["teach"])]
            self.think("Did the fix with step-by-step help: " + "; ".join(parts))
            self.think("Not proven yet: hasn’t done it without help")
            self.think("End with the rule so they can try it alone next time")
            if self.fault not in self.L["taught"]:
                self.L["taught"].append(self.fault)
            reply = f"Nice. Remember this one: {f['rule']} Next time, try that before you call me."
        self.finish()
        return self.say(reply)

    def finish(self):
        self.phase = "done"
        self.L["last_evidence"] = {k: v for k, v in self.ev.items()}
        prev = self.L["calls"][-1] if self.L["calls"] else None
        summary = {"call": f"Call {self.call_no}", "fault": self.fault, "questions": self.questions, "instructions": self.instructions}
        self.emit("summary", **summary)  # the server adds secs and stores it
        if prev and self.lead:
            self.think(f"Last call: {prev['questions']} questions, {prev['instructions']} instructions, {prev.get('secs', '?')} s. "
                       f"This call: {'none' if not (self.questions or self.instructions) else f'{self.questions} questions'}")

    def rule_out(self):
        f = self.fault
        self.ruled_out.append(f)
        self.probs[f] *= 0.02
        self.probs = normalise(self.probs)
        self.think(f"{self.F[f]['name']} fix didn't work: rule it out")
        self.emit("faults", probs={k: round(v, 3) for k, v in self.probs.items()})
        self.phase, self.fault, self.lead = "diagnose", None, False
        return self.next_move("Okay, so it's not that. ")

    def on_done(self, u):
        return self.say(self.kb["after_end"])

    on_handoff = on_done
