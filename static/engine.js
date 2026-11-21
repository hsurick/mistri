// Browser-only Mistri: the same engine as mistri/engine.py and the answer-word matching from
// mistri/understand.py, run in the page so the demo needs no server (e.g. on GitHub Pages).
// No Claude here: lines that don't use the knowledge base's answer words get "Sorry, I didn't catch that."

const RATE = {read_guided: .4, taught: .25, fixed_guided: .27, read_alone: .73, fixed_alone: .84};
const NEG = new Set(['no','not','dont','isnt','wont','cant','never','nothing','didnt','doesnt','without','aint','arent']);
const REPEAT = ['what','sorry','say again','repeat','pardon','come again','huh'];
const DONT_KNOW = ["don't know",'dont know','not sure','no idea',"can't tell",'cant tell'];
const FIXED = ['printing','it prints',"it's working",'its working','works now','working now','came out'];
const NOT_FIXED = ['still not','not printing',"didn't work",'didnt work',"still won't",'still wont','nothing happened','still orange','still blinking'];
const DONE = ['done','did it','lit','on now','pressed','finished','got it','closed it',"it's in",'its in','clicked','swapped','changed it','pulled it'];
const GENERIC = new Set(['yes','yeah','yep','no','nope','on','off','lit','none','nothing','dark','up','down','open','closed','clear','stuck']);

const outcomes = t => t.outcomes || t.answers.map(a => a.id);
const lik = (t, o, f) => (t.p[f] || t.p['*'])[o];
const entropy = ps => -ps.reduce((s, p) => s + (p > 0 ? p * Math.log2(p) : 0), 0);
const normalise = P => { const s = Object.values(P).reduce((a, b) => a + b, 0); return Object.fromEntries(Object.entries(P).map(([k, v]) => [k, v / s])) };
const pct = p => Math.round(p * 100) + '%';
const r2 = x => Math.round(x * 100) / 100;
const argmax = P => Object.keys(P).reduce((a, b) => P[b] > P[a] ? b : a);
const label = t => t.label.replace(/\?$/, '');

export function checkKb(kb) {
  for (const t of kb.tests) for (const f of kb.faults) {
    const s = outcomes(t).reduce((a, o) => a + lik(t, o, f.id), 0);
    if (Math.abs(s - 1) > 1e-6) throw new Error(`${t.id}/${f.id} likelihoods sum to ${s}`);
  }
  return kb;
}
export const newLearner = () => ({device: null, skills: {}, taught: [], calls: [], last_evidence: {}});

export class Call {
  constructor(kb, learner, callNo = 1) {
    this.kb = kb; this.L = learner || newLearner(); this.callNo = callNo;
    this.F = Object.fromEntries(kb.faults.map(f => [f.id, f]));
    this.T = Object.fromEntries(kb.tests.map(t => [t.id, t]));
    this.S = Object.fromEntries(kb.skills.map(s => [s.id, s]));
    this.threshold = kb.threshold ?? .9;
    this.probs = normalise(Object.fromEntries(kb.faults.map(f => [f.id, f.prior])));
    this.ev = {}; this.skipped = new Set(); this.ruledOut = [];
    this.returning = !!this.L.device;
    this.phase = this.returning ? 'diagnose' : 'device';
    this.lead = false; this.leadOffered = false; this.q = null; this.fault = null; this.step = 0;
    this.questions = 0; this.instructions = 0; this.touched = new Set();
    this.last = this.returning ? kb.greeting_returning : kb.greeting;
    this.out = [];
  }
  emit(type, kw = {}) { this.out.push({type, ...kw}) }
  think(text) { this.emit('thought', {text}) }
  title(title) { this.emit('think_title', {title}) }
  drain() { const o = this.out; this.out = []; return o }
  say(t) { this.last = t; return t }

  start() {
    if (!this.returning) { this.title('Picking up'); this.think('Unknown number, no history on file'); this.think('No device or problem known yet: ask an open question') }
    else { this.title('Recognising the caller'); this.think('Caller ID matches someone who called before'); this.loadGraph() }
    return this.last;
  }
  loadGraph() {
    const kb = this.kb;
    this.emit('device', {device: kb.device, graph: kb.graph_id});
    this.think(`${this.returning ? 'Printer on file' : 'Model'}: ${kb.device}. Loading its fault graph: ${kb.faults.length} causes, ${kb.tests.filter(t => !t.symptom).length} tests`);
    this.emit('graph', {faults: kb.faults.map(f => ({id: f.id, name: f.name, prior: f.prior})),
      tests: kb.tests.filter(t => !t.symptom).map(t => ({id: t.id, label: t.label, links: Object.keys(t.p).filter(k => k !== '*')})),
      skills: kb.skills, skill_edges: kb.skill_edges, threshold: this.threshold, symptom_title: kb.symptom_title || 'Symptom'});
    for (const [sid, s] of Object.entries(this.L.skills)) this.emit('skill', {skill: sid, p: s.p, proven: s.proven});
    if (!this.returning) { this.think('New caller, no skills on record: give full guidance'); this.emit('learner', {caller: 'new', support: 'full guidance'}) }
    this.L.device = kb.device;
  }

  outcome(t, a) {
    const c = t.compare;
    if (!c || (c.passthrough || []).includes(a)) return a;
    return c.same[a].includes(this.ev[c.test]) ? 'match' : 'mismatch';
  }
  available() { return this.kb.tests.filter(t => !(t.id in this.ev) && !this.skipped.has(t.id) && !t.symptom && (!t.requires || t.requires in this.ev)) }
  // expected entropy drop (bits); a check that unlocks another also counts the best follow-up (one step of lookahead)
  gain(t, probs = this.probs, ev = this.ev, lookahead = true) {
    const unlocks = lookahead ? this.kb.tests.filter(u => u.requires === t.id && !(u.id in ev)) : [];
    let exp = 0, follow = 0;
    for (const a of unlocks.length ? t.answers.map(x => x.id) : outcomes(t)) {
      const o = unlocks.length ? this.outcome(t, a) : a;
      const joint = Object.fromEntries(Object.entries(probs).map(([f, p]) => [f, p * lik(t, o, f)]));
      const po = Object.values(joint).reduce((x, y) => x + y, 0);
      if (po > 0) {
        const post = Object.fromEntries(Object.entries(joint).map(([f, v]) => [f, v / po]));
        exp += po * entropy(Object.values(post));
        if (unlocks.length) {
          const saved = this.ev; this.ev = {...ev, [t.id]: a};
          follow += po * Math.max(...unlocks.map(u => this.gain(u, post, this.ev, false)));
          this.ev = saved;
        }
      }
    }
    return entropy(Object.values(probs)) - exp + follow;
  }
  answer(t, a) { return t.answers.find(x => x.id === a) }
  observe(t, a, alone = false) {
    const o = this.outcome(t, a);
    this.probs = normalise(Object.fromEntries(Object.entries(this.probs).map(([f, p]) => [f, p * lik(t, o, f)])));
    this.ev[t.id] = a;
    const ans = this.answer(t, a);
    this.emit('evidence', {test: t.id, result: a, outcome: o, label: ans.label});
    if (t.compare && (o === 'match' || o === 'mismatch')) {
      const size = this.answer(this.T[t.compare.test], this.ev[t.compare.test]).label;
      this.think(`Setting says ${size}, tray holds ${ans.label}: they ${o === 'match' ? 'match' : 'don’t match'}`);
    } else if (ans.note) this.think(ans.note);
    if (t.skill) this.bump(t.skill, alone ? 'read_alone' : 'read_guided', alone);
  }
  bump(skill, kind, proven = false) {
    if (!this.S[skill] || this.S[skill].expert) return null;
    const s = this.L.skills[skill] ||= {p: 0, proven: false};
    s.p = r2(s.p + (1 - s.p) * RATE[kind]); s.proven = s.proven || proven;
    this.touched.add(skill);
    this.emit('skill', {skill, p: s.p, proven: s.proven});
    return s;
  }
  fmt(text) {
    const sz = this.ev.size_light ? this.answer(this.T.size_light, this.ev.size_light) : {};
    const pa = this.ev.paper ? this.answer(this.T.paper, this.ev.paper) : {};
    const v = {size_desc: sz.desc || 'a different size', paper_desc: pa.desc || 'other paper', paper: pa.paper || 'the right size'};
    return text.replace(/\{(\w+)\}/g, (_, k) => v[k] ?? '');
  }
  oddsLine(before) {
    const top = argmax(this.probs);
    if (top === before[0] && Object.keys(this.ev).length > 1) return `${this.F[top].name}: ${pct(before[1])} → ${pct(this.probs[top])}`;
    const second = Object.keys(this.probs).sort((a, b) => this.probs[a] - this.probs[b]).at(-2);
    return `Updated the odds: ${this.F[top].name.toLowerCase()} leads at ${pct(this.probs[top])}, ${this.F[second].name.toLowerCase()} ${pct(this.probs[second])}`;
  }
  best() { const k = argmax(this.probs); return [k, this.probs[k]] }

  handle(u) {
    if (u.intent === 'repeat') { this.title('Repeating'); this.think('Caller asked to hear it again'); return this.say(this.last) }
    return this['on_' + this.phase](u);
  }
  on_device(u) {
    this.title('Finding the device');
    if (u.device) {
      this.think(`Device: a ${this.kb.device_kind.name}. Model unknown`);
      this.think('Every printer has a different panel, so ask for the model before giving any steps');
      this.phase = 'model'; return this.say("Sure. What's the name on the front of it?");
    }
    this.think('No device named yet'); return this.say('I can help with printers. What are we fixing?');
  }
  on_model(u) {
    this.title('Loading the printer');
    if (u.model) { this.loadGraph(); this.phase = 'diagnose'; return this.say("Got it. What's it doing?") }
    this.think("Didn't recognise a model name"); return this.say("Which model is it? The name's on the front, like PIXMA G3010.");
  }
  apply(u, alone = false) {
    const order = this.kb.tests.map(t => t.id);
    const obs = (u.observations || []).filter(o => this.T[o.test] && !(o.test in this.ev) && this.T[o.test].answers.some(a => a.id === o.answer))
      .sort((a, b) => order.indexOf(a.test) - order.indexOf(b.test));
    const ok = new Set();
    for (const o of obs) { const req = this.T[o.test].requires; if (!req || req in this.ev || ok.has(req)) ok.add(o.test) }
    const got = obs.filter(o => ok.has(o.test)).map(o => [this.T[o.test], o.answer]);
    if (got.length) this.think(u.gloss || 'Heard: ' + got.map(([t, a]) => t.symptom ? this.answer(t, a).label.toLowerCase() : `${label(t).toLowerCase()} ${this.answer(t, a).label}`).join(', '));
    for (const [t, a] of got) this.observe(t, a, alone);
    return got;
  }
  on_diagnose(u) {
    const before = this.best();
    this.title(this.returning && !this.leadOffered && !Object.keys(this.ev).length ? 'Recognising the caller' : !this.q ? "Working out what's wrong" : `Reading the ${label(this.T[this.q]).toLowerCase()}`);
    const got = this.apply(u);
    if (!got.length && this.q && u.intent === 'dont_know') {
      this.title('Trying another check'); this.skipped.add(this.q); this.think("Caller can't check that: skip it");
      return this.nextMove("No problem, let's try something else. ");
    }
    if (!got.length && this.q) { this.title("Didn't catch that"); this.think(u.gloss || "Couldn't match that to the question"); return this.say(`${this.kb.fallback} ${this.T[this.q].ask}`) }
    if (got.length) this.emit('faults', {probs: this.roundedProbs()});
    const pending = this.L.taught.filter(f => !this.L.skills[this.F[f].teach]?.proven);
    if (this.returning && got.length && pending.length && !this.leadOffered) {
      this.leadOffered = true; this.phase = 'lead'; this.lead = true;
      const skill = this.S[this.F[pending.at(-1)].teach].name;
      const same = got.every(([t, a]) => this.L.last_evidence[t.id] === a);
      this.think(same ? `${got.map(([t, a]) => this.answer(t, a).label).join(', ')} again: same starting odds as last time` : 'Different start from last time');
      this.think(`${skill} was taught last call, not proven. Fade the help: ask them first`);
      this.emit('learner', {caller: 'returning', support: 'let them lead'});
      return this.say((same ? 'Same light as last time. ' : '') + 'What do you think it is?');
    }
    if (got.length && Math.max(...Object.values(this.probs)) < this.threshold) this.think(this.oddsLine(before));
    const ack = got.length === 1 ? (this.answer(...got[0]).ack || '') : '';
    return this.nextMove(ack ? ack + ' ' : '', before);
  }
  on_lead(u) {
    this.title('Letting the caller lead');
    const before = this.best();
    const got = this.apply(u, true);
    if (got.length) {
      this.think('Caller read the panel on their own: ' + got.map(([t, a]) => `${label(t).toLowerCase()} ${this.answer(t, a).label}`).join(', '));
      this.emit('faults', {probs: this.roundedProbs()});
      if (got.some(([t, a]) => t.id in this.L.last_evidence && this.L.last_evidence[t.id] !== a))
        this.think('Different setting from last time, same rule: they understood it, not just remembered a button');
    }
    const [top, p] = this.best();
    if (p >= this.threshold) return this.beginRepair(top, before);
    this.lead = false; this.phase = 'diagnose';
    this.emit('learner', {caller: 'returning', support: 'guided'});
    this.think('Not enough to go on yet: back to guided questions');
    return this.nextMove("Okay, let's check. ");
  }
  roundedProbs() { return Object.fromEntries(Object.entries(this.probs).map(([k, v]) => [k, Math.round(v * 1000) / 1000])) }
  nextMove(prefix = '', before = null) {
    const [top, p] = this.best();
    if (p >= this.threshold) return this.beginRepair(top, before || [top, p]);
    const scored = this.available().map(t => [this.gain(t), t]);
    if (!scored.length || Math.max(...scored.map(s => s[0])) < .01) {
      this.phase = 'handoff'; this.emit('handoff');
      this.think('No check left that would narrow it down: hand off to a technician');
      return this.say(prefix + this.kb.handoff);
    }
    const byGain = [...scored].sort((a, b) => b[0] - a[0]);
    const byValue = [...scored].sort((a, b) => b[0] / (b[1].effort || 1) - a[0] / (a[1].effort || 1));
    const q = byValue[0][1];
    this.q = q.id; this.questions++;
    this.think('Scored the next questions: ' + byGain.slice(0, 3).map(([g, t]) => `${label(t).toLowerCase()} ${g.toFixed(2)} bits`).join(', '));
    if (q !== byGain[0][1] || q.why_first) this.think(q.why_first || `${label(q)} is quicker for the caller to check: ask that first`);
    this.emit('next_question', {chosen: this.q, candidates: byGain.slice(0, 3).map(([g, t]) => ({test: t.id, ask: t.label, gain_bits: Math.round(g * 1000) / 1000}))});
    return this.say(prefix + q.ask);
  }
  beginRepair(fault, before) {
    this.phase = 'repair'; this.fault = fault; this.step = 0; this.q = null;
    const f = this.F[fault], p = this.probs[fault];
    this.think(`${f.name} ${before[0] === fault ? pct(before[1]) + ' → ' : ''}${pct(p)}, past the ${pct(this.threshold)} threshold: stop asking`);
    this.emit('stop', {fault, p: Math.round(p * 1000) / 1000, threshold: this.threshold});
    if (this.lead) { this.think('They have the fix. Stay out of the way'); return this.say('Go for it.') }
    this.emit('repair_step', {step: 0, text: this.fmt(f.steps[0].say)});
    this.think('Explain the cause in one sentence, then give the fix');
    this.bump(f.teach, 'taught'); this.instructions++;
    return this.say(`Found it. ${this.fmt(f.explain)} ${this.fmt(f.steps[0].say)}`);
  }
  on_repair(u) {
    const steps = this.F[this.fault].steps, cur = steps[Math.min(this.step, steps.length - 1)], i = u.intent;
    this.title(i === 'fixed' ? (this.lead ? 'Checking they did it alone' : 'Confirming the fix') : 'Guiding the fix');
    if (i === 'fixed') return this.confirm();
    if (i === 'not_fixed') return this.ruleOut();
    if (i === 'stuck' && !this.lead) { this.think(u.gloss || 'Caller is stuck on the step: describe it more exactly'); return this.say(cur.help) }
    if (i === 'done' && !this.lead) {
      this.step++;
      if (this.step < steps.length) {
        this.instructions++; this.think('Step done: give the next one');
        this.emit('repair_step', {step: this.step, text: this.fmt(steps[this.step].say)});
        return this.say('Good. ' + this.fmt(steps[this.step].say));
      }
      this.think('All steps done: check it worked'); return this.say('Is it printing now?');
    }
    if (this.lead) { this.think(u.gloss || 'Still working on it: wait'); return this.say('Take your time. Tell me when it prints.') }
    return this.say(`${this.kb.fallback} ${this.fmt(cur.say)}`);
  }
  confirm() {
    const f = this.F[this.fault];
    this.emit('repair_confirmed', {fault: this.fault, step: this.step});
    let reply;
    if (this.lead) {
      this.think('Repair confirmed, with no instructions from Mistri');
      const s = this.bump(f.teach, 'fixed_alone', true);
      this.think(`Diagnosed and fixed it alone: ${this.S[f.teach].name.toLowerCase()} proven, p ${s.p.toFixed(2)}`);
      reply = "Perfect. You didn't need me this time.";
    } else {
      this.think(`Repair confirmed: ${f.name.toLowerCase()} fixed`);
      const parts = [...this.touched].sort((a, b) => (a !== f.teach) - (b !== f.teach)).map(k => `${this.S[k].name.toLowerCase()} p ${this.bump(k, 'fixed_guided').p.toFixed(2)}`);
      this.think('Did the fix with step-by-step help: ' + parts.join('; '));
      this.think('Not proven yet: hasn’t done it without help');
      this.think('End with the rule so they can try it alone next time');
      if (!this.L.taught.includes(this.fault)) this.L.taught.push(this.fault);
      reply = `Nice. Remember this one: ${f.rule} Next time, try that before you call me.`;
    }
    this.finish();
    return this.say(reply);
  }
  finish() {
    this.phase = 'done';
    this.L.last_evidence = {...this.ev};
    const prev = this.L.calls.at(-1);
    this.emit('summary', {call: `Call ${this.callNo}`, fault: this.fault, questions: this.questions, instructions: this.instructions});
    if (prev && this.lead) this.think(`Last call: ${prev.questions} questions, ${prev.instructions} instructions, ${prev.secs ?? '?'} s. This call: ${this.questions || this.instructions ? this.questions + ' questions' : 'none'}`);
  }
  ruleOut() {
    const f = this.fault;
    this.ruledOut.push(f); this.probs[f] *= .02; this.probs = normalise(this.probs);
    this.think(`${this.F[f].name} fix didn't work: rule it out`);
    this.emit('faults', {probs: this.roundedProbs()});
    this.phase = 'diagnose'; this.fault = null; this.lead = false;
    return this.nextMove("Okay, so it's not that. ");
  }
  on_done() { return this.say(this.kb.after_end) }
  on_handoff() { return this.say(this.kb.after_end) }
}

// ---- understanding (answer words only)
export const norm = s => s.toLowerCase().replace(/×/g, 'x').replace(/’/g, "'").replace(/[^a-z0-9' ]+/g, ' ').replace(/'/g, '').replace(/\s+/g, ' ').trim();
// positions where kw appears as whole words, skipping ones with a negation in the 3 words before
function hits(text, kw, negOk = false) {
  const w = norm(text).split(' '), k = norm(kw).split(' '), out = [];
  for (let i = 0; i + k.length <= w.length; i++)
    if (k.every((x, j) => w[i + j] === x) && (negOk || NEG.has(k[0]) || !w.slice(Math.max(0, i - 3), i).some(x => NEG.has(x)))) out.push(i);
  return out;
}
const hit = (text, kw, negOk) => hits(text, kw, negOk).length > 0;

export function understand(call, text) {
  if (norm(text).split(' ').length <= 4 && REPEAT.some(k => hit(text, k, true))) return {intent: 'repeat'};
  if (call.phase === 'device') return call.kb.device_kind.keywords.some(k => hit(text, k)) ? {intent: 'answer', device: true} : {intent: 'other'};
  if (call.phase === 'model') return call.kb.model_keywords.some(k => hit(text, k, true)) ? {intent: 'answer', model: true} : {intent: 'other'};
  if (call.phase === 'diagnose' || call.phase === 'lead') {
    const asked = call.q && !(call.q in call.ev) ? [call.T[call.q]] : [];
    const tests = [...asked, ...call.kb.tests.filter(t => !(t.id in call.ev) && !asked.includes(t) && (t.volunteer || call.phase === 'lead'))];
    // each test: which answers match, and at which word positions
    const m = tests.map(t => ({t, opts: t.answers.map(a => ({a: a.id, pos: a.keywords.filter(k => asked.length === 1 && asked[0] === t || !GENERIC.has(norm(k))).flatMap(k => hits(text, k))})).filter(o => o.pos.length)}));
    // a word that settles one test can't also answer another: "LTR this time, and I've got A4 in" -> size LTR, paper A4
    for (let changed = true; changed;) {
      changed = false;
      const claimed = new Set(m.filter(x => x.opts.length === 1).flatMap(x => x.opts[0].pos));
      for (const x of m) if (x.opts.length > 1) {
        const keep = x.opts.filter(o => o.pos.some(p => !claimed.has(p)));
        if (keep.length && keep.length < x.opts.length) { x.opts = keep; changed = true }
      }
    }
    const obs = m.filter(x => x.opts.length === 1).map(x => ({test: x.t.id, answer: x.opts[0].a}));
    if (obs.length) return {intent: 'answer', observations: obs};
    if (DONT_KNOW.some(k => hit(text, k, true))) return {intent: 'dont_know'};
    return {intent: 'other'};
  }
  if (call.phase === 'repair') {
    if (NOT_FIXED.some(k => hit(text, k, true))) return {intent: 'not_fixed'};
    if (FIXED.some(k => hit(text, k))) return {intent: 'fixed'};
    if (/\b(which|where|how|what|two|both)\b/.test(norm(text)) || text.includes('?')) return {intent: 'stuck'};
    if (DONE.some(k => hit(text, k))) return {intent: 'done'};
  }
  return {intent: 'other'};
}
