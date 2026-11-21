# Mistri: repair help by voice

A live voice demo. You talk to Mistri about a broken printer. It works out what's
wrong, talks you through the fix, and remembers you. When you call again, it lets
you lead and only marks a skill as proven if you fix it without help.

The dashboard shows the call, Mistri's reasoning between turns, the fault graph
(odds of each cause) and the skill graph, all updating live.

## What's real

| Part | How |
|---|---|
| Speech to text | The browser's built-in speech recognition (Chrome). Needs internet. |
| Diagnosis | [mistri/engine.py](mistri/engine.py): Bayes' rule over the likelihood table in [kb/printer.json](kb/printer.json) updates the odds of 6 causes after every answer. The next question is the one with the most expected information gain. Past 85% it stops asking and gives the fix. |
| Understanding the caller | [mistri/understand.py](mistri/understand.py): answers that use the knowledge base's words ("4 by 6", "the bottom one", "normal") are matched instantly. Anything else goes to Claude through headless Claude Code (`claude -p`, about 3 to 4 s). |
| Memory | Callers are remembered in `data/learners.json`. A skill taught last call but not proven means Mistri asks you first and stays out of the way. |
| Text to speech | The browser's built-in voices. |

Mistri's spoken lines come from the knowledge base, filled in with what you said
("set for small photo paper, but there's normal A4 in it"). Claude interprets the
caller. It never writes Mistri's lines. That's the "guided" part.

## Set up (once, about 10 minutes)

You need a Mac or Linux laptop with:

1. **Python 3.9 or newer.** Check with `python3 --version`.
2. **Google Chrome.** Safari also works; Firefox has no speech recognition.
3. **Claude Code, logged in.** Install it:

   ```bash
   curl -fsSL https://claude.ai/install.sh | bash
   ```

   Then run `claude` once and log in with a Claude account. Mistri uses this login,
   so no API key is needed. Without it, the demo lines still work; only
   off-script phrasings get "Sorry, I didn't catch that."

Then get the code and install it:

```bash
git clone https://github.com/hsurick/mistri.git
cd mistri
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Run the demo

```bash
.venv/bin/uvicorn mistri.server:app --port 8000
```

1. Open http://localhost:8000 in Chrome.
2. Press **Start call** and allow the microphone.
3. Talk. Mistri answers out loud, then listens again. You can also type lines into the box.
4. Press **Forget caller** before each run so call 1 starts fresh.

Check it's healthy: http://localhost:8000/health should show `"claude"` with a path.
The first Claude answer after starting the server can take about 10 s; later ones take 3 to 4 s.

## Demo script

Two calls, three days apart. Say the caller lines; Mistri's replies are what
you should hear back.

**Call 1: first time, fully guided**

| Caller says | Mistri replies |
|---|---|
| (press Start call) | Hi, this is Mistri. What are we fixing today? |
| I need help fixing a printer. | Sure. What's the name on the front of it? |
| Canon PIXMA G3010. | Got it. What's it doing? |
| It won't print. There's an orange light on. | Look next to that orange light. There's a little column: A4, LTR, 4 by 6. Which one's lit? |
| Uh… the bottom one? | That's 4 by 6, the small photo size. What paper did you put in: normal sheets, or small photo paper? |
| Normal. | Found it. The printer's set for small photo paper, but there's normal A4 in it, so it stopped. Press the button with the page picture until A4 lights up, then press OK. |
| It's printing. | Nice. Remember this one: orange light, check the size light matches your paper. Next time, try that before you call me. |

The call ends by itself. Paper size is amber (taught, not proven).

**Call 2: press Start call again (don't press Forget caller)**

| Caller says | Mistri replies |
|---|---|
| (press Start call) | Hi again. What's the printer doing? |
| Hi, it's the printer again. Orange light. | Same light as last time. What do you think it is? |
| The size light says LTR this time, and I've got A4 in. I'll switch it. | Go for it. |
| A4's lit… OK… it's printing. | Perfect. You didn't need me this time. |

Paper size turns green (proven), and the Calls table shows call 2 needed no questions and no instructions.

Going off script works too, because the engine is real. Say "the top one"
(A4) and it moves on to other checks. Say "yeah there's paper stuck in there"
and it switches to the paper-jam fix. Say "still not printing" and it rules
that cause out and asks something else.

## Tips for stage

- Use a quiet spot and talk close to the laptop mic; the browser's recognition is the weakest link.
- If speech misfires, type the line into the box. It goes through the same path.
- Keep the laptop online: both speech recognition and Claude need internet.

## Tests

```bash
.venv/bin/pip install pytest
.venv/bin/pytest -q
```

## Optional: a real phone number

`/twiml` and `/relay` speak Twilio ConversationRelay, so a Twilio number can call
the same engine. A free Twilio trial can't get a number, so this needs a paid
account. Expose the server over HTTPS (`cloudflared tunnel --url
http://localhost:8000`) and set the number's voice webhook to
`https://<host>/twiml` (HTTP POST). Anyone with that URL can drive your Claude
Code login, so stop the tunnel after the demo. The Docker files deploy to a
server, but a server has no logged-in `claude`, so only instant answers work there.

## Settings

| Variable | Default | |
|---|---|---|
| `MISTRI_MODEL` | `haiku` | Model for headless Claude Code |
| `MISTRI_CLAUDE` | `on` | `off`: instant answers only |
| `MISTRI_CLAUDE_TIMEOUT` | `15` | Seconds before giving up on Claude |
| `MISTRI_KB` | `kb/printer.json` | Knowledge base |
