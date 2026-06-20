# computer-use-agent

A tiny **computer-use agent for macOS**: tell it any task in plain English and it
operates your Mac — looking at the screen and moving the mouse/keyboard — to do it.
The system prompt is **generic** (operating principles, not task steps), so the agent
**figures out *how* to do whatever you ask, live**: open an app, schedule a meeting and
resolve a guest's email from autocomplete, search the web, write a note, etc.

It also includes a **learn-then-replay** mode: record a task once, then replay it
deterministically for ~$0 — falling back to live vision the moment the screen diverges.
That's the hybrid pattern Cyberdesk uses, in miniature (see below).

> Built as a learning project. The interesting part isn't the lines of code — it's
> the mental model below.

## What a computer-use agent actually is

Two separate halves:

```
   ┌─────────────────────────┐         ┌──────────────────────────┐
   │   THE BRAIN (Claude)    │         │   THE HANDS (this repo)  │
   │   decides what to do    │ ◄─────► │   does it on the Mac     │
   │   runs at Anthropic     │         │   runs locally           │
   └─────────────────────────┘         └──────────────────────────┘
```

The model **never touches your computer**. It only ever looks at a screenshot and
replies with a structured action like `left_click [840, 210]` or `type "Coffee"`.
The local program is what performs the click. So "building a computer-use agent" is
mostly **building the hands**; the brain is Anthropic's `computer` tool.

The loop, which is the whole thing:

```
1. SCREENSHOT   grab the screen        ──►  send image to Claude
2. DECIDE       Claude picks ONE action
3. ACT          Claude replies {action: "left_click", coordinate: [840, 210]}
                this program performs that click
4. REPEAT       grab a NEW screenshot  ──►  back to step 2
                ... until Claude says it's done
```

- **Model-agnostic.** Any vision + tool-calling model can drive this loop; we use
  Claude because it has a first-party `computer` tool and strong vision, which is the
  fastest path. (OpenAI Operator, Gemini/Project Mariner, UI-TARS are alternatives.)
- **Screenshots aren't the only input.** You can instead read the UI *structurally*
  via the macOS **Accessibility API** (`AXUIElement`) — more reliable, but missing on
  legacy/remote apps. Screenshots are the universal option and what the `computer`
  tool expects. The Accessibility API is the obvious v2 upgrade.

## Files

| File | Role |
|---|---|
| `mac_control.py` | **the hands** — Retina-aware screenshots + mouse/keyboard |
| `agent.py` | **the brain wiring** — generic Claude `computer`-tool loop + learn/replay |
| `trajectories/` | recorded action + screen-fingerprint sequences for `--replay` |

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
```

**Grant two macOS permissions** to the terminal you run this from
(System Settings → Privacy & Security):
- **Screen Recording** — or screenshots come back blank.
- **Accessibility** — or clicks and keystrokes silently do nothing.

Sanity-check the hands (safe — no clicks):

```bash
python mac_control.py   # should print your screen's logical size
```

## Run

**Dry run first** (one model call, prints the first action it *would* take, clicks nothing):

```bash
python agent.py "Open TextEdit and write a haiku about Houston" --dry-run
```

**Vision mode** (Claude figures the task out live, and the run is recorded). The prompt
is generic, so any of these work:

```bash
python agent.py "Open TextEdit and write a haiku about Houston" --name haiku
python agent.py "Schedule a 15-min 'catch up' with Mac Ajwani in Google Calendar Sun Jun 21 3:30-3:45pm" --name mac_call
```

For an outward-facing action (sending an invite/email, submitting a form), the agent
**composes everything then stops** with a `READY TO SEND:` summary — press Enter to
finish, or type `approved — send it` to let it go. For the meeting task, the showpiece
is that it **resolves the guest's email itself** from the autocomplete dropdown.

**Replay mode** — replay the recorded run, **$0 and no model calls**, *but* it checks the
screen still matches before each step and **falls back to live vision on a surprise**:

```bash
python agent.py --replay mac_call               # deterministic; vision only if needed
python agent.py --replay mac_call --threshold 0.2   # how different counts as a "surprise"
```

## Managing what it's memorized

A "memory" is just a JSON file in `trajectories/` (local only, git-ignored — nothing in
a database or the cloud). A new vision run with the same `--name` (or same task text)
**overwrites** that file; a `Ctrl-C`'d run saves nothing.

```bash
python agent.py --list              # show memorized tasks (name, step count, goal)
python agent.py --forget mac_call   # delete one
python agent.py --forget all        # delete all (or: rm trajectories/*.json)
```

## Cost

A vision run re-sends every prior screenshot each step (~4,600 tokens/screenshot), so
input grows each turn. At Opus 4.8 pricing it adds up fast — which is the whole point
of replay. The agent prints a live `[usage]` line per step and a `[cost]` total.

| | per run |
|---|---|
| Vision run, **no caching** | ~$2–5 |
| Vision run, **with prompt caching** (on — prior screenshots re-read at ~0.1×) | ~$0.50–0.80 |
| `--replay` (deterministic; vision only if the screen diverges) | **$0** (until a surprise) |

Caching is built in (a cached system block + a moving cache breakpoint on the latest
screenshot). Billed to your Anthropic API account — see console.anthropic.com → Usage.

## Why the "replay" mode mirrors Cyberdesk

Cyberdesk (YC S25) describes their agent as one that *"follow[s] memorized steps
reliably, only falling back to computer use models during expected popups,"* so
repeated tasks run *"100% deterministically, upwards of 3x faster, and almost zero
cost."* In their words: *vision is the fallback, not the hot path.*

`--replay` implements that loop in miniature:

1. **Learn once.** Vision mode records each action *plus a small grayscale thumbnail of
   the screen right before it* — a cheap "fingerprint" of the expected state.
2. **Replay deterministically.** On replay, before each step it screenshots and compares
   to the recorded fingerprint. If they match (within `--threshold`), it executes the
   recorded action with **no model call** — fast and free.
3. **Fall back on surprise.** If the screen has diverged (a popup, a moved window, a
   different autocomplete list), it prints `*** SURPRISE ***` and hands off to a live
   vision step to recover and finish.

This is **demo-grade, not production-grade**: the divergence check is a pixel-difference
heuristic. Real systems fingerprint by *element identity* (the accessibility tree / DOM),
which is robust to layout shifts — and they re-anchor and resume the recording after a
surprise instead of finishing in pure vision. The architecture is right; the robustness
is where the engineering goes.

## Safety

This program moves your real mouse and types real keystrokes.

- **Kill switch:** **`Ctrl-C` in the terminal** always works. Also: hold the mouse in
  any screen **corner** — a margin-based check aborts before the next action (more
  reliable than pyautogui's exact-pixel FAILSAFE, which only fires at the very corner
  pixel and only at the instant an action starts, not during the model's thinking).
- **Approval gate:** it stops before sending any invite/email; nothing goes out until
  you approve.
- **Step cap:** stops after 40 steps so it can't loop forever.
- Run it **attended**, and close sensitive windows first.
