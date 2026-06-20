# computer-use-agent

A tiny **computer-use agent for macOS**: tell it a task in plain English and it
operates your Mac — looking at the screen and moving the mouse/keyboard — to do it.
The v0 task is *"schedule a meeting with someone,"* and the showpiece is that it
**figures out the person's email itself** by typing their name and reading the
autocomplete dropdown, instead of being handed the address.

It also includes a small **record-and-replay** mode (see below), a miniature of the
hybrid approach Cyberdesk uses.

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
| `agent.py` | **the brain wiring** — the Claude `computer`-tool loop + record/replay |
| `trajectories/` | recorded action sequences for `--replay` |

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
python agent.py "Schedule a 15-min 'catch up' with Mac Ajwani Sun Jun 21 3:30pm" --dry-run
```

**Vision mode** (Claude drives, and the run is recorded):

```bash
python agent.py "Schedule a 15-min 'catch up' with Mac Ajwani Sun Jun 21 3:30pm" --name mac_call
```

It opens your browser to **Google Calendar** (`calendar.google.com`), creates the
event, types the person's name, reads Google's autocomplete to **resolve their email**,
and then **stops before saving** with a `READY TO SEND:` summary. Press Enter to finish,
or type `approved — send the invite` to let it save and send.

**Replay mode** (the Cyberdesk stub — deterministic, **no model calls, ~zero cost**):

```bash
python agent.py --replay coffee_mac
```

## Cost

A vision run re-sends every prior screenshot each step (~4,600 tokens/screenshot), so
input grows each turn. At Opus 4.8 pricing it adds up fast — which is the whole point
of replay. The agent prints a live `[usage]` line per step and a `[cost]` total.

| | per run |
|---|---|
| Vision run, **no caching** | ~$2–5 |
| Vision run, **with prompt caching** (on — prior screenshots re-read at ~0.1×) | ~$0.50–0.80 |
| `--replay` (no model calls at all) | **$0.00** |

Caching is built in (a cached system block + a moving cache breakpoint on the latest
screenshot). Billed to your Anthropic API account — see console.anthropic.com → Usage.

## Why the "replay" mode mirrors Cyberdesk

Cyberdesk (YC S25) describes their agent as one that *"follow[s] memorized steps
reliably, only falling back to computer use models during expected popups,"* so
repeated tasks run *"100% deterministically, upwards of 3x faster, and almost zero
cost."* In their words: *vision is the fallback, not the hot path.*

`--replay` is a miniature of that: vision mode records the exact action sequence once;
replay re-executes it with no screenshots and no LLM calls. (A production version
would add a cheap per-step check that the screen looks as expected and fall back to a
live vision step on a surprise — that "fall back on surprise" piece is the part this
stub leaves as the obvious next step.)

## Safety

This program moves your real mouse and types real keystrokes.

- **Kill switch:** slam the mouse into any screen **corner** to abort instantly
  (pyautogui FAILSAFE).
- **Approval gate:** it stops before sending any invite/email; nothing goes out until
  you approve.
- **Step cap:** stops after 40 steps so it can't loop forever.
- Run it **attended**, and close sensitive windows first.
