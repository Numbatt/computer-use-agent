"""agent.py — "the brain wiring" + the record/replay stub.

This is the loop that makes a computer-use agent a computer-use agent:

    screenshot  ->  Claude picks ONE action  ->  we do it on the Mac  ->  new
    screenshot  ->  ... repeat until Claude says it's done.

Claude never touches the machine. It only looks at screenshots (via Anthropic's
`computer` tool) and replies with structured actions; mac_control.py performs them.

Two modes:
  * VISION mode (default): Claude drives, and we LOG every action it takes to
    trajectories/<name>.json.
  * REPLAY mode (--replay <name>): we replay that logged action sequence with NO
    model calls at all — the deterministic, ~free, fast path. This is a tiny
    version of Cyberdesk's "memorize the steps, replay deterministically, fall
    back to the vision model only on surprises" idea.

Usage:
    python agent.py "Schedule a 30-min Coffee with Mac Ajwani tomorrow at 3pm"
    python agent.py "..." --name coffee_mac          # name the recorded trajectory
    python agent.py --replay coffee_mac              # deterministic replay, no LLM
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import os

import anthropic

import mac_control as mac


def _load_dotenv() -> None:
    """Load KEY=VALUE lines from a local .env (next to this file) into os.environ,
    without overriding anything already set. No dependency, so `pip install` stays tiny."""
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


MODEL = "claude-opus-4-8"
MAX_TOKENS = 4096
DEFAULT_MAX_STEPS = 40
TRAJ_DIR = Path(__file__).parent / "trajectories"

# Computer use is gated behind a beta header AND the tool TYPE is version-pinned to
# the model generation — they must match. Newer models (Opus 4.5/4.8) use the
# 20251124 pair; older ones use 20250124. We try newest first and cache the winner.
TOOL_BETA_CANDIDATES = [
    ("computer_20251124", "computer-use-2025-11-24"),
    ("computer_20250124", "computer-use-2025-01-24"),
]

SYSTEM_PROMPT = """\
You are operating Diego's personal Mac by looking at screenshots and issuing mouse
and keyboard actions. You are careful, take one deliberate action at a time, and
re-check the screen after each step.

To schedule a meeting with a person:
1. Open the app with Spotlight: press cmd+space, type the app name (Calendar), press Return.
2. Create a new event (in Calendar, press cmd+N), and set the title, day, and time.
3. Add the person as an invitee. IMPORTANT: you usually will NOT be told their email.
   Type their NAME into the invitee/"Add Invitees" field, then look at the autocomplete
   dropdown that appears and pick the entry whose email most plausibly belongs to that
   person. If several appear, choose the best match and say why. If NO plausible match
   appears, stop and report that you could not resolve the email — do not guess an address.

CRITICAL — do not send anything without approval. When the event is fully composed and
the invitee's email is resolved, DO NOT click Send/Add/Done that would dispatch the
invite. Instead, stop and end your turn with a line beginning exactly:
  READY TO SEND:
followed by a one-line summary (event title, time, and the resolved invitee email).
A human will review and approve.

General rules: never open or read unrelated apps, messages, or files. If something
unexpected blocks you (a dialog, a login), describe it and stop rather than clicking
blindly.
"""


# --------------------------------------------------------------------------- #
# Executing one action that Claude requested, on the Mac.
# --------------------------------------------------------------------------- #
def perform(action_input: dict) -> str | None:
    """Run one computer action. Returns a string for actions that report data
    (e.g. cursor_position), else None. Raises on truly unknown actions."""
    a = action_input.get("action")
    coord = action_input.get("coordinate")

    if a == "screenshot":
        return None  # the loop captures a fresh screenshot regardless
    if a == "left_click":
        mac.left_click(*coord)
    elif a == "right_click":
        mac.right_click(*coord)
    elif a == "middle_click":
        mac.middle_click(*coord)
    elif a == "double_click":
        mac.double_click(*coord)
    elif a == "triple_click":
        mac.triple_click(*coord)
    elif a == "mouse_move":
        mac.move(*coord)
    elif a == "left_click_drag":
        sx, sy = action_input["start_coordinate"]
        ex, ey = coord
        mac.left_click_drag(sx, sy, ex, ey)
    elif a == "type":
        mac.type_text(action_input["text"])
    elif a == "key":
        mac.press_key(action_input["text"])
    elif a == "hold_key":
        # Approximate: press the chord once (duration not honored — fine for our tasks).
        mac.press_key(action_input["text"])
    elif a == "scroll":
        mac.scroll(
            coord[0] if coord else mac.cursor_position()[0],
            coord[1] if coord else mac.cursor_position()[1],
            action_input.get("scroll_direction", "down"),
            int(action_input.get("scroll_amount", 3)),
        )
    elif a == "wait":
        mac.wait(float(action_input.get("duration", 1)))
    elif a == "cursor_position":
        x, y = mac.cursor_position()
        return f"({x}, {y})"
    elif a in ("left_mouse_down", "left_mouse_up"):
        import pyautogui

        (pyautogui.mouseDown if a == "left_mouse_down" else pyautogui.mouseUp)()
    else:
        raise ValueError(f"Unknown action: {a!r}")
    return None


def describe(action_input: dict) -> str:
    """A short human-readable line for the live console, so the demo is legible."""
    a = action_input.get("action", "?")
    if "coordinate" in action_input:
        a += f" {tuple(action_input['coordinate'])}"
    if "text" in action_input:
        a += f"  {action_input['text']!r}"
    if a.startswith("scroll"):
        a += f"  {action_input.get('scroll_direction','')}x{action_input.get('scroll_amount','')}"
    return a


def text_of(content_blocks) -> str:
    return "".join(b.text for b in content_blocks if getattr(b, "type", None) == "text").strip()


# --------------------------------------------------------------------------- #
# VISION mode — Claude drives; we record the actions.
# --------------------------------------------------------------------------- #
def run_vision(task: str, name: str, max_steps: int) -> None:
    client = anthropic.Anthropic()
    w, h = mac.logical_size()
    print(f"[setup] logical screen {w}x{h}; model {MODEL}")
    print(f"[setup] task: {task}\n")
    tool, beta = _pick_tool_and_beta(client, w, h)

    b64, _, _ = mac.screenshot()
    messages: list[dict] = [{
        "role": "user",
        "content": [
            {"type": "text", "text": task},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}},
        ],
    }]

    recorded: list[dict] = []

    step = 0
    while True:
        resp = client.beta.messages.create(
            model=MODEL, max_tokens=MAX_TOKENS, system=SYSTEM_PROMPT,
            tools=[tool], betas=[beta], messages=messages,
        )
        messages.append({"role": "assistant", "content": resp.content})

        say = text_of(resp.content)
        if say:
            print(f"[claude] {say}")

        if resp.stop_reason == "pause_turn":
            continue  # server-side loop paused; just re-send to resume

        if resp.stop_reason != "tool_use":
            # Claude is done with this turn. Hand control to the human (the approval beat).
            if _human_continue(messages):
                continue
            break

        tool_results = []
        for block in resp.content:
            if getattr(block, "type", None) != "tool_use":
                continue
            action_input = block.input
            print(f"  -> {describe(action_input)}")
            reported = perform(action_input)
            recorded.append(action_input)
            time.sleep(0.4)  # let the UI settle before we look again

            if reported is not None and action_input.get("action") == "cursor_position":
                content = reported
            else:
                shot, _, _ = mac.screenshot()
                content = [{"type": "image", "source": {
                    "type": "base64", "media_type": "image/png", "data": shot}}]
            tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": content})

        messages.append({"role": "user", "content": tool_results})

        step += 1
        if step >= max_steps:
            print(f"\n[stop] hit the {max_steps}-step cap. Ending for safety.")
            break

    _save_trajectory(name, task, recorded)


def _human_continue(messages: list[dict]) -> bool:
    """After Claude ends a turn, let the human approve continuing (e.g. 'send it')
    or finish. Returns True if we appended an instruction and should keep looping."""
    print("\n[paused] Claude stopped. Press Enter to finish, or type an instruction "
          "to continue (e.g. 'approved — send the invite'):")
    try:
        reply = input("> ").strip()
    except EOFError:
        reply = ""
    if not reply:
        return False
    messages.append({"role": "user", "content": [{"type": "text", "text": reply}]})
    return True


def _pick_tool_and_beta(client, w: int, h: int) -> tuple[dict, str]:
    """Find the (tool type, beta header) pair this model accepts. Cheap probe call."""
    last_err = None
    for tool_type, beta in TOOL_BETA_CANDIDATES:
        tool = {"type": tool_type, "name": "computer",
                "display_width_px": w, "display_height_px": h}
        try:
            client.beta.messages.create(
                model=MODEL, max_tokens=16, system=SYSTEM_PROMPT,
                tools=[tool], betas=[beta],
                messages=[{"role": "user", "content": "Reply with the single word: ready"}],
            )
            print(f"[setup] using tool={tool_type}, beta={beta}")
            return tool, beta
        except anthropic.BadRequestError as e:
            last_err = e
            continue
    raise SystemExit(f"No computer-use tool/beta accepted by {MODEL}. Last error:\n{last_err}")


def _save_trajectory(name: str, task: str, actions: list[dict]) -> None:
    TRAJ_DIR.mkdir(exist_ok=True)
    path = TRAJ_DIR / f"{name}.json"
    path.write_text(json.dumps({"task": task, "actions": actions}, indent=2))
    print(f"\n[recorded] {len(actions)} actions -> {path}")
    print(f"           replay deterministically with:  python agent.py --replay {name}")


# --------------------------------------------------------------------------- #
# DRY-RUN — one model call, prints the first action it WOULD take. No clicks.
# Safe end-to-end check of: API key, beta header, screenshot encoding, grounding.
# --------------------------------------------------------------------------- #
def run_dryrun(task: str) -> None:
    client = anthropic.Anthropic()
    w, h = mac.logical_size()
    tool, beta = _pick_tool_and_beta(client, w, h)
    b64, _, _ = mac.screenshot()
    messages = [{"role": "user", "content": [
        {"type": "text", "text": task},
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}},
    ]}]
    resp = client.beta.messages.create(
        model=MODEL, max_tokens=MAX_TOKENS, system=SYSTEM_PROMPT,
        tools=[tool], betas=[beta], messages=messages,
    )
    u = resp.usage
    print(f"\n[dry-run] stop_reason={resp.stop_reason}")
    print(f"[dry-run] usage: input={u.input_tokens} output={u.output_tokens} "
          f"(one screenshot + system + task in this call)")
    say = text_of(resp.content)
    if say:
        print(f"[claude] {say}")
    actions = [b.input for b in resp.content if getattr(b, "type", None) == "tool_use"]
    if actions:
        print(f"[dry-run] first action it WOULD take: {describe(actions[0])}")
        print("[dry-run] (nothing was executed — pipeline verified end to end ✅)")
    else:
        print("[dry-run] no action proposed; Claude responded with text only.")


# --------------------------------------------------------------------------- #
# REPLAY mode — the Cyberdesk stub: run the recorded actions with NO model calls.
# --------------------------------------------------------------------------- #
def run_replay(name: str) -> None:
    path = TRAJ_DIR / f"{name}.json"
    if not path.exists():
        raise SystemExit(f"No trajectory named {name!r} at {path}")
    data = json.loads(path.read_text())
    actions = data["actions"]
    print(f"[replay] {name}: {len(actions)} recorded actions, ZERO model calls.")
    print(f"[replay] task was: {data['task']}")
    print("[replay] starting in 3s — move mouse to a corner to abort.\n")
    time.sleep(3)
    for i, action_input in enumerate(actions, 1):
        print(f"  [{i}/{len(actions)}] {describe(action_input)}")
        perform(action_input)
        time.sleep(0.5)
    print("\n[replay] done — deterministic, no screenshots sent, no tokens spent.")


# --------------------------------------------------------------------------- #
def main() -> None:
    _load_dotenv()
    p = argparse.ArgumentParser(description="A Mac computer-use agent (vision + replay).")
    p.add_argument("task", nargs="?", help="natural-language task for vision mode")
    p.add_argument("--replay", metavar="NAME", help="replay a recorded trajectory, no LLM")
    p.add_argument("--dry-run", action="store_true",
                   help="one model call; print the first action it WOULD take, execute nothing")
    p.add_argument("--name", help="name for the recorded trajectory (default: slug of task)")
    p.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    args = p.parse_args()

    if args.replay:
        run_replay(args.replay)
        return
    if args.dry_run:
        if not args.task:
            p.error("--dry-run needs a task")
        run_dryrun(args.task)
        return
    if not args.task:
        p.error("provide a task, or use --replay NAME")

    name = args.name or re.sub(r"[^a-z0-9]+", "_", args.task.lower()).strip("_")[:40] or "task"
    print("=" * 70)
    print("  Mac computer-use agent — VISION mode")
    print("  Move the mouse to a screen CORNER at any time to abort.")
    print("=" * 70)
    run_vision(args.task, name, args.max_steps)


if __name__ == "__main__":
    sys.exit(main())
