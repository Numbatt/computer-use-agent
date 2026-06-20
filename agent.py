"""agent.py — "the brain wiring" + the record/replay stub.

This is the loop that makes a computer-use agent a computer-use agent:

    screenshot  ->  Claude picks ONE action  ->  we do it on the Mac  ->  new
    screenshot  ->  ... repeat until Claude says it's done.

Claude never touches the machine. It only looks at screenshots (via Anthropic's
`computer` tool) and replies with structured actions; mac_control.py performs them.

The system prompt is GENERIC — it gives operating principles, not task steps. Claude
figures out *how* to do whatever goal you give it, live.

Two modes:
  * VISION mode (default): Claude drives toward the goal, and we LOG each action plus a
    small screen 'fingerprint' (thumbnail) to trajectories/<name>.json.
  * REPLAY mode (--replay <name>): replay those actions with NO model calls — BUT before
    each one, check the screen still matches the recorded fingerprint. If it matches,
    execute deterministically (~$0). If it diverges ("surprise"), fall back to live
    vision to finish. This is Cyberdesk's "memorize steps, replay deterministically,
    fall back to the vision model on surprises" pattern, in miniature.

Usage:
    python agent.py "Open TextEdit and write me a haiku about Houston"
    python agent.py "..." --name my_task          # name the recorded trajectory
    python agent.py --replay my_task              # verified replay; vision on surprise
    python agent.py "..." --dry-run               # 1 call, prints first action, no clicks
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import time
from pathlib import Path

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
You operate the user's Mac by looking at screenshots and issuing mouse/keyboard actions
to accomplish the user's goal. You figure out HOW yourself — no one hands you steps.

Operating principles:
- Take ONE action at a time, and re-check the screen after each before deciding the next.
- Open apps with Spotlight: press cmd+space, type the app name, press Return. If
  cmd+space seems to do nothing, click the Spotlight icon in the top-right menu bar.
- Prefer keyboard shortcuts and typed input over hunting for tiny targets when practical.
  To set a value in a field, select the whole field and type the full value at once;
  don't nudge steppers one digit at a time (they fight you).
- When a field has autocomplete (emails, contacts, search, addresses), type enough of
  the NAME to disambiguate, then read the dropdown and pick the entry that best matches.
  Say why you picked it. If nothing plausible appears, stop and report — never guess.
- If something unexpected blocks you (a dialog, a login, a permission prompt), describe
  what you see and stop rather than clicking blindly.

SAFETY — never take an irreversible or outward-facing action without approval. That
includes sending an email / message / calendar invite, submitting a form, deleting, or
purchasing. When everything is composed and you are about to do such an action, DO NOT
do it. Instead stop and end your turn with a line beginning exactly:
  READY TO SEND:
followed by a one-line summary of exactly what you are about to do. A human will approve.
"""

# Cached form of the system prompt — the stable, identical-every-turn prefix.
SYSTEM_BLOCKS = [{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}]


# --------------------------------------------------------------------------- #
# Executing one action that Claude requested, on the Mac.
# --------------------------------------------------------------------------- #
def perform(action_input: dict) -> str | None:
    """Run one computer action. Returns a string for actions that report data
    (e.g. cursor_position), else None. Raises on truly unknown actions."""
    mac.check_abort()  # margin-based corner kill switch, before every action
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


def _mark_cache(messages: list[dict]) -> None:
    """Move a single cache breakpoint to the most recent user turn. Combined with the
    cached system block, this makes every re-sent screenshot from prior turns bill at
    ~0.1x instead of full price — the dominant cost in a computer-use loop."""
    for m in messages:  # clear stale breakpoints first (max 4 allowed per request)
        if m["role"] == "user" and isinstance(m.get("content"), list):
            for blk in m["content"]:
                if isinstance(blk, dict):
                    blk.pop("cache_control", None)
    for m in reversed(messages):  # set one on the last user message's last block
        if m["role"] == "user" and isinstance(m.get("content"), list) and m["content"]:
            last = m["content"][-1]
            if isinstance(last, dict):
                last["cache_control"] = {"type": "ephemeral"}
            break


# --------------------------------------------------------------------------- #
# Shared helpers for vision mode and the replay-with-fallback path.
# --------------------------------------------------------------------------- #
def _seed_messages(task: str, screenshot_b64: str) -> list[dict]:
    return [{"role": "user", "content": [
        {"type": "text", "text": task},
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": screenshot_b64}},
    ]}]


def _print_cost(tally: dict) -> None:
    # Opus 4.8: $5/M input, $0.50/M cached read, $6.25/M cached write, $25/M output.
    cost = (tally["in"] * 5 + tally["cache_read"] * 0.5
            + tally["cache_write"] * 6.25 + tally["out"] * 25) / 1e6
    print(f"\n[cost] tokens {tally}")
    print(f"[cost] ~${cost:.3f} for this vision segment (cached reads at ~0.1x)")


def _vision_loop(client, tool, beta, messages, max_steps, recorded) -> dict:
    """The core perception->decision->action loop. If `recorded` is a list, append a
    {action, thumb} entry per executed action — the screen 'fingerprint' BEFORE the
    action ran, so replay can later tell whether the world still matches."""
    tally = {"in": 0, "cache_read": 0, "cache_write": 0, "out": 0}
    step = 0
    while True:
        _mark_cache(messages)
        resp = client.beta.messages.create(
            model=MODEL, max_tokens=MAX_TOKENS, system=SYSTEM_BLOCKS,
            tools=[tool], betas=[beta], messages=messages,
        )
        messages.append({"role": "assistant", "content": resp.content})

        u = resp.usage
        tally["in"] += u.input_tokens
        tally["cache_read"] += (u.cache_read_input_tokens or 0)
        tally["cache_write"] += (u.cache_creation_input_tokens or 0)
        tally["out"] += u.output_tokens
        print(f"[usage] in={u.input_tokens} cache_read={u.cache_read_input_tokens or 0} "
              f"cache_write={u.cache_creation_input_tokens or 0} out={u.output_tokens}")

        say = text_of(resp.content)
        if say:
            print(f"[claude] {say}")

        if resp.stop_reason == "pause_turn":
            continue  # server-side loop paused; just re-send to resume
        if resp.stop_reason != "tool_use":
            if _human_continue(messages):  # approval beat / human steering
                continue
            break

        tool_results = []
        for block in resp.content:
            if getattr(block, "type", None) != "tool_use":
                continue
            action_input = block.input
            print(f"  -> {describe(action_input)}")
            thumb = mac.screen_thumb() if recorded is not None else None
            reported = perform(action_input)
            if recorded is not None:
                recorded.append({"action": action_input,
                                 "thumb": base64.b64encode(thumb).decode("ascii")})
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
    return tally


# --------------------------------------------------------------------------- #
# VISION mode — Claude figures the task out live (generic prompt); we record it.
# --------------------------------------------------------------------------- #
def run_vision(task: str, name: str, max_steps: int) -> None:
    client = anthropic.Anthropic()
    w, h = mac.logical_size()
    print(f"[setup] logical screen {w}x{h}; model {MODEL}")
    print(f"[setup] goal: {task}\n")
    tool, beta = _pick_tool_and_beta(client, w, h)

    b64, _, _ = mac.screenshot()
    messages = _seed_messages(task, b64)
    recorded: list[dict] = []
    tally = _vision_loop(client, tool, beta, messages, max_steps, recorded)
    _print_cost(tally)
    print("[cost] (a --replay of this trajectory costs $0 unless it hits a surprise)")
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


def _save_trajectory(name: str, task: str, steps: list[dict]) -> None:
    TRAJ_DIR.mkdir(exist_ok=True)
    path = TRAJ_DIR / f"{name}.json"
    existed = path.exists()
    path.write_text(json.dumps({"task": task, "steps": steps}, indent=2))
    verb = "overwrote" if existed else "recorded"
    print(f"\n[{verb}] {len(steps)} steps (action + screen fingerprint) -> {path}")
    print(f"           replay it (deterministic, ~$0) with:  python agent.py --replay {name}")


def list_trajectories() -> None:
    files = sorted(TRAJ_DIR.glob("*.json")) if TRAJ_DIR.exists() else []
    if not files:
        print("No memorized trajectories.")
        return
    print(f"Memorized trajectories in {TRAJ_DIR}:")
    for f in files:
        try:
            d = json.loads(f.read_text())
            print(f"  {f.stem:24} {len(d.get('steps', [])):>3} steps  — {d.get('task','')[:56]}")
        except Exception:
            print(f"  {f.stem:24} (unreadable)")


def forget_trajectory(name: str) -> None:
    """Delete one memorized trajectory, or all of them with name='all'."""
    if name == "all":
        files = list(TRAJ_DIR.glob("*.json")) if TRAJ_DIR.exists() else []
        for f in files:
            f.unlink()
        print(f"Forgot {len(files)} trajectory(ies).")
        return
    path = TRAJ_DIR / f"{name}.json"
    if path.exists():
        path.unlink()
        print(f"Forgot '{name}'.")
    else:
        print(f"No trajectory named '{name}' (try --list).")


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
# REPLAY mode — the Cyberdesk pattern in miniature: replay recorded steps with NO
# model calls, but verify the screen still matches before each one, and fall back
# to live vision the moment it diverges ("surprise").
# --------------------------------------------------------------------------- #
def run_replay(name: str, threshold: float = 0.12, max_fallback_steps: int = 25) -> None:
    path = TRAJ_DIR / f"{name}.json"
    if not path.exists():
        raise SystemExit(f"No trajectory named {name!r} at {path}")
    data = json.loads(path.read_text())
    steps = data.get("steps", [])
    goal = data.get("task", "")
    if not steps or "thumb" not in steps[0]:
        raise SystemExit(f"Trajectory {name!r} has no screen fingerprints — re-record it "
                         f"with this version before replaying.")
    print(f"[replay] {name}: {len(steps)} recorded steps.")
    print(f"[replay] Replaying deterministically (no model calls); falls back to live")
    print(f"[replay] vision if the screen diverges by more than {threshold:.2f}.")
    print(f"[replay] goal was: {goal}")
    print("[replay] starting in 3s — Ctrl-C or a screen corner aborts.\n")
    time.sleep(3)

    client = tool = beta = None
    for i, step in enumerate(steps, 1):
        action = step["action"]
        expected = base64.b64decode(step["thumb"])
        diff = mac.thumb_diff(expected, mac.screen_thumb())
        if diff <= threshold:
            print(f"  [{i}/{len(steps)}] diff={diff:.3f} ok    {describe(action)}")
            perform(action)
            time.sleep(0.5)
            continue

        # --- SURPRISE: screen isn't what we recorded. Hand off to live vision. ---
        print(f"  [{i}/{len(steps)}] diff={diff:.3f} > {threshold:.2f}  *** SURPRISE ***")
        print("[replay] screen diverged from the recording — falling back to live vision.")
        if client is None:
            client = anthropic.Anthropic()
            w, h = mac.logical_size()
            tool, beta = _pick_tool_and_beta(client, w, h)
        b64, _, _ = mac.screenshot()
        recovery = (f"{goal}\n\n(You are partway through this task but the screen is not in "
                    f"the expected state. Look at the current screen and continue from here "
                    f"to finish the goal.)")
        tally = _vision_loop(client, tool, beta, _seed_messages(recovery, b64),
                             max_fallback_steps, None)
        _print_cost(tally)
        print("[replay] vision took over after the surprise and finished. Stopping replay.")
        return

    print("\n[replay] done — every step matched the recording. Deterministic, $0, no tokens.")


# --------------------------------------------------------------------------- #
def main() -> None:
    _load_dotenv()
    p = argparse.ArgumentParser(description="A Mac computer-use agent (vision + replay).")
    p.add_argument("task", nargs="?", help="natural-language task for vision mode")
    p.add_argument("--replay", metavar="NAME",
                   help="replay a recorded trajectory; falls back to vision on a surprise")
    p.add_argument("--dry-run", action="store_true",
                   help="one model call; print the first action it WOULD take, execute nothing")
    p.add_argument("--name", help="name for the recorded trajectory (default: slug of task)")
    p.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    p.add_argument("--threshold", type=float, default=0.12,
                   help="replay: screen-divergence (0..1) above which we fall back to vision")
    p.add_argument("--list", action="store_true", help="list memorized trajectories")
    p.add_argument("--forget", metavar="NAME",
                   help="delete a memorized trajectory (or 'all' to clear them)")
    args = p.parse_args()

    if args.list:
        list_trajectories()
        return
    if args.forget:
        forget_trajectory(args.forget)
        return
    if args.replay:
        run_replay(args.replay, threshold=args.threshold)
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
