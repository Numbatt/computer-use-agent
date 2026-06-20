"""mac_control.py — "the hands".

The local macOS executor for the computer-use agent. Claude (the brain) only ever
emits structured actions like {"action": "left_click", "coordinate": [840, 210]};
this module is what actually performs them on the Mac, and what produces the
screenshots Claude looks at.

Two macOS specifics are handled here so nothing downstream has to think about them:

1. Retina scaling. `screencapture` writes a NATIVE-pixel image (e.g. 2880x1864),
   but the mouse/keyboard APIs work in LOGICAL points (e.g. 1470x956). We downscale
   every screenshot to the logical size before handing it to Claude, and we tell
   Claude the logical size as the display dimensions. Result: the coordinates Claude
   returns are already in logical points — the exact space pyautogui clicks in — so
   there is no coordinate math anywhere else. 1:1.

2. Permissions. The terminal running this must have Screen Recording (for the
   screenshot) and Accessibility (for clicks/keys) granted in
   System Settings -> Privacy & Security. Without them, screenshots come back blank
   and clicks silently do nothing.
"""

from __future__ import annotations

import base64
import io
import subprocess
import tempfile
import time
from pathlib import Path

import pyautogui
from PIL import Image

# Slam the mouse into a screen corner to abort everything — our physical kill switch.
pyautogui.FAILSAFE = True
# Small pause after each pyautogui action so the UI can keep up.
pyautogui.PAUSE = 0.15


def logical_size() -> tuple[int, int]:
    """The screen size in logical points — the coordinate space Claude operates in."""
    w, h = pyautogui.size()
    return int(w), int(h)


def screenshot() -> tuple[str, int, int]:
    """Capture the screen and return (base64_png, logical_w, logical_h).

    Captured at native resolution via `screencapture`, then downscaled to logical
    size so Claude's coordinates land 1:1 on what pyautogui clicks.
    """
    logical_w, logical_h = logical_size()
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        # -x: silent (no shutter sound). Captures the main display.
        result = subprocess.run(
            ["screencapture", "-x", "-t", "png", tmp_path],
            capture_output=True, text=True,
        )
        if result.returncode != 0 or not Path(tmp_path).stat().st_size:
            raise RuntimeError(
                "screencapture failed — this is almost always the macOS Screen "
                "Recording permission.\nFix: System Settings -> Privacy & Security "
                "-> Screen Recording -> enable your terminal app (Terminal/iTerm), "
                "then QUIT and reopen it.\n"
                f"(screencapture said: {result.stderr.strip() or 'could not create image from display'})"
            )
        img = Image.open(tmp_path).convert("RGB")
        if img.size != (logical_w, logical_h):
            img = img.resize((logical_w, logical_h), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.standard_b64encode(buf.getvalue()).decode("ascii")
        return b64, logical_w, logical_h
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# --- xdotool key names (what the computer tool emits) -> pyautogui key names ----
_KEYMAP = {
    "return": "enter",
    "enter": "enter",
    "tab": "tab",
    "space": "space",
    "backspace": "backspace",
    "delete": "delete",
    "escape": "esc",
    "esc": "esc",
    "up": "up",
    "down": "down",
    "left": "left",
    "right": "right",
    "home": "home",
    "end": "end",
    "page_up": "pageup",
    "page_down": "pagedown",
    "prior": "pageup",
    "next": "pagedown",
    # modifiers — on macOS "cmd"/"super" is the Command key
    "cmd": "command",
    "command": "command",
    "super": "command",
    "super_l": "command",
    "super_r": "command",
    "meta": "command",
    "ctrl": "ctrl",
    "control": "ctrl",
    "control_l": "ctrl",
    "control_r": "ctrl",
    "alt": "option",
    "alt_l": "option",
    "alt_r": "option",
    "option": "option",
    "shift": "shift",
    "shift_l": "shift",
    "shift_r": "shift",
}


def _map_key(name: str) -> str:
    key = name.strip()
    return _KEYMAP.get(key.lower(), key.lower())


def press_key(combo: str) -> None:
    """Press a key or chord. `combo` is xdotool-style, e.g. 'Return' or 'cmd+space'."""
    keys = [_map_key(part) for part in combo.replace(" ", "").split("+") if part]
    if len(keys) == 1:
        pyautogui.press(keys[0])
    else:
        pyautogui.hotkey(*keys)


def type_text(text: str) -> None:
    pyautogui.typewrite(text, interval=0.01)


def move(x: int, y: int) -> None:
    pyautogui.moveTo(x, y)


def click(x: int, y: int, button: str = "left", clicks: int = 1) -> None:
    pyautogui.click(x=x, y=y, button=button, clicks=clicks, interval=0.08)


def left_click(x: int, y: int) -> None:
    click(x, y, "left")


def right_click(x: int, y: int) -> None:
    click(x, y, "right")


def middle_click(x: int, y: int) -> None:
    click(x, y, "middle")


def double_click(x: int, y: int) -> None:
    click(x, y, "left", clicks=2)


def triple_click(x: int, y: int) -> None:
    click(x, y, "left", clicks=3)


def left_click_drag(x1: int, y1: int, x2: int, y2: int) -> None:
    pyautogui.moveTo(x1, y1)
    pyautogui.dragTo(x2, y2, duration=0.4, button="left")


def scroll(x: int, y: int, direction: str, amount: int = 3) -> None:
    pyautogui.moveTo(x, y)
    clicks = amount * 100
    if direction == "down":
        clicks = -clicks
    if direction in ("left", "right"):
        pyautogui.hscroll(clicks if direction == "right" else -clicks, x=x, y=y)
    else:
        pyautogui.scroll(clicks, x=x, y=y)


def cursor_position() -> tuple[int, int]:
    p = pyautogui.position()
    return int(p.x), int(p.y)


def wait(seconds: float) -> None:
    time.sleep(min(seconds, 5.0))  # cap so a runaway "wait" can't hang the demo


if __name__ == "__main__":
    # Quick self-test: capture a screenshot and report sizes. No clicks (safe to run).
    b64, w, h = screenshot()
    print(f"logical size: {w}x{h}")
    print(f"screenshot bytes (decoded): {len(base64.b64decode(b64))}")
    print("OK — if the size looks like your screen in points, screenshots work.")
