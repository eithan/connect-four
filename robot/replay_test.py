"""
Connect Four Vision — Replay & Test Runner

Feeds saved session screenshots back through the detection pipeline to verify
that a bug fix produces the correct board state on every captured event frame.

MODES
─────
  --inspect SESSION_SHOTS_DIR
      Re-run detection on every event frame in a session.  No assertions.
      Shows current raw_board output next to what was logged so you can see
      exactly how a code change shifts detection on historical data.

  --test-case TEST_CASE_DIR
      Run detection on all frames listed in expected.json, assert that the
      resulting stable board matches expected_stable_board for every frame.
      Prints PASS / FAIL per frame and exits 0 on full pass, 1 on any failure.

  --create-case SESSION_SHOTS_DIR --output TEST_CASE_DIR [--description TEXT]
      Copy raw frames from a session into a new test-case directory and
      generate expected.json from the session's logged stable boards.
      Edit expected.json afterwards to correct the bug-frame entries before
      committing the test case to git.

  --all
      Run all test cases found under test_cases/ and report aggregate results.

WORKFLOW
────────
  1. A bug occurs during a game session.  The session is saved automatically
     under logs/game_YYYYMMDD_HHMMSS_shots/ with raw frames + sidecar JSON.
  2. You fix the bug, then run:
       python replay_test.py --create-case logs/game_DATE_shots --output test_cases/case_my_bug
  3. Edit test_cases/case_my_bug/expected.json — change expected_stable_board
     for the frame(s) that were wrong (the rest are already correct).
  4. Confirm the test passes:
       python replay_test.py --test-case test_cases/case_my_bug
  5. Commit test_cases/case_my_bug/ to git.  logs/ stays gitignored.

REPLAY MECHANICS
────────────────
  For each frame in a test case the script:
    1. Feeds the raw image repeatedly until the detector is locked (first frame only).
    2. Feeds the raw image FRAME_SETTLE times so the temporal smoother converges.
    3. Reads the resulting stable board and compares to expected_stable_board.

  This mirrors real camera use: the camera dwells on each board state for many
  frames before the human touches a piece.  FRAME_SETTLE is intentionally larger
  than ADD_THRESHOLD (3) and smaller than REMOVE_THRESHOLD (15) so pieces that
  appear in the image commit, and pieces that vanish don't get removed.
"""

import argparse
import json
import os
import re
import shutil
import sys

import cv2
import numpy as np

# ── Detector import ─────────────────────────────────────────────────────────

from board_detector import (
    YOLOEnhancedBoardDetector,
    LockedBoardDetector,
    PHYSICAL_CONFIG,
    board_to_string,
)

# ── Constants ────────────────────────────────────────────────────────────────

LOCK_MAX_FEEDS   = 60    # max frames fed to achieve lock (LOCK_FRAMES=4 so 60 is plenty)
FRAME_SETTLE     = 12    # frames fed per event image (> ADD_THRESHOLD=3, < REMOVE_THRESHOLD=15)
TEST_CASES_ROOT  = "test_cases"


def _dump_expected(obj) -> str:
    """Serialize expected.json with boards on compact row-per-line format.

    Normal json.dumps nests each integer on its own line (verbose).  This
    writes board rows as single-line arrays so the file is easy to read and
    edit by hand:

        "expected_stable_board": [
          [0, 0, 0, 0, 0, 0, 0],
          ...
          [0, 0, 0, 0, 1, 0, 0]   ← easy to spot and fix the phantom piece
        ]
    """
    raw = json.dumps(obj, indent=2)
    # Collapse any 7-element sub-array that contains only digits, commas,
    # spaces and brackets onto a single line.
    raw = re.sub(
        r'\[\s*((?:\d+,\s*){6}\d+)\s*\]',
        lambda m: '[' + re.sub(r'\s+', ' ', m.group(1).strip()) + ']',
        raw,
    )
    return raw


# ── Helpers ──────────────────────────────────────────────────────────────────

def _board_equal(a: list, b: list) -> bool:
    return np.array_equal(np.array(a, dtype=np.int8), np.array(b, dtype=np.int8))


def _board_diff(got: np.ndarray, want: list) -> list[tuple]:
    """Return list of (row, col, got_val, want_val) for every mismatch."""
    want_arr = np.array(want, dtype=np.int8)
    diffs = []
    for r, c in np.argwhere(got != want_arr):
        diffs.append((int(r), int(c), int(got[r, c]), int(want_arr[r, c])))
    return diffs


CELL_NAMES = {0: "empty", 1: "red", 2: "yellow"}


def _fmt_cell(v: int) -> str:
    return CELL_NAMES.get(v, str(v))


def _fmt_board(board: list | np.ndarray) -> str:
    arr = np.array(board, dtype=np.int8)
    return board_to_string(arr)


def _load_image(path: str) -> np.ndarray | None:
    img = cv2.imread(path)
    if img is None:
        print(f"  [ERROR] Could not load image: {path}")
    return img


def _make_detector(yolo_model_path: str | None) -> LockedBoardDetector:
    """Create detector; use YOLO if model available, else HSV-only."""
    det = YOLOEnhancedBoardDetector(model_path=yolo_model_path, config=PHYSICAL_CONFIG)
    return det


def _lock_detector(detector: LockedBoardDetector, image: np.ndarray) -> bool:
    """Feed image repeatedly until detector locks.  Returns True on success."""
    for _ in range(LOCK_MAX_FEEDS):
        detector.detect(image)
        if detector.is_locked:
            return True
    return False


def _settle_detector(detector: LockedBoardDetector, image: np.ndarray) -> np.ndarray:
    """Feed image FRAME_SETTLE times and return final stable board."""
    result = None
    for _ in range(FRAME_SETTLE):
        result = detector.detect(image)
    return result.board if result is not None else np.zeros((6, 7), dtype=np.int8)


# ── --inspect mode ───────────────────────────────────────────────────────────

def cmd_inspect(session_dir: str):
    """Re-run detection on all event frames and show diffs vs logged output."""
    manifest_path = os.path.join(session_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        print(f"[ERROR] No manifest.json found in {session_dir}")
        print("  This session may pre-date the replay feature.")
        _inspect_legacy(session_dir)
        return

    with open(manifest_path) as f:
        manifest = json.load(f)

    raw_frames = manifest.get("raw_frames", [])
    if not raw_frames:
        print("[ERROR] manifest.json has no raw_frames entries.")
        return

    print(f"\nInspecting session: {manifest['session']}")
    print(f"  Started:      {manifest.get('started', '?')}")
    print(f"  Complete:     {manifest.get('complete', '?')}")
    print(f"  Quit reason:  {manifest.get('quit_reason', '?')}")
    print(f"  Frames:       {len(raw_frames)}")

    detector = _make_detector(yolo_model_path=None)

    any_diff = False
    for i, raw_fname in enumerate(raw_frames):
        img_path = os.path.join(session_dir, raw_fname)
        base = raw_fname.replace("_raw.jpg", "")
        json_path = os.path.join(session_dir, base + ".json")

        logged = {}
        if os.path.exists(json_path):
            with open(json_path) as f:
                logged = json.load(f)

        print(f"\n[{i+1}/{len(raw_frames)}] {base}  (event={logged.get('event', '?')})")

        img = _load_image(img_path)
        if img is None:
            continue

        if i == 0:
            ok = _lock_detector(detector, img)
            if not ok:
                print("  [WARN] Detector did not lock on first frame — results may be unreliable")
                continue
            print(f"  Detector locked ✓")

        stable = _settle_detector(detector, img)

        # Compare to logged raw_board (single-frame detection, the most diagnostic)
        logged_raw   = logged.get("raw_board")
        logged_stable = logged.get("stable_board")

        print(f"  Current stable board:")
        for line in _fmt_board(stable).splitlines():
            print(f"    {line}")

        if logged_raw:
            diffs = _board_diff(stable, logged_raw)
            if diffs:
                any_diff = True
                print(f"  DIFFERS from logged raw_board ({len(diffs)} cell(s)):")
                for r, c, got, want in diffs:
                    print(f"    ({r},{c})  now={_fmt_cell(got)}  was={_fmt_cell(want)}")
            else:
                print(f"  Matches logged raw_board ✓")
        elif logged_stable:
            diffs = _board_diff(stable, logged_stable)
            if diffs:
                any_diff = True
                print(f"  DIFFERS from logged stable_board ({len(diffs)} cell(s)):")
                for r, c, got, want in diffs:
                    print(f"    ({r},{c})  now={_fmt_cell(got)}  was={_fmt_cell(want)}")
            else:
                print(f"  Matches logged stable_board ✓")
        else:
            print(f"  (no logged board data to compare)")

    print()
    if any_diff:
        print("Summary: detection output differs from logged data on at least one frame.")
    else:
        print("Summary: detection output matches logged data on all frames.")


def _inspect_legacy(session_dir: str):
    """Fallback for sessions without manifest.json — scan for raw frames directly."""
    raw_frames = sorted(f for f in os.listdir(session_dir) if f.endswith("_raw.jpg"))
    if not raw_frames:
        print("[ERROR] No *_raw.jpg frames found either.  Cannot inspect.")
        return
    print(f"Found {len(raw_frames)} raw frames (legacy mode — no manifest).")
    detector = _make_detector(yolo_model_path=None)
    for i, fname in enumerate(raw_frames):
        img = _load_image(os.path.join(session_dir, fname))
        if img is None:
            continue
        if i == 0:
            if not _lock_detector(detector, img):
                print("[WARN] Did not lock on first frame")
                continue
        stable = _settle_detector(detector, img)
        print(f"\n{fname}:")
        for line in _fmt_board(stable).splitlines():
            print(f"  {line}")


# ── --test-case mode ──────────────────────────────────────────────────────────

def cmd_test_case(case_dir: str, verbose: bool = True) -> bool:
    """Run a single test case.  Returns True if all frames pass."""
    expected_path = os.path.join(case_dir, "expected.json")
    if not os.path.exists(expected_path):
        print(f"[ERROR] No expected.json in {case_dir}")
        return False

    with open(expected_path) as f:
        spec = json.load(f)

    description = spec.get("description", os.path.basename(case_dir))
    frames      = spec.get("frames", [])
    yolo_path   = spec.get("yolo_model")   # None → use default / auto-detect

    print(f"\n{'='*60}")
    print(f"Test case: {description}")
    print(f"  Dir:    {case_dir}")
    print(f"  Frames: {len(frames)}")
    print(f"{'='*60}")

    if not frames:
        print("[ERROR] expected.json has no frames.")
        return False

    detector = _make_detector(yolo_model_path=yolo_path)

    passed = 0
    failed = 0
    failed_frames = []

    for i, frame_spec in enumerate(frames):
        fname   = frame_spec["file"]
        event   = frame_spec.get("event", "?")
        expected_stable = frame_spec.get("expected_stable_board")

        img_path = os.path.join(case_dir, fname)
        img = _load_image(img_path)
        if img is None:
            failed += 1
            failed_frames.append(fname)
            continue

        if i == 0:
            ok = _lock_detector(detector, img)
            if not ok:
                print(f"  [FAIL] Frame 1 ({fname}): detector did not lock")
                failed += 1
                failed_frames.append(fname)
                continue
            if verbose:
                print(f"\n  Frame 1 ({fname}) — board locked ✓")

        stable = _settle_detector(detector, img)

        if expected_stable is None:
            if verbose:
                print(f"\n  Frame {i+1} ({fname}, event={event}) — no expected board, skipping assertion")
            continue

        diffs = _board_diff(stable, expected_stable)
        if not diffs:
            passed += 1
            if verbose:
                print(f"  Frame {i+1} ({fname}, event={event})  PASS ✓")
        else:
            failed += 1
            failed_frames.append(fname)
            print(f"\n  Frame {i+1} ({fname}, event={event})  FAIL ✗")
            print(f"    Got stable board:")
            for line in _fmt_board(stable).splitlines():
                print(f"      {line}")
            print(f"    Expected stable board:")
            for line in _fmt_board(expected_stable).splitlines():
                print(f"      {line}")
            print(f"    Mismatches ({len(diffs)}):")
            for r, c, got, want in diffs:
                print(f"      ({r},{c})  got={_fmt_cell(got)}  want={_fmt_cell(want)}")

    total = passed + failed
    print(f"\nResult: {passed}/{total} frames passed")
    if failed_frames:
        print(f"Failed frames: {', '.join(failed_frames)}")
    overall = failed == 0
    print(f"{'PASS' if overall else 'FAIL'} — {description}")
    return overall


# ── --create-case mode ────────────────────────────────────────────────────────

def cmd_create_case(session_dir: str, output_dir: str, description: str):
    """Build a test case directory from a captured session."""
    manifest_path = os.path.join(session_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        print(f"[ERROR] No manifest.json in {session_dir}")
        print("  Run a game session with the updated game_loop.py to generate one.")
        sys.exit(1)

    with open(manifest_path) as f:
        manifest = json.load(f)

    raw_frames = manifest.get("raw_frames", [])
    if not raw_frames:
        print("[ERROR] manifest has no raw_frames.")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    frames_spec = []
    for raw_fname in raw_frames:
        src = os.path.join(session_dir, raw_fname)
        if not os.path.exists(src):
            print(f"  [WARN] Raw frame missing: {src} — skipping")
            continue

        dest = os.path.join(output_dir, raw_fname)
        shutil.copy2(src, dest)

        # Read logged stable board from sidecar JSON
        base = raw_fname.replace("_raw.jpg", "")
        json_src = os.path.join(session_dir, base + ".json")
        stable_board = None
        event = "?"
        if os.path.exists(json_src):
            with open(json_src) as f:
                meta = json.load(f)
            stable_board = meta.get("stable_board")
            event = meta.get("event", "?")

        frames_spec.append({
            "file":                   raw_fname,
            "event":                  event,
            "expected_stable_board":  stable_board,
        })
        print(f"  Copied {raw_fname}  (event={event})")

    expected = {
        "description": description or f"Test case from {manifest['session']}",
        "human_color": manifest.get("human_color", 1),
        "yolo_model":  None,   # null = use default model path
        "frames":      frames_spec,
    }

    out_path = os.path.join(output_dir, "expected.json")
    with open(out_path, "w") as f:
        f.write(_dump_expected(expected))

    print(f"\nTest case created: {output_dir}")
    print(f"  {len(frames_spec)} frames  |  expected.json written")
    print()
    print("Next steps:")
    print("  1. Open expected.json and correct expected_stable_board for any bug frames.")
    print("     (The logged values reflect what the detector DID produce, including bugs.)")
    print("  2. Run:  python replay_test.py --test-case", output_dir)
    print("  3. Iterate until all frames pass, then commit test_cases/ to git.")


# ── --all mode ────────────────────────────────────────────────────────────────

def cmd_all(root: str = TEST_CASES_ROOT) -> bool:
    """Run every test case under root/. Returns True if all pass."""
    if not os.path.isdir(root):
        print(f"[ERROR] Test cases root not found: {root}")
        return False

    case_dirs = sorted(
        os.path.join(root, d) for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
           and os.path.exists(os.path.join(root, d, "expected.json"))
    )

    if not case_dirs:
        print(f"No test cases found under {root}/")
        return True

    print(f"Running {len(case_dirs)} test case(s) from {root}/\n")
    results = {}
    for case_dir in case_dirs:
        name = os.path.basename(case_dir)
        results[name] = cmd_test_case(case_dir, verbose=False)

    print(f"\n{'='*60}")
    print("Test suite summary:")
    passed = sum(1 for v in results.values() if v)
    failed = sum(1 for v in results.values() if not v)
    for name, ok in results.items():
        status = "PASS ✓" if ok else "FAIL ✗"
        print(f"  {status}  {name}")
    print(f"\n{passed}/{len(results)} test cases passed")
    print(f"{'='*60}")
    return failed == 0


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Connect Four vision replay & test runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--inspect",     metavar="SESSION_DIR",
                       help="Inspect a session — re-run detection, show diffs vs logged")
    group.add_argument("--test-case",   metavar="CASE_DIR",
                       help="Run a single test case, assert expected boards match")
    group.add_argument("--create-case", metavar="SESSION_DIR",
                       help="Create a test case directory from a session")
    group.add_argument("--all",         action="store_true",
                       help="Run all test cases under test_cases/")

    parser.add_argument("--output",      metavar="CASE_DIR",
                        help="Output directory for --create-case")
    parser.add_argument("--description", metavar="TEXT",
                        help="Description for --create-case")
    parser.add_argument("--verbose",     action="store_true", default=True,
                        help="Verbose per-frame output (default: on)")

    args = parser.parse_args()

    if args.inspect:
        cmd_inspect(args.inspect)

    elif args.test_case:
        ok = cmd_test_case(args.test_case, verbose=args.verbose)
        sys.exit(0 if ok else 1)

    elif args.create_case:
        if not args.output:
            parser.error("--create-case requires --output TEST_CASE_DIR")
        cmd_create_case(args.create_case, args.output, args.description or "")

    elif args.all:
        ok = cmd_all()
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
