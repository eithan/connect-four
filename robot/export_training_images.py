"""
export_training_images.py

Exports a diverse set of raw frames from game logs for YOLO fine-tuning.

Usage:
    pip install imagehash
    python export_training_images.py

Output:
    training_export/images/   — flat folder of JPGs ready to upload to Roboflow
"""

import json
import shutil
import random
from pathlib import Path

try:
    import imagehash
    from PIL import Image
except ImportError:
    raise SystemExit(
        "Missing dependency: pip install imagehash Pillow"
    )

# ── Config ────────────────────────────────────────────────────────────────────
LOGS_DIR = Path("logs")
OUTPUT_DIR = Path("training_export/images")
MAX_PER_SESSION = 10      # cap per game session for diversity
TARGET_TOTAL = 400        # hard cap; randomly sample down if above this
HASH_THRESHOLD = 8        # hamming distance; images within this are duplicates
RANDOM_SEED = 42
# ─────────────────────────────────────────────────────────────────────────────


def phash(img_path: Path) -> imagehash.ImageHash:
    return imagehash.phash(Image.open(img_path))


def collect_sessions(logs_dir: Path) -> list[Path]:
    """Return sorted list of game session directories."""
    return sorted(d for d in logs_dir.iterdir() if d.is_dir() and d.name.startswith("game_"))


def piece_count(raw_jpg: Path) -> int:
    """Return number of non-empty cells from the sidecar JSON, or 0 if missing."""
    json_path = raw_jpg.with_name(raw_jpg.name.replace("_raw.jpg", ".json"))
    if not json_path.exists():
        return 0
    try:
        data = json.loads(json_path.read_text())
        board = data.get("stable_board") or data.get("raw_board") or []
        return sum(cell != 0 for row in board for cell in row)
    except Exception:
        return 0


def pick_from_session(session_dir: Path, max_frames: int) -> list[Path]:
    """Return up to max_frames raw jpg paths from a session.

    Sorts frames by piece count descending so piece-rich frames are preferred,
    then pads with the remaining frames (sorted by sequence) to fill the cap.
    This gives good class balance (red/yellow pieces) while still including
    some empty-board frames for the empty_hole class.
    """
    raw_frames = sorted(session_dir.glob("*_raw.jpg"))
    if not raw_frames:
        return []
    if len(raw_frames) <= max_frames:
        return raw_frames

    scored = sorted(raw_frames, key=piece_count, reverse=True)
    # Take the top half by piece count, fill remainder with evenly-spaced picks
    top_n = max(1, max_frames // 2)
    top = scored[:top_n]
    rest = [f for f in raw_frames if f not in set(top)]
    step = max(1, len(rest) / (max_frames - top_n))
    filler = [rest[int(i * step)] for i in range(max_frames - top_n)]
    return top + filler


def dedup(candidates: list[Path], threshold: int) -> list[Path]:
    """Remove perceptually duplicate images. Returns unique subset."""
    accepted: list[Path] = []
    accepted_hashes: list[imagehash.ImageHash] = []
    for path in candidates:
        try:
            h = phash(path)
        except Exception:
            continue
        if all(h - ah >= threshold for ah in accepted_hashes):
            accepted.append(path)
            accepted_hashes.append(h)
    return accepted


def main() -> None:
    random.seed(RANDOM_SEED)

    if not LOGS_DIR.exists():
        raise SystemExit(f"Logs directory not found: {LOGS_DIR.resolve()}")

    sessions = collect_sessions(LOGS_DIR)
    print(f"Found {len(sessions)} game sessions")

    # Phase 1: collect candidates (max per session)
    candidates: list[Path] = []
    for session in sessions:
        picks = pick_from_session(session, MAX_PER_SESSION)
        candidates.extend(picks)

    print(f"Candidates after per-session cap: {len(candidates)}")

    # Phase 2: perceptual dedup
    print("Running perceptual dedup (this may take a minute)...")
    unique = dedup(candidates, HASH_THRESHOLD)
    print(f"Unique images after dedup: {len(unique)}")

    # Phase 3: sample down to TARGET_TOTAL if needed
    if len(unique) > TARGET_TOTAL:
        unique = random.sample(unique, TARGET_TOTAL)
        print(f"Sampled down to {TARGET_TOTAL}")

    # Phase 4: copy to output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for src in unique:
        # Prefix with session name to avoid collisions
        session_name = src.parent.name
        dest_name = f"{session_name}__{src.name}"
        shutil.copy2(src, OUTPUT_DIR / dest_name)

    print(f"\nDone. {len(unique)} images written to: {OUTPUT_DIR.resolve()}")
    print("\nNext steps:")
    print("  1. Upload training_export/images/ to a new Roboflow Object Detection project")
    print("  2. Label 3 classes: red_piece, yellow_piece, empty_hole")
    print("  3. Export as YOLOv8 format and unzip to training_data/connect4_v2/")


if __name__ == "__main__":
    main()
