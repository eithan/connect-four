#!/usr/bin/env bash
# =============================================================
#  view_episode.sh
#  Play back episodes from a local LeRobot dataset with
#  navigation, rich parquet info, soft-delete, and compaction.
#
#  USAGE:
#    ./view_episode.sh                        # latest episode (interactive)
#    ./view_episode.sh 5                      # episode 5
#    ./view_episode.sh --col 0 5              # episode 5, col 0
#    ./view_episode.sh --dataset /path/to/ds  # explicit dataset path
#    ./view_episode.sh --list                 # table of all episodes
#    ./view_episode.sh --latest               # latest episode
#    ./view_episode.sh --compact              # hard-delete marked episodes
#
#  INTERACTIVE CONTROLS (after each episode plays):
#    n  →  next episode          p  →  previous episode
#    d  →  mark for deletion     u  →  unmark deletion
#    r  →  replay                q  →  quit
#
#  SOFT-DELETE:
#    Episodes are marked in  <dataset>/.deleted_episodes
#    They are skipped during navigation and flagged in --list.
#    Run --compact to permanently remove them from the dataset.
#
#  NOTES:
#    lerobot v3 concatenates all episodes into a single video
#    file (chunk-000/file-000.mp4). Seek positions are derived
#    from parquet frame counts + a lead_in correction for the
#    ~1s of reset-phase frames lerobot appends per episode.
# =============================================================

set -euo pipefail

# ── Args ──────────────────────────────────────────────────────────
COL=3
EPISODE_ARG=""
LIST_MODE=false
COMPACT_MODE=false
DATASET_OVERRIDE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --col|-c)         shift; COL="$1"; shift ;;
        --dataset|-d)     shift; DATASET_OVERRIDE="$1"; shift ;;
        --list|-l)        LIST_MODE=true; shift ;;
        --latest)         EPISODE_ARG="latest"; shift ;;
        --compact)        COMPACT_MODE=true; shift ;;
        [0-9]*)           EPISODE_ARG="$1"; shift ;;
        *)
            echo "Usage: $0 [--col N] [--dataset PATH] [EPISODE | --list | --latest | --compact]"
            exit 1 ;;
    esac
done

# ── Config ────────────────────────────────────────────────────────
HF_USER="eithanz"
REPO_ID="${HF_USER}/connect_four_chute5_pick_col${COL}"
if [[ -n "$DATASET_OVERRIDE" ]]; then
    DATASET_ROOT="$DATASET_OVERRIDE"
else
    DATASET_ROOT="${HOME}/lerobot_datasets/${REPO_ID}"
fi
VIDEO_ROOT="${DATASET_ROOT}/videos"
FRONT=""   # set by check_dataset
HAND=""    # set by check_dataset (may stay empty)
DELETED_FILE="${DATASET_ROOT}/.deleted_episodes"

echo "  Dataset: ${DATASET_ROOT}"

die() { echo "ERROR: $*" >&2; exit 1; }

check_dataset() {
    [[ -d "$DATASET_ROOT" ]] || die "Dataset not found: $DATASET_ROOT"
    [[ -d "$VIDEO_ROOT" ]]   || die "No videos/ directory in: $DATASET_ROOT"

    # Discover camera streams — pick first as FRONT, second as HAND
    local cams=()
    while IFS= read -r d; do
        local candidate="${d}/chunk-000/file-000.mp4"
        [[ -f "$candidate" ]] && cams+=("$candidate")
    done < <(find "$VIDEO_ROOT" -mindepth 1 -maxdepth 1 -type d | sort)

    [[ ${#cams[@]} -gt 0 ]] || die "No video files found under $VIDEO_ROOT"
    FRONT="${cams[0]}"
    HAND="${cams[1]:-}"   # empty string if only one camera

    echo "  Cameras: $(for c in "${cams[@]}"; do basename "$(dirname "$(dirname "$c")")"; done | tr '\n' ' ')"
}

# ── Soft-delete helpers ───────────────────────────────────────────
is_deleted() {
    [[ -f "$DELETED_FILE" ]] && grep -qx "$1" "$DELETED_FILE" 2>/dev/null
}

mark_deleted() {
    echo "$1" >> "$DELETED_FILE"
    sort -nu "$DELETED_FILE" -o "$DELETED_FILE"
    echo "  ✗ Episode $1 marked for deletion."
}

unmark_deleted() {
    if [[ -f "$DELETED_FILE" ]]; then
        grep -vx "$1" "$DELETED_FILE" > "${DELETED_FILE}.tmp" 2>/dev/null || true
        mv "${DELETED_FILE}.tmp" "$DELETED_FILE"
    fi
    echo "  ✓ Episode $1 unmarked."
}

count_deleted() {
    if [[ -f "$DELETED_FILE" ]]; then
        wc -l < "$DELETED_FILE" | tr -d ' '
    else
        echo 0
    fi
}

get_deleted_list() {
    [[ -f "$DELETED_FILE" ]] && cat "$DELETED_FILE" || true
}

# ── Dataset helpers ───────────────────────────────────────────────
get_total() {
    python3 -c "
import json
d = json.load(open('${DATASET_ROOT}/meta/info.json'))
print(d.get('total_episodes', 0))
"
}

# ── compute_timings ───────────────────────────────────────────────
# Outputs one line per episode:  episode_idx  seek_seconds  duration_seconds
compute_timings() {
    python3 - "$DATASET_ROOT" <<'PYEOF'
import sys, json, pathlib, subprocess
import pandas as pd

root = pathlib.Path(sys.argv[1])
fps  = float(json.loads((root / "meta" / "info.json").read_text()).get("fps", 15))

# ── Episode lengths ───────────────────────────────────────────────
lengths = None
for f in sorted((root / "meta" / "episodes").glob("chunk-*/*.parquet")):
    df = pd.read_parquet(f)
    if "length" in df.columns and "episode_index" in df.columns:
        df = df.sort_values("episode_index").reset_index(drop=True)
        lengths = df["length"].tolist()
        break

if lengths is None:
    for f in sorted((root / "data").glob("chunk-*/*.parquet")):
        df = pd.read_parquet(f, columns=["episode_index"])
        ep_col = df["episode_index"]
        if ep_col.dtype == object:
            ep_col = ep_col.apply(lambda x: int(x[0]) if hasattr(x, "__len__") else int(x))
            df["episode_index"] = ep_col
        counts  = df.groupby("episode_index").size()
        lengths = [int(counts.get(i, 0)) for i in range(max(counts.index) + 1)]
        break

if lengths is None:
    print("ERROR: could not load episode lengths", file=sys.stderr)
    sys.exit(1)

# ── Lead-in correction ────────────────────────────────────────────
video_path = next(
    (p for p in sorted((root / "videos").glob("*/chunk-000/file-000.mp4"))),
    None
)
lead_in         = 0.0
total_parquet_s = sum(lengths) / fps
try:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1", str(video_path)],
        capture_output=True, text=True, timeout=10)
    video_dur = float(r.stdout.split("=")[1].strip())
    excess    = video_dur - total_parquet_s
    if excess > 0:
        lead_in = excess / len(lengths)
except Exception:
    pass

# ── Emit timings ──────────────────────────────────────────────────
cumulative = 0.0
for i, length in enumerate(lengths):
    seek_s = i * lead_in + cumulative / fps
    dur_s  = length / fps
    print(f"{i} {seek_s:.3f} {dur_s:.3f}")
    cumulative += length
PYEOF
}

# ── get_episode_info ──────────────────────────────────────────────
# Prints rich info about an episode from the data parquet.
get_episode_info() {
    python3 - "$DATASET_ROOT" "$1" <<'PYEOF'
import sys, json, pathlib
import pandas as pd

root   = pathlib.Path(sys.argv[1])
ep_idx = int(sys.argv[2])
info   = json.loads((root / "meta" / "info.json").read_text())
fps    = float(info.get("fps", 15))

# ── Load episode rows ─────────────────────────────────────────────
data = None
for f in sorted((root / "data").glob("chunk-*/*.parquet")):
    df = pd.read_parquet(f)
    ep_col = df["episode_index"]
    if ep_col.dtype == object:
        ep_col = ep_col.apply(lambda x: int(x[0]) if hasattr(x, "__len__") else int(x))
        df["episode_index"] = ep_col
    sub = df[df["episode_index"] == ep_idx]
    if len(sub) > 0:
        data = sub.reset_index(drop=True)
        break

if data is None:
    print("  (no parquet data found for this episode)")
    sys.exit(0)

n_frames = len(data)
duration = n_frames / fps
print(f"  frames   : {n_frames}")
print(f"  duration : {duration:.2f}s  @{fps:.0f} fps")

# ── Action stats ──────────────────────────────────────────────────
action_cols = [c for c in data.columns if "action" in c.lower()]
for col in action_cols[:1]:
    vals = data[col].tolist()
    first = vals[0]
    if hasattr(first, "__len__"):
        import numpy as np
        arr  = pd.DataFrame(vals).values.astype(float)
        dims = arr.shape[1]
        # Try to label dims sensibly for a 6-DOF + gripper arm
        labels = ["x","y","z","rx","ry","rz","grip"] if dims <= 7 \
                 else [str(i) for i in range(dims)]
        print(f"  action   : {dims} dims")
        for i in range(min(dims, 7)):
            mn, mx, mu = arr[:, i].min(), arr[:, i].max(), arr[:, i].mean()
            bar_range  = mx - mn
            print(f"    [{labels[i]:4s}] min={mn:+.3f}  max={mx:+.3f}  "
                  f"mean={mu:+.3f}  range={bar_range:.3f}")
    else:
        unique = data[col].unique().tolist()
        print(f"  action   : {unique[:6]}")

# ── Task / language annotation ────────────────────────────────────
for col in ["task", "language_instruction", "task_index"]:
    if col in data.columns:
        val = data[col].iloc[0]
        if hasattr(val, "__len__") and not isinstance(val, str):
            val = val[0] if len(val) else val
        print(f"  {col:<9}: {val}")

# ── Observation columns (non-image) ──────────────────────────────
obs_cols = [
    c for c in data.columns
    if c.startswith("observation.") and "image" not in c
]
if obs_cols:
    print(f"  obs cols : {', '.join(obs_cols[:5])}")
PYEOF
}

# ── list_episodes ─────────────────────────────────────────────────
list_episodes() {
    check_dataset
    local total del_count
    total=$(get_total)
    del_count=$(count_deleted)

    echo "Dataset  : ${REPO_ID}"
    echo "Episodes : ${total}  (${del_count} marked for deletion)"
    echo ""
    printf "  %-8s  %-10s  %-12s  %s\n" "Episode" "Seek (s)" "Duration(s)" "Status"
    printf "  %-8s  %-10s  %-12s  %s\n" "-------" "--------" "-----------" "------"

    compute_timings | while read -r ep seek dur; do
        local status=""
        is_deleted "$ep" && status="[DELETE]"
        printf "  %-8d  %-10.2f  %-12.2f  %s\n" "$ep" "$seek" "$dur" "$status"
    done

    if [[ "$del_count" -gt 0 ]]; then
        echo ""
        echo "  Run: $0 --compact  to permanently remove marked episodes."
    fi
}

# ── play_episode ──────────────────────────────────────────────────
# After ffplay exits, prompts user for next action.
# Sets global NAV_ACTION: next | prev | replay | quit
NAV_ACTION="next"

play_episode() {
    local ep_num="$1"
    local total
    total=$(get_total)
    [[ "$ep_num" -lt "$total" ]] || die "Episode ${ep_num} doesn't exist (total: ${total})."

    local seek_s dur_s
    read -r _ seek_s dur_s < <(compute_timings | awk -v ep="$ep_num" '$1==ep')

    # ── Header ──────────────────────────────────────────────────
    local del_tag=""
    is_deleted "$ep_num" && del_tag="  ★ MARKED FOR DELETION"
    printf "\n"
    printf "  ══════════════════════════════════════════════════════\n"
    printf "  Episode  : %d / %d   (col %d)%s\n" \
           "$ep_num" "$((total - 1))" "$COL" "$del_tag"
    printf "  Seek     : %.2fs   Duration: %.2fs\n" "$seek_s" "$dur_s"
    printf "  ──────────────────────────────────────────────────────\n"
    get_episode_info "$ep_num"
    printf "  ══════════════════════════════════════════════════════\n"
    echo   "  ffplay:  space=pause   q=quit   ← / →=seek"
    echo   ""

    # ── Encode + play ────────────────────────────────────────────
    local LABEL="Ep ${ep_num}/${total} | Col ${COL}"
    local TMPFILE
    TMPFILE=$(mktemp /tmp/c4_ep_XXXXXX.mkv)

    printf "  Encoding clip... "
    if [[ -f "$HAND" ]]; then
        ffmpeg -y -hide_banner -loglevel warning \
            -ss "$seek_s" -t "$dur_s" -i "$FRONT" \
            -ss "$seek_s" -t "$dur_s" -i "$HAND" \
            -filter_complex "
                [0:v]scale=640:480,setpts=PTS-STARTPTS,
                     drawtext=text='${LABEL}':x=10:y=10:fontsize=22:fontcolor=white:box=1:boxcolor=black@0.5[v0];
                [1:v]scale=640:480,setpts=PTS-STARTPTS[v1];
                [v0][v1]hstack[out]
            " \
            -map "[out]" -c:v libx264 -crf 18 -preset ultrafast \
            "$TMPFILE"
    else
        ffmpeg -y -hide_banner -loglevel warning \
            -ss "$seek_s" -t "$dur_s" -i "$FRONT" \
            -vf "scale=640:480,setpts=PTS-STARTPTS,
                 drawtext=text='${LABEL}':x=10:y=10:fontsize=22:fontcolor=white:box=1:boxcolor=black@0.5" \
            -c:v libx264 -crf 18 -preset ultrafast \
            "$TMPFILE"
    fi
    echo "done."

    ffplay -hide_banner -loglevel warning \
        -window_title "Col ${COL} | Ep ${ep_num}/${total}" \
        "$TMPFILE"

    rm -f "$TMPFILE"

    # ── Post-play menu ───────────────────────────────────────────
    echo ""
    printf "  ──────────────────────────────────────────────────────\n"
    if is_deleted "$ep_num"; then
        printf "  [n]ext  [p]rev  [u]ndelete  [r]eplay  [q]uit  >  "
    else
        printf "  [n]ext  [p]rev  [d]elete    [r]eplay  [q]uit  >  "
    fi

    local key
    read -rsn1 key || key="q"
    printf "%s\n\n" "$key"

    case "$key" in
        n|"") NAV_ACTION="next"   ;;
        p)    NAV_ACTION="prev"   ;;
        r)    NAV_ACTION="replay" ;;
        d)    mark_deleted "$ep_num"; NAV_ACTION="next"   ;;
        u)    unmark_deleted "$ep_num"; NAV_ACTION="replay" ;;
        q)    echo "  Bye."; exit 0 ;;
        *)    NAV_ACTION="next"   ;;
    esac
}

# ── nav_loop ──────────────────────────────────────────────────────
# Runs the interactive browse-and-navigate loop.
nav_loop() {
    local ep="$1"
    local total
    total=$(get_total)

    # Helper: advance ep by delta (+1 or -1), skipping deleted
    advance_ep() {
        local delta="$1"
        local tries=0
        ep=$(( (ep + delta + total) % total ))
        while is_deleted "$ep"; do
            tries=$((tries + 1))
            if [[ $tries -ge $total ]]; then
                echo "  All episodes are marked for deletion."
                echo "  Run: $0 --compact  or unmark some with [u]."
                exit 0
            fi
            ep=$(( (ep + delta + total) % total ))
        done
    }

    while true; do
        play_episode "$ep"
        case "$NAV_ACTION" in
            next)   advance_ep  1 ;;
            prev)   advance_ep -1 ;;
            replay) : ;;
        esac
    done
}

# ── do_compact ────────────────────────────────────────────────────
# Permanently removes soft-deleted episodes by rewriting the
# dataset's parquet files and re-encoding the video streams.
do_compact() {
    check_dataset
    local del_count total
    del_count=$(count_deleted)
    total=$(get_total)

    if [[ "$del_count" -eq 0 ]]; then
        echo "No episodes marked for deletion. Nothing to compact."
        exit 0
    fi

    echo "Dataset  : ${REPO_ID}"
    echo "Episodes : ${total}  (${del_count} to remove)"
    echo ""
    echo "  Marked for deletion:"
    get_deleted_list | while read -r ep; do
        printf "    Episode %d\n" "$ep"
    done
    echo ""
    printf "  Permanently remove these %d episode(s)? [y/N]  " "$del_count"
    local confirm
    read -r confirm
    [[ "$confirm" == [yY] ]] || { echo "  Cancelled."; exit 0; }
    echo ""

    python3 - "$DATASET_ROOT" "$DELETED_FILE" <<'PYEOF'
import sys, json, pathlib, subprocess, shutil, tempfile, re
import pandas as pd
import numpy as np

root         = pathlib.Path(sys.argv[1])
deleted_file = pathlib.Path(sys.argv[2])

# ── 1. Load deleted list + build keep map ─────────────────────────
deleted = set()
if deleted_file.exists():
    deleted = {int(l.strip()) for l in deleted_file.read_text().splitlines() if l.strip()}

info      = json.loads((root / "meta" / "info.json").read_text())
fps       = float(info.get("fps", 15))
total_eps = int(info.get("total_episodes", 0))
keep      = [i for i in range(total_eps) if i not in deleted]

if not keep:
    print("ERROR: cannot delete all episodes — at least one must remain.", file=sys.stderr)
    sys.exit(1)

old_to_new = {old: new for new, old in enumerate(keep)}
new_to_old = list(keep)  # new_to_old[new_idx] = old_idx

print(f"  Keeping {len(keep)} of {total_eps} episodes.  Removing: {sorted(deleted)}")

# ── 2. Helpers ────────────────────────────────────────────────────
def norm_ep_col(col):
    """Normalise episode_index column to plain Python ints."""
    if col.dtype == object:
        return col.apply(lambda x: int(x[0]) if hasattr(x, "__len__") else int(x))
    return col.astype(int)

# ── 3. Load episode lengths from meta/episodes parquet ───────────
lengths     = {}   # old_ep_idx → frame count
ep_meta_df  = None
ep_meta_path = None

for f in sorted((root / "meta" / "episodes").glob("chunk-*/*.parquet")):
    df = pd.read_parquet(f)
    df["episode_index"] = norm_ep_col(df["episode_index"])
    if "length" in df.columns:
        lengths = dict(zip(df["episode_index"].tolist(), df["length"].astype(int).tolist()))
        ep_meta_df   = df
        ep_meta_path = f
        break

if not lengths:
    # Fallback: count from data parquet
    for f in sorted((root / "data").glob("chunk-*/*.parquet")):
        df = pd.read_parquet(f, columns=["episode_index"])
        df["episode_index"] = norm_ep_col(df["episode_index"])
        counts  = df.groupby("episode_index").size()
        lengths = {int(i): int(c) for i, c in counts.items()}
        break

if not lengths:
    print("ERROR: could not load episode lengths.", file=sys.stderr)
    sys.exit(1)

# ── 4. Compute original seek positions (for video cutting) ────────
front_path      = root / "videos" / "observation.images.front" / "chunk-000" / "file-000.mp4"
lead_in         = 0.0
total_parquet_s = sum(lengths.get(i, 0) for i in range(total_eps)) / fps
try:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1", str(front_path)],
        capture_output=True, text=True, timeout=10)
    vdur   = float(r.stdout.split("=")[1].strip())
    excess = vdur - total_parquet_s
    if excess > 0:
        lead_in = excess / total_eps
except Exception:
    pass

orig_seek = {}   # old_ep_idx → (seek_s, dur_s)
cumulative = 0
for i in range(total_eps):
    length         = lengths.get(i, 0)
    seek_s         = i * lead_in + cumulative / fps
    orig_seek[i]   = (seek_s, length / fps)
    cumulative    += length

# ── 5. New video timestamps for kept episodes ─────────────────────
# After compact the video has no lead_in gaps; episodes are packed
# back-to-back.  We derive timestamps from frame counts so there is
# no floating-point drift vs. the parquet timestamp column.
new_vid_start = {}   # old_ep_idx → absolute start time in new video
new_vid_end   = {}
cum_frames = 0
for old_i in keep:
    l = lengths.get(old_i, 0)
    new_vid_start[old_i]  = cum_frames / fps
    cum_frames           += l
    new_vid_end[old_i]    = cum_frames / fps

# ── 6. Re-encode video streams ────────────────────────────────────
video_keys    = []
video_streams = sorted(p for p in (root / "videos").iterdir() if p.is_dir())

with tempfile.TemporaryDirectory(prefix="c4_compact_") as tmp_str:
    tmp = pathlib.Path(tmp_str)

    for stream_dir in video_streams:
        src = stream_dir / "chunk-000" / "file-000.mp4"
        if not src.exists():
            continue
        vkey = stream_dir.name
        video_keys.append(vkey)
        print(f"\n  Video: {vkey}")

        parts = []
        for ep_i in keep:
            seek_s, dur_s = orig_seek[ep_i]
            slug     = re.sub(r"[^a-z0-9]", "_", vkey.lower())
            out_part = tmp / f"{slug}_ep{ep_i}.mp4"
            print(f"    ep {ep_i}  seek={seek_s:.2f}s  dur={dur_s:.2f}s")
            subprocess.run([
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "warning",
                "-ss", f"{seek_s:.3f}", "-i", str(src),
                "-t",  f"{dur_s:.3f}",
                "-c:v", "libx264", "-crf", "18", "-preset", "fast",
                str(out_part)
            ], check=True)
            parts.append(out_part)

        concat_txt = tmp / f"{slug}_concat.txt"
        concat_txt.write_text("\n".join(f"file '{p}'" for p in parts))
        out_joined = tmp / f"{slug}_joined.mp4"
        print(f"    Joining {len(parts)} clips...")
        subprocess.run([
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "warning",
            "-f", "concat", "-safe", "0", "-i", str(concat_txt),
            "-c", "copy", str(out_joined)
        ], check=True)

        bak = src.with_suffix(".bak.mp4")
        shutil.copy(src, bak)
        shutil.copy(out_joined, src)
        print(f"    {src.name} updated  (backup: {bak.name})")

    # ── 7. Rewrite data parquet ───────────────────────────────────
    print("\n  Rewriting data parquet...")
    for f in sorted((root / "data").glob("chunk-*/*.parquet")):
        df = pd.read_parquet(f)
        df["episode_index"] = norm_ep_col(df["episode_index"])
        df = df[df["episode_index"].isin(keep)].copy()
        df["episode_index"] = df["episode_index"].map(old_to_new)

        sort_cols = ["episode_index"] + (["frame_index"] if "frame_index" in df.columns else [])
        df = df.sort_values(sort_cols).reset_index(drop=True)

        # Recompute global frame index (absolute position across all episodes)
        if "index" in df.columns:
            df["index"] = df.index

        df.to_parquet(f, index=False)
        print(f"    {f.name}: {len(df)} rows")

    # ── 8. Rewrite meta/episodes parquet ─────────────────────────
    print("\n  Rewriting meta/episodes parquet...")
    if ep_meta_df is not None and ep_meta_path is not None:
        df = ep_meta_df[ep_meta_df["episode_index"].isin(keep)].copy()
        df["episode_index"] = df["episode_index"].map(old_to_new)
        df = df.sort_values("episode_index").reset_index(drop=True)

        # Recompute data frame offsets (absolute global frame indices)
        if "data_start_frame" in df.columns and "data_end_frame" in df.columns:
            cum, starts, ends = 0, [], []
            for new_ep in df["episode_index"].tolist():
                old_ep = new_to_old[int(new_ep)]
                l      = lengths.get(old_ep, 0)
                starts.append(cum)
                ends.append(cum + l)
                cum += l
            df["data_start_frame"] = starts
            df["data_end_frame"]   = ends

        # All chunk/file index columns → 0  (single shard after compact)
        for col in df.columns:
            if re.search(r"chunk_index|file_index|/chunk|/file", col):
                df[col] = 0

        # Update per-camera video timestamp columns.
        # Column names vary by lerobot version; match by checking whether
        # a known video-key substring appears alongside "start"/"end" or
        # "from"/"to" in the column name.
        for col in df.columns:
            col_slug = re.sub(r"[^a-z0-9]", "_", col.lower())
            matched_key = next(
                (vk for vk in video_keys
                 if re.sub(r"[^a-z0-9]", "_", vk.lower()) in col_slug),
                None
            )
            if matched_key is None:
                continue
            if re.search(r"start|from|timestamp_from", col_slug):
                df[col] = df["episode_index"].apply(
                    lambda ep: new_vid_start.get(new_to_old[int(ep)], 0.0))
            elif re.search(r"end|to|timestamp_to", col_slug):
                df[col] = df["episode_index"].apply(
                    lambda ep: new_vid_end.get(new_to_old[int(ep)], 0.0))

        df.to_parquet(ep_meta_path, index=False)
        print(f"    {ep_meta_path.name}: {len(df)} rows")
        print(f"    Columns present: {list(df.columns)}")

    # ── 9. Recompute stats.json ───────────────────────────────────
    # Re-read the now-filtered data parquet so stats reflect only the
    # kept episodes.  Image features are skipped (no raw pixels in
    # parquet; their stats are pixel-range constants that don't
    # meaningfully vary with episode selection).
    print("\n  Recomputing stats.json...")
    stats_path     = root / "meta" / "stats.json"
    existing_stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}

    frames = pd.concat(
        [pd.read_parquet(f) for f in sorted((root / "data").glob("chunk-*/*.parquet"))],
        ignore_index=True
    )

    new_stats = dict(existing_stats)  # inherit image stats unchanged

    for feat, _feat_stats in existing_stats.items():
        if re.search(r"image", feat, re.IGNORECASE):
            continue   # keep existing image normalization constants
        if feat not in frames.columns:
            continue

        vals  = frames[feat].tolist()
        first = vals[0]
        arr   = (np.array(vals, dtype=np.float64)
                 if hasattr(first, "__len__")
                 else np.array(vals, dtype=np.float64).reshape(-1, 1))

        std = arr.std(axis=0)
        std = np.where(std < 1e-8, 1e-8, std)   # guard zero-std features

        new_stats[feat] = {
            "mean": arr.mean(axis=0).tolist(),
            "std":  std.tolist(),
            "min":  arr.min(axis=0).tolist(),
            "max":  arr.max(axis=0).tolist(),
        }
        print(f"    {feat}: shape {arr.shape}")

    stats_path.write_text(json.dumps(new_stats, indent=2))
    print("    stats.json written")

    # ── 10. Update info.json ──────────────────────────────────────
    total_kept_frames = sum(lengths.get(i, 0) for i in keep)

    # Count actual video files on disk after rewrite
    total_video_files = sum(
        1 for d in video_streams
        for _ in (d / "chunk-000").glob("*.mp4")
    )

    info["total_episodes"] = len(keep)
    info["total_frames"]   = total_kept_frames
    info["total_videos"]   = total_video_files
    info["total_chunks"]   = 1   # single chunk-000 after compact
    info["splits"]         = {"train": f"0:{len(keep)}"}
    (root / "meta" / "info.json").write_text(json.dumps(info, indent=2))
    print(f"\n  info.json → total_episodes={len(keep)}, "
          f"total_frames={total_kept_frames}, "
          f"total_videos={total_video_files}")

print(f"\n  Compact complete.  {len(deleted)} removed, {len(keep)} remain.")
PYEOF

    # Clear the deleted-episodes file
    > "$DELETED_FILE"
    echo ""
    echo "  Run: $0 --list  to verify."
}

# ── Main ──────────────────────────────────────────────────────────
check_dataset

if $COMPACT_MODE; then
    do_compact

elif $LIST_MODE; then
    list_episodes

elif [[ -z "$EPISODE_ARG" ]]; then
    nav_loop 0

elif [[ "$EPISODE_ARG" == "latest" ]]; then
    ep=$(python3 -c "
import json
d = json.load(open('${DATASET_ROOT}/meta/info.json'))
print(d.get('total_episodes', 1) - 1)
")
    [[ "$ep" -ge 0 ]] || die "No episodes recorded yet."
    nav_loop "$ep"

else
    nav_loop "$EPISODE_ARG"
fi
