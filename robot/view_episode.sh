#!/usr/bin/env bash
# =============================================================
#  view_episode.sh
#  Play back a recorded episode from the local LeRobot dataset.
#
#  USAGE:
#    ./view_episode.sh                  # latest episode, col 3
#    ./view_episode.sh 5                # episode 5, col 3
#    ./view_episode.sh --col 0 5        # episode 5, col 0
#    ./view_episode.sh --list           # show all episode timings
#    ./view_episode.sh --latest         # latest episode
#
#  NOTES:
#  - lerobot v3.0 concatenates all episodes in a chunk into a single
#    video file (chunk-000/file-000.mp4).
#  - The `timestamp` column in the data parquet is episode-relative
#    (resets to 0 per episode). Video seek positions are derived from
#    cumulative parquet frame counts + a lead_in correction for the
#    ~1s of reset-phase frames lerobot appends after each episode.
# =============================================================

set -euo pipefail

# ── Args ──────────────────────────────────────────────────────────────────────
COL=3
EPISODE_ARG=""
LIST_MODE=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --col|-c)   shift; COL="$1"; shift ;;
        --list|-l)  LIST_MODE=true; shift ;;
        --latest)   EPISODE_ARG="latest"; shift ;;
        [0-9]*)     EPISODE_ARG="$1"; shift ;;
        *)
            echo "Usage: $0 [--col N] [EPISODE | --list | --latest]"
            exit 1 ;;
    esac
done

# ── Config ────────────────────────────────────────────────────────────────────
HF_USER="eithanz"
REPO_ID="${HF_USER}/connect_four_chute5_pick_col${COL}"
DATASET_ROOT="${HOME}/lerobot_datasets/${REPO_ID}"
VIDEO_ROOT="${DATASET_ROOT}/videos"
FRONT="${VIDEO_ROOT}/observation.images.front/chunk-000/file-000.mp4"
HAND="${VIDEO_ROOT}/observation.images.hand/chunk-000/file-000.mp4"

die() { echo "ERROR: $*" >&2; exit 1; }

check_dataset() {
    [[ -d "$DATASET_ROOT" ]] || die "Dataset not found: $DATASET_ROOT
  Run: ./record_first_dataset.sh ${COL}"
    [[ -f "$FRONT" ]]        || die "No video at: $FRONT
  Record at least one episode first."
}

# ── Core timing computation ───────────────────────────────────────────────────
# Outputs one line per episode:  episode_idx  seek_seconds  duration_seconds
#
# Why not use the parquet's `timestamp` column directly for seeking?
# Because `timestamp` is episode-relative (resets to 0 each episode).
# Seeking to timestamp 0.0 for episode 5 would land at the start of
# the video (episode 0), not episode 5.
#
# Instead we:
#   1. Read per-episode frame counts from meta/episodes parquet.
#   2. Get the actual video duration via ffprobe.
#   3. Distribute the gap (video_dur − parquet_total) as lead_in:
#      lerobot appends ~1s of reset-phase frames after each episode
#      in the video that aren't in the parquet.
#   4. seek[N] = N × lead_in + sum(frame_counts[:N]) / fps
#
compute_timings() {
    python3 - "$DATASET_ROOT" <<'PYEOF'
import sys, json, pathlib, subprocess
import pandas as pd

root = pathlib.Path(sys.argv[1])
fps  = float(json.loads((root / "meta" / "info.json").read_text()).get("fps", 15))

# ── Episode lengths from meta/episodes parquet ────────────────────────────────
lengths = None
for f in sorted((root / "meta" / "episodes").glob("chunk-*/*.parquet")):
    df = pd.read_parquet(f)
    if "length" in df.columns and "episode_index" in df.columns:
        df = df.sort_values("episode_index").reset_index(drop=True)
        lengths = df["length"].tolist()
        break

if lengths is None:
    # Fallback: count frames from the data parquet
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

# ── Lead-in: extra frames appended per episode (reset-phase) ─────────────────
video_path    = root / "videos" / "observation.images.front" / "chunk-000" / "file-000.mp4"
lead_in       = 0.0
total_parquet_s = sum(lengths) / fps
try:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1", str(video_path)],
        capture_output=True, text=True, timeout=10
    )
    video_dur = float(r.stdout.split("=")[1].strip())
    excess    = video_dur - total_parquet_s
    if excess > 0:
        lead_in = excess / len(lengths)
except Exception:
    pass  # no correction; seek will be approximate

# ── Emit one line per episode ─────────────────────────────────────────────────
cumulative = 0.0
for i, length in enumerate(lengths):
    seek_s = i * lead_in + cumulative / fps
    dur_s  = length / fps
    print(f"{i} {seek_s:.3f} {dur_s:.3f}")
    cumulative += length
PYEOF
}

# ── List mode ─────────────────────────────────────────────────────────────────
list_episodes() {
    check_dataset
    local total
    total=$(python3 -c "import json; d=json.load(open('${DATASET_ROOT}/meta/info.json')); print(d.get('total_episodes',0))")

    echo "Dataset:  ${REPO_ID}"
    echo "Episodes: ${total}"
    echo ""
    printf "  %7s  %10s  %12s\n" "Episode" "Seek (s)" "Duration (s)"
    printf "  %7s  %10s  %12s\n" "-------" "--------" "------------"

    compute_timings | while read -r ep seek dur; do
        printf "  %7d  %10.2f  %12.2f\n" "$ep" "$seek" "$dur"
    done
}

# ── Play episode ──────────────────────────────────────────────────────────────
play_episode() {
    local ep_num="$1"
    local total
    total=$(python3 -c "import json; d=json.load(open('${DATASET_ROOT}/meta/info.json')); print(d.get('total_episodes',0))")
    [[ $ep_num -lt $total ]] || die "Episode ${ep_num} doesn't exist (dataset has ${total} episodes, 0-indexed)."

    local seek_s dur_s
    read -r _ seek_s dur_s < <(compute_timings | awk -v ep="$ep_num" '$1==ep')

    echo "Playing  col ${COL}  episode ${ep_num} / $((total-1))"
    printf "  Seek: %.2fs   Duration: %.2fs\n" "$seek_s" "$dur_s"
    echo ""
    echo "Controls: space=pause  q=quit  ←/→=seek"
    echo ""

    local LABEL="Ep ${ep_num} / $((total-1)) | Col ${COL}"

    if [[ -f "$HAND" ]]; then
        ffmpeg -hide_banner -loglevel warning \
            -ss "$seek_s" -t "$dur_s" -i "$FRONT" \
            -ss "$seek_s" -t "$dur_s" -i "$HAND" \
            -filter_complex "
                [0:v]scale=640:480,setpts=PTS-STARTPTS,
                     drawtext=text='${LABEL}':x=10:y=10:fontsize=22:fontcolor=white:box=1:boxcolor=black@0.5[v0];
                [1:v]scale=640:480,setpts=PTS-STARTPTS[v1];
                [v0][v1]hstack[out]
            " \
            -map "[out]" \
            -f nut pipe:1 2>/dev/null \
        | ffplay -hide_banner -loglevel warning \
            -window_title "Col ${COL} | Ep ${ep_num}/${total} — front | hand" \
            -
    else
        ffmpeg -hide_banner -loglevel warning \
            -ss "$seek_s" -t "$dur_s" -i "$FRONT" \
            -vf "scale=640:480,setpts=PTS-STARTPTS,
                 drawtext=text='${LABEL}':x=10:y=10:fontsize=22:fontcolor=white:box=1:boxcolor=black@0.5" \
            -f nut pipe:1 2>/dev/null \
        | ffplay -hide_banner -loglevel warning \
            -window_title "Col ${COL} | Ep ${ep_num}/${total} — front only" \
            -
    fi
}

# ── Main ──────────────────────────────────────────────────────────────────────
check_dataset

if $LIST_MODE; then
    list_episodes

elif [[ -z "$EPISODE_ARG" || "$EPISODE_ARG" == "latest" ]]; then
    ep=$(python3 -c "import json; d=json.load(open('${DATASET_ROOT}/meta/info.json')); print(d.get('total_episodes',1)-1)")
    [[ $ep -ge 0 ]] || die "No episodes recorded yet."
    play_episode "$ep"

else
    play_episode "$EPISODE_ARG"
fi
