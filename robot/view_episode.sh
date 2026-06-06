#!/usr/bin/env bash
# =============================================================
#  view_episode.sh — GLOBAL SCRUB + EPISODE OVERLAY VERSION
#
#  FEATURES:
#  - Continuous timeline across ALL episodes
#  - No drift (timestamp-based)
#  - Episode index overlay on video
# =============================================================

set -euo pipefail

# ── Args ─────────────────────────────────────────────────────
COL=3
EPISODE_ARG=""
LIST_MODE=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --col|-c)  shift; COL="$1"; shift ;;
        --list|-l) LIST_MODE=true; shift ;;
        --latest)  EPISODE_ARG="--latest"; shift ;;
        [0-9]*)    EPISODE_ARG="$1"; shift ;;
        *)
            echo "Usage: $0 [--col N] [EPISODE | --list | --latest]"
            exit 1
            ;;
    esac
done

# ── Config ───────────────────────────────────────────────────
HF_USER="eithanz"
REPO_ID="${HF_USER}/connect_four_chute5_pick_col${COL}"
DATASET_ROOT="${HOME}/lerobot_datasets/${REPO_ID}"

VIDEO_ROOT="${DATASET_ROOT}/videos"
FRONT="${VIDEO_ROOT}/observation.images.front/chunk-000/file-000.mp4"
HAND="${VIDEO_ROOT}/observation.images.hand/chunk-000/file-000.mp4"

die() { echo "ERROR: $*" >&2; exit 1; }

check_dataset() {
    [[ -d "${DATASET_ROOT}" ]] || die "Dataset not found: ${DATASET_ROOT}"
}

count_episodes() {
    python3 -c "
import json
d=json.load(open('${DATASET_ROOT}/meta/info.json'))
print(d.get('total_episodes', 0))
"
}

# ── BUILD GLOBAL TIMELINE ────────────────────────────────────
# Returns:
# ep start end duration
build_timeline() {
python3 - "$DATASET_ROOT" <<'PY'
import sys
import pandas as pd
import pathlib

root = pathlib.Path(sys.argv[1])
df = pd.read_parquet(root / "data/chunk-000/file-000.parquet")

episodes = []
for ep in sorted(df["episode_index"].unique()):
    ep_df = df[df["episode_index"] == ep]

    start = float(ep_df["timestamp"].iloc[0])
    end   = float(ep_df["timestamp"].iloc[-1])

    episodes.append((ep, start, end, end - start))

# normalize to GLOBAL timeline (fix offsets)
global_start = episodes[0][1]
episodes = [(e, s-global_start, t-global_start, d) for (e,s,t,d) in episodes]

for e,s,t,d in episodes:
    print(e, s, t, d)
PY
}

# ── GET EPISODE RANGE ────────────────────────────────────────
get_episode_range() {
    local ep=$1

    python3 - "$DATASET_ROOT" "$ep" <<'PY'
import pandas as pd
import pathlib

root = pathlib.Path(sys.argv[1])
ep = int(sys.argv[2])

df = pd.read_parquet(root / "data/chunk-000/file-000.parquet")
ep_df = df[df["episode_index"] == ep]

start = float(ep_df["timestamp"].iloc[0])
end   = float(ep_df["timestamp"].iloc[-1])

print(start, end)
PY
}

# ── LIST (GLOBAL TIMELINE SCRUB VIEW) ───────────────────────
list_episodes() {
    check_dataset

    echo "=== GLOBAL TIMELINE (SCRUB BAR VIEW) ==="
    echo ""

    build_timeline | awk '
    {
        printf "Episode %-2s | start: %8.3fs | end: %8.3fs | dur: %6.3fs\n",
               $1, $2, $3, $4
    }'
}

# ── PLAY WITH OVERLAY ───────────────────────────────────────
play_episode() {
    local ep=$1

    read start end < <(get_episode_range "$ep")

    dur=$(python3 -c "print($end - $start)")

    echo "Playing Episode $ep"
    echo "Start: $start  End: $end  Duration: $dur"

    # overlay text
    TEXT="Episode ${ep} | Col ${COL}"

    ffmpeg -hide_banner -loglevel warning \
        -ss "$start" -t "$dur" -i "$FRONT" \
        -ss "$start" -t "$dur" -i "$HAND" \
        -filter_complex "
        [0:v]scale=640:480,setpts=PTS-STARTPTS,
             drawtext=text='${TEXT}':x=10:y=10:fontsize=24:fontcolor=white:box=1:boxcolor=black@0.5[v0];
        [1:v]scale=640:480,setpts=PTS-STARTPTS[v1];
        [v0][v1]hstack[out]
        " \
        -map "[out]" \
        -f nut pipe:1 2>/dev/null \
    | ffplay -hide_banner -loglevel warning \
        -window_title "Episode ${ep} | Col ${COL}" \
        -
}

# ── LATEST ───────────────────────────────────────────────────
get_latest_episode() {
    python3 -c "
import json
d=json.load(open('${DATASET_ROOT}/meta/info.json'))
print(d.get('total_episodes',0)-1)
"
}

# ── MAIN ─────────────────────────────────────────────────────
check_dataset

if $LIST_MODE; then
    list_episodes

elif [[ -z "$EPISODE_ARG" || "$EPISODE_ARG" == "--latest" ]]; then
    ep=$(get_latest_episode)
    play_episode "$ep"

else
    play_episode "$EPISODE_ARG"
fi