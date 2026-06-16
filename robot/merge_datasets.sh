#!/bin/bash

# Target Directories
CACHE_DIR="$HOME/.cache/huggingface/lerobot/eithanz"
MASTER_NAME="c4_col3_merged"
ROLLING_NAME="c4_col3_rolling"
MASTER_JSON="$CACHE_DIR/$MASTER_NAME/meta/info.json"

# Ensure the script stops immediately if any command fails
set -e

# Function to safely parse total_episodes from a given info.json path
get_episode_count() {
    local json_path="$1"
    if [ -f "$json_path" ]; then
        python3 -c "import json; print(json.load(open('$json_path'))['total_episodes'])"
    else
        echo "0"
    fi
}

# Workaround for lerobot bug #2679: merge strips fps from scalar features in
# info.json, which causes feature-mismatch errors on subsequent merges.
patch_scalar_fps() {
    local info_json="$1"
    python3 - "$info_json" <<'EOF'
import json, sys, pathlib
p = pathlib.Path(sys.argv[1])
info = json.loads(p.read_text())
fps = info.get("fps")
if fps is None:
    print("   ⚠️  fps not found in info.json — skipping scalar fps patch")
    sys.exit(0)
scalar_dtypes = {"float32", "int64", "int32", "bool"}
patched = 0
for k, v in info.get("features", {}).items():
    if v.get("dtype") in scalar_dtypes and "fps" not in v:
        v["fps"] = fps
        patched += 1
p.write_text(json.dumps(info, indent=2))
if patched:
    print(f"   ✅ Patched fps={fps} back into {patched} scalar feature(s) (lerobot bug #2679)")
else:
    print("   ℹ️  All scalar features already have fps — no patch needed")
EOF
}

# CREATE ARCHIVE DIRECTORY FIRST
mkdir -p "$CACHE_DIR/archived_chunks"

# --- FIRST RUN INITIALIZATION EDGE CASE ---
if [ ! -d "$CACHE_DIR/$MASTER_NAME" ]; then
    echo "🔍 No master dataset found. Scanning for raw initialization chunks..."
    
    # Get sorted list of raw chunks (excluding internal script tracker names)
    RAW_CHUNKS=($(ls -1 "$CACHE_DIR" | grep -v "$MASTER_NAME" | grep -v "$ROLLING_NAME" | grep -v "archived_chunks" || true))
    
    # Check if we have at least 2 chunks to initialize the merge setup
    if [ ${#RAW_CHUNKS[@]} -lt 2 ]; then
        echo "❌ Error: Could not find '$MASTER_NAME' and there are fewer than 2 raw datasets available to initialize."
        echo "   Please record at least 2 separate batches inside LeLab before running this script."
        exit 1
    fi
    
    CHUNK1="${RAW_CHUNKS[0]}"
    CHUNK2="${RAW_CHUNKS[1]}"
    
    CHUNK1_JSON="$CACHE_DIR/$CHUNK1/meta/info.json"
    CHUNK2_JSON="$CACHE_DIR/$CHUNK2/meta/info.json"
    
    C1_EPS=$(get_episode_count "$CHUNK1_JSON")
    C2_EPS=$(get_episode_count "$CHUNK2_JSON")
    EXPECTED_TOTAL=$((C1_EPS + C2_EPS))
    
    echo "------------------------------------------------------"
    echo "🌟 First-Time Setup Initialization:"
    echo "   • Found Base Chunk 1: $CHUNK1 ($C1_EPS episodes)"
    echo "   • Found Base Chunk 2: $CHUNK2 ($C2_EPS episodes)"
    echo "   • Initializing Master Target: $MASTER_NAME ($EXPECTED_TOTAL episodes)"
    echo "------------------------------------------------------"
    
    echo "🚀 Step 1: Initializing first merge via LeRobot CLI..."
    HF_HUB_OFFLINE=1 lerobot-edit-dataset \
      --repo_id "eithanz/$CHUNK1" \
      --operation.type merge \
      --operation.repo_ids "['eithanz/$CHUNK1', 'eithanz/$CHUNK2']" \
      --new_repo_id "eithanz/$MASTER_NAME"
      
    echo "🧹 Step 2: Archiving raw foundation components..."
    mv "$CACHE_DIR/$CHUNK1" "$CACHE_DIR/archived_chunks/"
    mv "$CACHE_DIR/$CHUNK2" "$CACHE_DIR/archived_chunks/"
    
    echo "🔧 Step 3: Patching scalar fps (lerobot bug #2679 workaround)..."
    patch_scalar_fps "$MASTER_JSON"

    ACTUAL_TOTAL=$(get_episode_count "$MASTER_JSON")
    echo "------------------------------------------------------"
    echo "✅ Success! Master setup initialized."
    echo "🎉 Verified Master Count: $ACTUAL_TOTAL total episodes now in '$MASTER_NAME'."
    echo "------------------------------------------------------"
    exit 0
fi

# --- STANDARD ROLLING WORKFLOW (RUNS ONCE MASTER EXISTS) ---
echo "======================================================"
echo "Available new chunks inside cache directory:"
ls -1 "$CACHE_DIR" | grep -v "$MASTER_NAME" | grep -v "$ROLLING_NAME" | grep -v "archived_chunks" || echo "(No chunks found)"
echo "======================================================"
echo -n "Enter the EXACT name of the new chunk to merge (e.g., c4_col3_part3_...): "
read NEW_CHUNK

CHUNK_JSON="$CACHE_DIR/$NEW_CHUNK/meta/info.json"

# Validate input folder and its metadata exist
if [ ! -d "$CACHE_DIR/$NEW_CHUNK" ]; then
    echo "❌ Error: Folder '$CACHE_DIR/$NEW_CHUNK' does not exist."
    exit 1
fi
if [ ! -f "$CHUNK_JSON" ]; then
    echo "❌ Error: '$CHUNK_JSON' not found. Is this a valid dataset?"
    exit 1
fi

# Fetch and report pre-merge numbers
CURRENT_MASTER_EPS=$(get_episode_count "$MASTER_JSON")
NEW_CHUNK_EPS=$(get_episode_count "$CHUNK_JSON")
EXPECTED_TOTAL=$((CURRENT_MASTER_EPS + NEW_CHUNK_EPS))

echo "------------------------------------------------------"
echo "📊 Pre-Merge Audit:"
echo "   • Current Master ($MASTER_NAME): $CURRENT_MASTER_EPS episodes"
echo "   • Incoming Chunk ($NEW_CHUNK): $NEW_CHUNK_EPS episodes"
echo "   • Target Expected Total: $EXPECTED_TOTAL episodes"
echo "------------------------------------------------------"

echo "🚀 Step 1: Executing rolling data merge via LeRobot CLI..."
HF_HUB_OFFLINE=1 lerobot-edit-dataset \
  --repo_id "eithanz/$MASTER_NAME" \
  --operation.type merge \
  --operation.repo_ids "['eithanz/$MASTER_NAME', 'eithanz/$NEW_CHUNK']" \
  --new_repo_id "eithanz/$ROLLING_NAME"

echo "🧹 Step 2: Safe file cleanup..."
ROLLING_JSON="$CACHE_DIR/$ROLLING_NAME/meta/info.json"

if [ -d "$CACHE_DIR/$ROLLING_NAME" ] && [ -f "$ROLLING_JSON" ]; then
    ACTUAL_TOTAL=$(get_episode_count "$ROLLING_JSON")
    
    if [ "$ACTUAL_TOTAL" -ne "$EXPECTED_TOTAL" ]; then
        echo "⚠️ Warning: Merged total ($ACTUAL_TOTAL) does not match expected total ($EXPECTED_TOTAL)!"
    fi

    echo "   Removing old master directory..."
    rm -rf "$CACHE_DIR/$MASTER_NAME"
    
    echo "   Promoting rolling dataset to master..."
    mv "$CACHE_DIR/$ROLLING_NAME" "$CACHE_DIR/$MASTER_NAME"
    
    echo "   Archiving raw component chunk folder..."
    mv "$CACHE_DIR/$NEW_CHUNK" "$CACHE_DIR/archived_chunks/"

    echo "🔧 Step 3: Patching scalar fps (lerobot bug #2679 workaround)..."
    patch_scalar_fps "$CACHE_DIR/$MASTER_NAME/meta/info.json"

    echo "------------------------------------------------------"
    echo "✅ Success! Master dataset updated."
    echo "🎉 Verified Final Count: $ACTUAL_TOTAL total episodes now in '$MASTER_NAME'."
    echo "------------------------------------------------------"
else
    echo "❌ Error: Merging failed or metadata missing. Parent directories left untouched."
    exit 1
fi

