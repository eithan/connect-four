# Connect Four Robot — Training Plan

Step-by-step operational guide from first recording to a working 7-column policy.
All scripts are in `robot/`. Run recording steps on Ubuntu, training steps on Mac (M4 Max).

---

## Phase 1 — Record Column 0 (Ubuntu)

**Goal:** 20 clean demos of pick-from-chute → drop-into-column-0.

### Step 1: Clean up any test data from earlier sessions

```bash
rm -rf ~/lerobot_datasets/eithanz/connect_four_chute5_pick_col0/
```

### Step 2: Physical setup checklist (before every session)
- Arms plugged in, calibrated, teleop verified
- Chute at its marked position (tape mark on desk)
- Connect Four board at its marked position
- 5 discs loaded in the chute
- Column 0 clear (or at least has room for more pieces)
- Camera locked in position

### Step 3: Start recording

```bash
cd ~/development/connect-four/robot
./record_first_dataset.sh
```

**Episode workflow:**
1. Wait for "Recording episode N" in the log
2. Perform the task: descend → grasp piece → lift → move to target column → drop → return arm to home
3. Press **Right Arrow** in the recording terminal to end the episode early (or wait 30s)
4. The 8-second reset countdown starts — reload a disc if needed, then press **Right Arrow** again to skip the wait
5. Repeat

**When to stop:**
- Wait until you see `Recording episode 21` in the log before pressing Ctrl-C (confirms episode 20 is fully saved — data is written during the reset phase)
- Press **Ctrl-C** in the recording terminal

### Step 4: Verify the recordings

```bash
./view_episode.sh --list          # should show 20 episodes
./view_episode.sh 0               # check first episode
./view_episode.sh 19              # check last episode
```

Use `ffplay` or `mpv` directly if view_episode.sh has issues:
```bash
mpv ~/lerobot_datasets/eithanz/connect_four_chute5_pick_col0/videos/observation.images.front/chunk-000/file-000.mp4
```

**What to look for:** consistent start position, clean grasp, piece lands in column 0, arm returns home. Discard any episode where the arm collides, the piece misses, or the trajectory looks wildly different from the others.

---

## Phase 2 — Train ACT on Column 0 (Mac)

**Goal:** Prove the full pipeline (record → train → eval) works. Learn what a training run looks like.

### Step 5: Transfer dataset to Mac

On Mac:
```bash
rsync -av eithan@ubuntu-host:~/lerobot_datasets/ ~/lerobot_datasets/
```

### Step 6: Apply the local-dataset patch on Mac (one-time)

```bash
python3 - <<'EOF'
import pathlib
f = pathlib.Path("/path/to/lerobot/src/lerobot/datasets/utils.py")
# Update path above to wherever lerobot is installed on Mac
src = f.read_text()
old = '    repo_refs = api.list_repo_refs(repo_id, repo_type="dataset")'
new = '''    try:
        repo_refs = api.list_repo_refs(repo_id, repo_type="dataset")
    except Exception:
        return []  # Hub unreachable — local-only dataset'''
if old in src:
    f.write_text(src.replace(old, new))
    print("Patched.")
elif "local-only dataset" in src:
    print("Already patched.")
else:
    print("Not found — check path.")
EOF
```

### Step 7: Train ACT

`train_act.sh` is already configured. One change: reduce steps since we have 20 demos not 50.

Edit `train_act.sh` temporarily:
- `DEVICE="mps"`
- `--steps=50000` (not 100000 — 20 demos converge faster)
- `--save_freq=5000`

Then run:
```bash
cd ~/development/cursor/connect-four/robot
./train_act.sh
```

**Expected duration:** 1–3 hours on M4 Max with MPS.

**What to watch:** the loss in the log should decrease and stabilize. It won't go to zero — that's fine.

**Checkpoints land in:** `outputs/train/act_c4_pick_place_col0/checkpoints/`

### Step 8: Copy checkpoint to Ubuntu

On Mac:
```bash
rsync -av outputs/train/act_c4_pick_place_col0/ \
  eithan@ubuntu-host:~/development/connect-four/robot/outputs/train/act_c4_pick_place_col0/
```

### Step 9: Eval ACT on Ubuntu

Copy-paste the eval block from `train_act.sh` (the commented-out section at the bottom).
Run it on Ubuntu with the arm connected.

**What to expect:** 20–50% success rate on 10 eval episodes. The arm will do something recognizable — approach the chute, attempt a grasp, move toward the board. Failures are normal and informative.

**This step's real purpose:** see the full loop working. Watch the arm execute a policy it *learned*. That's the milestone.

---

## Phase 3 — SmolVLA on Same Data (Mac)

**Goal:** See the upgrade from ACT to a foundation model on identical data. No new recording.

### Step 10: Train SmolVLA

No dataset changes needed — `train_smolvla.sh` points at the same column 0 dataset.

```bash
cd ~/development/cursor/connect-four/robot
./train_smolvla.sh
```

**First run:** downloads `lerobot/smolvla_base` (~1.5 GB). Needs internet once. Subsequent runs are offline.

**Expected duration:** longer than ACT (larger model), probably 2–5 hours on M4 Max.

**Checkpoints land in:** `outputs/train/smolvla_c4_col0/checkpoints/`

### Step 11: Copy checkpoint and eval on Ubuntu

Same process as ACT eval — use the eval block in `train_smolvla.sh`.

**What to expect:** meaningfully better than ACT on the same 20 demos — SmolVLA starts with pre-trained manipulation knowledge so fewer demos go further.

**The comparison:** side-by-side eval of ACT vs SmolVLA on identical data is the most instructive experiment in this whole project. Write down the success rates.

---

## Phase 4 — Columns 1–6 (Ubuntu → Mac)

**Goal:** Combined 7-column policy. The robot plays Connect Four.

### Step 12: Record 10 demos for each remaining column

```bash
./record_first_dataset.sh 1   # column 1, 10 episodes
./record_first_dataset.sh 2   # column 2, 10 episodes
# ... through column 6
./record_first_dataset.sh 6
```

Each run saves into its own dataset (`connect_four_chute5_pick_col1`, etc.) and resumes automatically if interrupted.

**Total additional recording:** 60 demos across 6 sessions (can do multiple columns per session).

**Columns 0 and 6 (edges) and column 3 (center) will have the most distinctive trajectories** — record these cleanly. Middle columns are easier.

### Step 13: Transfer all datasets to Mac

```bash
rsync -av eithan@ubuntu-host:~/lerobot_datasets/ ~/lerobot_datasets/
```

### Step 14: Merge datasets and train combined SmolVLA

Merge all 7 column datasets into one:
```bash
# lerobot merge command (check lerobot-merge-datasets --help for exact flags)
lerobot-merge-datasets \
  --repo_ids \
    eithanz/connect_four_chute5_pick_col0 \
    eithanz/connect_four_chute5_pick_col1 \
    eithanz/connect_four_chute5_pick_col2 \
    eithanz/connect_four_chute5_pick_col3 \
    eithanz/connect_four_chute5_pick_col4 \
    eithanz/connect_four_chute5_pick_col5 \
    eithanz/connect_four_chute5_pick_col6 \
  --output_repo_id eithanz/connect_four_all_columns \
  --root ~/lerobot_datasets
```

Then train:
```bash
# Edit train_smolvla.sh:
# REPO_ID="${HF_USER}/connect_four_all_columns"
# JOB_NAME="smolvla_c4_all_columns"
./train_smolvla.sh
```

### Step 15: Eval combined policy on Ubuntu

Run 10 eval episodes with the game loop driving column selection (AlphaZero picks the column, policy executes the move).

**What to expect:** 50–75% per-move success. With the vision-based retry logic (Phase 4 of the main project plan), effective game completion climbs significantly.

---

## Summary

| Phase | Where | Time est. | Milestone |
|---|---|---|---|
| Record col 0 (20 demos) | Ubuntu | 1–2 sessions | Dataset exists |
| Train ACT | Mac | 1–3 hrs | Full pipeline proven |
| Eval ACT | Ubuntu | 20 min | Policy running on real arm |
| Train SmolVLA | Mac | 2–5 hrs | Foundation model working |
| Eval SmolVLA | Ubuntu | 20 min | Visible improvement over ACT |
| Record cols 1–6 (10 each) | Ubuntu | 2–3 sessions | All columns covered |
| Merge + train combined | Mac | 3–6 hrs | 7-column policy |
| Eval combined | Ubuntu | 30 min | Robot plays Connect Four |

---

## Notes

- **Discard sloppy demos.** Consistency of demonstrations matters more than count. A clean 15-demo dataset beats a messy 20-demo dataset.
- **Don't change the physical setup between recording and eval.** Camera position, chute location, and board position must be identical. Mark everything with tape.
- **The lerobot patch** (`get_repo_versions` try/except in `utils.py`) must be applied on both Ubuntu and Mac before training.
- **Ctrl-C safety rule:** always wait for `Recording episode N` in the log before stopping — episode data is written during the reset phase.
