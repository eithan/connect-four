# First Task Checklist — SO-101 Pick & Place (v1: fixed pick → column 0)

The end-to-end chain for the first learned task. File references point to the
scaffolding in `robot/`. Goal: prove the record → train → deploy loop on the
narrowest task before scaling to all 7 columns.

## 1. Print the pickup fixture

- [ ] Open `connect_four_piece_edge_nest.scad` in OpenSCAD; press F5 and read the
      echoed footprint / exposed-grip-height / suggested jaw-center Z in the console.
- [ ] (Optional) Enter your gripper's measured `jaw_max_open` and `finger_len`
      so the sanity-check echo is meaningful.
- [ ] F6 to render, then File → Export → Export as STL.
- [ ] Slice and print in PLA (prints "as drawn", no supports). ~under an hour.

## 2. Bench setup (fixed poses are critical)

- [ ] Bolt the nest to the bench through its 4 holes. It must not move between episodes.
- [ ] Orient the nest so its X axis (the jaw-closing direction) is perpendicular
      to the board face — then the lifted disc is already aligned to the slot.
- [ ] Place the Connect Four board so column 0 is comfortably reachable.
- [ ] Seat one yellow disc on edge in the nest.

## 3. Pre-record sanity (do this once)

- [x] Add a udev rule so the arms can't swap on reboot — done: stable symlinks
      `/dev/so101_leader`→ttyACM0, `/dev/so101_follower`→ttyACM1. Scripts use these.
- [ ] Rename camera key `laptop` → `front` in your teleop command to match
      `record_first_dataset.sh` (or revert both to `laptop`). Keys are permanent once recorded.
- [ ] `nvidia-smi` — confirm a CUDA GPU exists for training (set `DEVICE` in `train_act.sh` later).
- [ ] Teleop-teach and note the fixed pickup pose at the nest.

## 4. Throwaway test record (catch problems before 50 episodes)

- [ ] Run `record_first_dataset.sh` for just 2–3 episodes (Escape to stop early).
- [ ] In the Rerun viewer, confirm both cameras hold 30 fps (not silently 15).
- [ ] Confirm the grasp is repeatable and the disc drops cleanly into column 0.
- [ ] Delete the test dataset.

## 5. Record the dataset

- [ ] Set `HF_USER` in `record_first_dataset.sh`.
- [ ] Run `record_first_dataset.sh` → 50 episodes (one pick-and-place each).
      Right Arrow = next, Left Arrow = re-record, Escape = stop.
- [ ] Reload a disc into the nest during each reset window.
- [ ] Schema reference: `dataset_schema.md`.

## 6. Train ACT

- [ ] Set `HF_USER` and `DEVICE` in `train_act.sh` (match the record REPO_ID).
- [ ] Run `train_act.sh`. Checkpoints save every 10k steps under
      `outputs/train/act_c4_pick_place_col0/checkpoints/`.

## 7. Evaluate / deploy on the real arm

- [ ] Uncomment and run the deploy block at the bottom of `train_act.sh`
      (uses `--policy.path` to drive the follower; no teleop).
- [ ] Keep a hand near the power/e-stop on the first rollout.
- [ ] Match camera keys + disc/board layout to recording exactly.
- [ ] Measure task success across ~10 episodes vs. the scripted baseline.

## After v1

- [ ] Widen scope: same fixture, vary target column 0–6, then add the second color
      (switch `single_task` → the templated `task`, bump `repo_id`).
- [ ] Then SmolVLA fine-tune — see `VLA_FINETUNING_PLAN.md` §4.2.
- [ ] (Hardware) Stage 2: add an inclined gravity-feed channel onto the nest so
      discs auto-feed instead of hand-reloading.
