# Connect Four — LeRobot Dataset Schema

**Status:** v1 (first single-task dataset). Last updated 2026-05-22.

This file pins down *what* we record before we record anything, so the dataset
is consistent across episodes and VLA-ready (SmolVLA / π₀) from episode one.
It is the concrete instance of the annotation schema sketched in
[`VLA_FINETUNING_PLAN.md`](./VLA_FINETUNING_PLAN.md) §4.1.

---

## 1. Task scope (v1)

The narrowest useful motion: **fixed pick → fixed place.**

- **Pick:** one disc, standing on edge, from the edge-nest fixture
  (`connect_four_piece_edge_nest.scad`) at a single fixed bench pose.
- **Place:** drop it into **column 0** of the Connect Four board.
- **No column selection, no color selection yet.** Both are pinned to
  constants so the only thing the policy has to learn is the pick-lift-carry-
  drop trajectory. Column/color generalization comes in v2.

Why this scope: the goal of the first dataset is to prove the whole
record → train (ACT) → deploy loop on the easiest possible task before adding
the 7-column problem. See [PROJECT_PLAN / SESSION_STATUS] "smallest first task".

---

## 2. Language annotation

Every episode in v1 carries the **same** instruction string (recorded via
`--dataset.single_task`):

```
Pick a yellow piece from the tray and drop it into column 0
```

This is a frozen special case of the v2 template, which we will adopt once the
dataset grows to multiple colors/columns:

```
pick a {color} piece and place it in column {col}
   color ∈ {red, yellow}
   col   ∈ {0 .. 6}
```

Keep the v1 string phrased in the same shape as the v2 template (verb + color +
"column" + integer) so a later mixed dataset reads consistently.

---

## 3. Episode definition

- **One pick-and-place = one episode.** Start: arm at home, disc seated in the
  nest. End: disc released into column 0 and arm returning toward home.
- **Target count (v1):** 50 episodes. Enough for an ACT baseline; cheap to top
  up later with `--resume=true`.
- **Reset between episodes:** during the reset window, hand-reload one disc into
  the nest (Stage-1 fixture is hand-fed) and confirm the board column is clear.

---

## 4. Observation / action shape (don't hand-edit; recorded automatically)

| Key | Source | Notes |
|---|---|---|
| `observation.state` | follower joint positions (6) | SO-101: 5 arm joints + gripper |
| `action` | leader joint positions (6) | what you commanded via teleop |
| `observation.images.front` | scene camera (`/dev/video0`) | was named `laptop` — see §6 |
| `observation.images.hand`  | wrist camera (`/dev/video2`) | wrist-mounted |

`fps = 30` for both cameras and the control loop. Confirm the cameras actually
deliver 30 fps in the Rerun viewer during a throwaway test record — they
silently drop to 15 under USB-bandwidth pressure, and you don't want to find out
after 50 episodes.

---

## 5. Per-dataset metadata (record once, keep in the dataset card)

Not all of these are first-class LeRobot fields; capture them in the dataset
card / README on the Hub so the conditions are reproducible:

- Disc geometry: 32 mm dia × 8.5 mm thick (measured 2026-05-01).
- Pickup fixture: `connect_four_piece_edge_nest.scad`, bolted at a fixed pose.
- Color (v1): yellow. Column (v1): 0.
- Camera mounting: front = fixed scene cam; hand = wrist cam. Note exact
  positions / focal settings so they can be reproduced.
- Lighting condition (e.g. "overhead office LED, blinds closed").
- Operator, date, LeRobot version, calibration file ids.

---

## 6. Naming + format guardrails

- **Camera keys are permanent.** They become `observation.images.<key>` and the
  policy keys off them — you cannot rename without reprocessing the dataset.
  Recommendation: rename `laptop` → `front` in BOTH the teleop and record
  commands *before* the first real episode. (The record script already uses
  `front`; make your teleop command match, or revert both to `laptop`.)
- **LeRobot format only.** `lerobot-record` writes LeRobot v2 format by default —
  this is the universal interchange format every VLA recipe (ACT, SmolVLA, π₀,
  π₀.₅) consumes. Do not convert to anything else.
- **repo_id convention:** `<hf-username>/connect_four_pick_place_col0`. Bump the
  suffix when the task scope changes (e.g. `..._allcols`, `..._twocolor`).

---

## 7. Next after v1

1. Train ACT on this dataset; evaluate task success vs. a scripted baseline.
2. If the loop is solid, widen scope: same fixture, vary the target column
   (0–6), then add the second color. Update the `single_task` → templated
   `task` per episode, and bump `repo_id`.
3. Then SmolVLA fine-tune on the widened dataset (see `VLA_FINETUNING_PLAN.md` §4.2).
