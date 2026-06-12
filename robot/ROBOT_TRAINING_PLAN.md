# Connect Four Robot Arm Training Plan

## Overview

This plan covers training an ACT (Action Chunking Transformer) model using LeRobot on a Hiwonder SO-101 arm to play connect four. The robot handles all AI moves; the human handles their own moves physically. The vision and AI layer (column selection, board state) is already complete.

**Hardware:** Hiwonder SO-101 leader + follower arms, M4 Max MacBook Pro  
**Framework:** HuggingFace LeRobot  
**Policy:** ACT (one policy per column, 7 total)  
**Pickup:** Single piece placed flat on a fixed spot on the desk  

---

## Demo Log

Keep this updated as you record. Add a row for each session.

| Date | Column | # Demos | Notes |
|------|--------|---------|-------|
|      |        |         |       |

---

## Status Tracker

- [ ] Phase 1 — Setup on M4 Max
- [ ] Phase 1b — Lock down physical setup
- [ ] Phase 2 — Record column 3 demos (20)
- [ ] Phase 3 — Train and test column 3 policy
- [ ] Phase 4 — Connect vision AI for column 3
- [ ] Phase 5 — Fix bugs, freeze physical setup
- [ ] Phase 6 — Record all remaining columns (20 each)
- [ ] Phase 7 — Train all column policies
- [ ] Phase 7b (maybe) — 3D print funnel for outer columns
- [ ] Phase 8 — Dry run all columns
- [ ] Phase 9 — Full game integration

---

## Phase 1 — Setup on M4 Max

Install dependencies and verify the arms work as well as they did on Ubuntu.

```bash
# Install homebrew dependencies
brew install python@3.10 ffmpeg

# Clone and install LeRobot
git clone https://github.com/huggingface/lerobot.git
cd lerobot
conda create -y -n lerobot python=3.10
conda activate lerobot
conda install ffmpeg -c conda-forge
pip install -e ".[feetech]"

# Find your arm ports — run this twice, once per arm
# Plug in one arm, run the script, unplug when prompted, note the port
lerobot-find-port
```

Note the two port names (e.g. `/dev/tty.usbmodem5A460820701`) and save them:

```bash
# robot/config/ports.env
LEADER_PORT=/dev/tty.usbmodemXXXXXXX
FOLLOWER_PORT=/dev/tty.usbmodemYYYYYYY
```

Verify teleoperation works — leader arm should drive follower smoothly:

```bash
source robot/config/ports.env

python -m lerobot.teleoperate \
  --robot.type=so101_follower \
  --robot.port=$FOLLOWER_PORT \
  --robot.id=follower \
  --teleop.type=so101_leader \
  --teleop.port=$LEADER_PORT \
  --teleop.id=leader \
  --display_data=true
```

**Done when:** follower mirrors leader with no lag or missed commands, matching Ubuntu behavior.

---

## Phase 1b — Lock Down Physical Setup

**Do this before recording a single demo. Do not change it after Phase 5.**

- [ ] Mark arm base position with tape
- [ ] Mark board position with tape
- [ ] Mark camera position and angle with tape
- [ ] Mark pickup spot with a small circle of tape
- [ ] Take a photo of the full setup and save it to `robot/setup_reference.jpg`

The pickup spot should be positioned so the arm has clean access to it and all 7 columns without awkward joint configurations at the extremes.

---

## Phase 2 — Record Column 3 Demos (20)

Column 3 first because it is closest to your existing 10-demo baseline and is the easiest trajectory.

```bash
source robot/config/ports.env

python -m lerobot.record \
  --robot.type=so101_follower \
  --robot.port=$FOLLOWER_PORT \
  --robot.id=follower \
  --teleop.type=so101_leader \
  --teleop.port=$LEADER_PORT \
  --teleop.id=leader \
  --dataset.repo_id=local/connectfour_col3 \
  --dataset.num_episodes=20 \
  --dataset.single_task="Pick up yellow piece and place in column 3" \
  --robot.cameras="{ \
    top: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, \
    wrist: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}}" \
  --display_data=true
```

**Recording tips:**
- Place the piece on the marked pickup spot before each episode
- Move the leader arm smoothly and deliberately — ACT learns your timing
- Keep speed consistent across all 20 demos
- If a demo goes wrong, discard it immediately (do not include bad examples)
- Pause between episodes to reset

**Log each session in the Demo Log table above.**

---

## Phase 3 — Train and Test Column 3 Policy

```bash
python -m lerobot.train \
  --dataset.repo_id=local/connectfour_col3 \
  --policy.type=act \
  --output_dir=robot/outputs/act_col3 \
  --training.num_epochs=100 \
  --device=mps
```

Monitor training loss — it should plateau before you stop. If not converged at 100 epochs, increase to 200.

Test the policy:

```bash
source robot/config/ports.env

python -m lerobot.eval \
  --robot.type=so101_follower \
  --robot.port=$FOLLOWER_PORT \
  --robot.id=follower \
  --policy.path=robot/outputs/act_col3/checkpoints/last \
  --robot.cameras="{ \
    top: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, \
    wrist: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}}"
```

**Common issues:**
| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| Jerky motion | Inconsistent demo speeds | Re-record demos |
| Wrong trajectory | Camera indices swapped | Check and correct camera config |
| Overshooting | Overfitting | Reduce epochs |
| Slow inference | Not using MPS | Verify `--device=mps` |
| Misses the column | Pickup spot or board shifted | Check physical setup against reference photo |

**Done when:** policy reliably picks up the piece and drops it into column 3 at least 8/10 attempts.

---

## Phase 4 — Connect Vision AI for Column 3

Build the robot server before wiring up the vision AI. This gives you a clean interface that works for all columns later.

```python
# robot/robot_server.py
from fastapi import FastAPI
import asyncio

app = FastAPI()
busy = False

policies = {
    3: "robot/outputs/act_col3/checkpoints/last",
    # Add columns here as they are trained
}

def run_policy(policy_path: str):
    # Replace with your actual LeRobot inference call
    import subprocess
    subprocess.run([
        "python", "-m", "lerobot.eval",
        "--policy.path", policy_path,
        # add robot/camera args here
    ])

@app.get("/status")
def status():
    return {"busy": busy}

@app.post("/place/{column}")
async def place(column: int):
    global busy
    if busy:
        return {"status": "busy"}
    if column not in policies:
        return {"status": "error", "message": f"Column {column} not trained yet"}
    busy = True
    try:
        run_policy(policies[column])
        return {"status": "ok"}
    finally:
        busy = False
```

```bash
# Start the server
pip install fastapi uvicorn httpx
uvicorn robot.robot_server:app --port 8000
```

Add a robot client to your vision AI:

```python
# robot/robot_client.py
import httpx
import asyncio

ROBOT_URL = "http://localhost:8000"

async def place_piece(column: int) -> bool:
    async with httpx.AsyncClient(timeout=30) as client:
        # Wait if robot is busy
        while True:
            status = await client.get(f"{ROBOT_URL}/status")
            if not status.json()["busy"]:
                break
            await asyncio.sleep(0.5)
        response = await client.post(f"{ROBOT_URL}/place/{column}")
        return response.json()["status"] == "ok"
```

In your game loop:

```python
if is_ai_turn:
    column = ai.get_move(board)
    success = await place_piece(column)
else:
    notify_human_to_play()
    wait_for_vision_to_detect_human_move()
```

**Done when:** vision AI correctly triggers the column 3 policy and the full loop (AI decides → robot picks up → robot drops in column 3) runs end to end.

---

## Phase 5 — Fix Bugs, Freeze Physical Setup

Iterate until the column 3 full loop is reliable. Once it is:

- [ ] Do not move the arm base, board, camera, or pickup spot again
- [ ] Update `robot/setup_reference.jpg` if anything was intentionally adjusted
- [ ] Note any config changes in the Demo Log

---

## Phase 6 — Record All Remaining Columns (20 each)

Record in this order (center outward, alternating sides) so you can catch drift early:

**Order: col4 → col2 → col5 → col6 → col1 → col7**

```bash
source robot/config/ports.env

# Replace COL_NUMBER and col# with the target column each time
python -m lerobot.record \
  --robot.type=so101_follower \
  --robot.port=$FOLLOWER_PORT \
  --robot.id=follower \
  --teleop.type=so101_leader \
  --teleop.port=$LEADER_PORT \
  --teleop.id=leader \
  --dataset.repo_id=local/connectfour_colCOL_NUMBER \
  --dataset.num_episodes=20 \
  --dataset.single_task="Pick up yellow piece and place in column COL_NUMBER" \
  --robot.cameras="{ \
    top: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, \
    wrist: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}}" \
  --display_data=true
```

**Log each session in the Demo Log table.**

---

## Phase 7b (Maybe) — 3D Print Funnel for Outer Columns

The arm may not have sufficient dexterity to drop pieces precisely into columns 1 and 7. If testing reveals this problem, design and print a funnel to guide pieces in.

- Design a funnel with a wide entry (~3cm diameter) tapering to the column width
- Mount it on the board at columns 1 and 7 only
- Re-record demos for those columns with the funnel in place
- The funnel can stay on the board permanently during play

**Revisit this after Phase 8 testing. Skip if columns 1 and 7 work without it.**

---

## Phase 7 — Train All Column Policies

```bash
for col in 1 2 3 4 5 6 7; do
  python -m lerobot.train \
    --dataset.repo_id=local/connectfour_col${col} \
    --policy.type=act \
    --output_dir=robot/outputs/act_col${col} \
    --training.num_epochs=100 \
    --device=mps
done
```

Update `robot/robot_server.py` policies dict:

```python
policies = {
    1: "robot/outputs/act_col1/checkpoints/last",
    2: "robot/outputs/act_col2/checkpoints/last",
    3: "robot/outputs/act_col3/checkpoints/last",
    4: "robot/outputs/act_col4/checkpoints/last",
    5: "robot/outputs/act_col5/checkpoints/last",
    6: "robot/outputs/act_col6/checkpoints/last",
    7: "robot/outputs/act_col7/checkpoints/last",
}
```

---

## Phase 8 — Dry Run All Columns

Before connecting the vision AI, manually test every column at least 3 times each with no pieces in the board:

```bash
# Test each column manually
curl -X POST http://localhost:8000/place/1
# Reset piece, then...
curl -X POST http://localhost:8000/place/7
# etc. through all 7
```

**Done when:** all 7 columns succeed at least 8/10 attempts consistently.

If columns 1 or 7 are unreliable, proceed to Phase 7b (funnel) before continuing.

---

## Phase 9 — Full Game Integration

The robot client from Phase 4 already handles all columns — just ensure the policies dict is fully populated and the server is running.

Run a full game end to end, AI vs human:
- AI moves: robot picks up piece from marked spot and drops in the chosen column
- Human moves: human places their piece physically; vision detects it
- AI never touches human pieces; human never touches AI pieces

**Done when:** a complete game runs from first move to win condition without manual intervention.

---

## Directory Structure

```
robot/
├── ROBOT_TRAINING_PLAN.md     # This file
├── config/
│   └── ports.env              # Leader and follower USB port names
├── setup_reference.jpg        # Photo of locked physical setup
├── robot_server.py            # FastAPI server wrapping LeRobot inference
├── robot_client.py            # Client used by vision AI to send commands
├── datasets/                  # Symlink or notes pointing to LeRobot dataset cache
└── outputs/
    ├── act_col1/
    ├── act_col2/
    ├── act_col3/
    ├── act_col4/
    ├── act_col5/
    ├── act_col6/
    └── act_col7/
```
