# CARLA Self-Driving Agent

## Setup

```cmd
py -3.12 -m venv venv
venv\Scripts\activate
pip install pygame torch torchvision opencv-python numpy
pip install "<CARLA_ROOT>\PythonAPI\carla\dist\carla-0.9.16-cp312-cp312-win_amd64.whl"
python initial_setup.py
```

## Run

**Evaluate (pre-trained model):**
```cmd
python main.py --start 0 --goal 83 --loop
```

**Train (learn from collisions):**
```cmd
python train.py --start 0 --goal 83 --loop
```

**PID Drive (collision-free, no ML):**
```cmd
python drive.py --start 0 --goal 83
```

**Round trip (start = goal):**
```cmd
python drive.py --start 0 --goal 0
```

**Manual cleanup (if simulator gets stuck):**
```cmd
python force_cleanup.py
```

## Keyboard Controls

| Key | View |
|-----|------|
| 1 | Front Camera (AI) |
| 2 | Third-Person |
| 3 | Top-Down |
| 4 | Side View |
| 5 | Semantic Segmentation |
| 6 | Depth View |
| 7 | Lidar Scan |
| 8 | Radar View |
| 9 | GPS / IMU |
| ESC | Quit |

## Arguments

| Arg | Description |
|-----|-------------|
| `--start N` | Spawn point index |
| `--goal N` | Goal spawn point index |
| `--loop` | New random goal after each episode |
| `--once` | Stop after first episode |

Start CARLA simulator before running any script.
