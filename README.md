# 🌀 Setup Instructions

---

🌐 [Project Page](https://github.com/siyuan-liu-casia/A-Muscle-Joint-Controller-in-Mujoco) 

---

Train a 1-DOF 2-muscle controller to reach random target angles.

<video controls  width="300">
  <source src="test_video.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

---

## 🛠️ Setup

### 🔹 Step 1: Create and Activate Conda Environment

```bash
conda create -n arm python=3.9
conda activate arm
```

### 🔹 Step 2:  Install Dependencies

```bash
pip install stable-baselines3 tensorboard scikit-video mujoco
```

### 📕 Training

To train the agent, run:

```bash
python train.py 
```

## 🎯 Testing

To test a trained model, run the following command. The script will automatically save performance plots and videos.

```bash
python test.python
```

## 🌳 Project Structure

```bash
project/
├── assets/  
│   └── arm12.xml                                        # MuJoCo model XML files
├── output/
│   ├── PPO-Arm12-Time-2025-12-03-11-42-seed-2025       # Training outputs alpha = 0
│   └── PPO-Arm12-Time-2025-12-03-12-54-seed-2025       # Training outputs alpha = 0.5
│       └── models/                                      # Saved model checkpoints
│       └── logs/                                        # TensorBoard logs
├── test/
│   ├── alpha_0_10000000                                 # Testing results alpha = 0
│   └── alpha_0.5_10000000                               # Testing results alpha = 0.5
├── arm12.py                                             # Custom ARM12 environment
├── train.py                                             # Training script
├── test.py                                              # Testing script
├── video.py                                             # Continuous Video generation
└── utils.py                                             # Utility functions
```
