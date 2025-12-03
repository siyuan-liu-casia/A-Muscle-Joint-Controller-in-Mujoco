# 🌀 Setup Instructions

---

🌐 [Project Page](https://github.com/siyuan-liu-casia/A-Muscle-Joint-Controller-in-Mujoco) 

---

Train a 1-DOF 2-muscle controller to reach random target angles.

<img src="https://picgo-liusiyuan.oss-cn-beijing.aliyuncs.com/picgo-lsy/202512031135279.png" width="20%" alt="">

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
├── output/PPO-Arm12-Time-2025-12-02-17-17-seed-2025     # Training outputs
│   ├── models/                                          # Saved model checkpoints
│   └── logs/PPO_1                                       # TensorBoard logs
├── test/10000000                                         # Test results
├── arm12.py                                             # Custom ARM12 environment
├── train.py                                             # Training script
├── test.py                                              # Testing script
└── utils.py                                             # Utility functions
```
