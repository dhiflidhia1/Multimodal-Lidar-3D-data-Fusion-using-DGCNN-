# Quick Start Guide

Get up and running with DGCNN-MidFusion in 5 minutes!

## 🚀 Installation (3 minutes)

### Using pip (Recommended)

```bash
# 1. Clone repository
git clone https://github.com/yourusername/DGCNN-MidFusion-Model.git
cd DGCNN-MidFusion-Model

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
```

### Using Conda

```bash
# 1. Clone repository
git clone https://github.com/yourusername/DGCNN-MidFusion-Model.git
cd DGCNN-MidFusion-Model

# 2. Create conda environment
conda env create -f environment.yml
conda activate dgcnn

# 3. Verify
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
```

## 📂 Prepare Data (1 minute)

Place your data file in the `data/` directory:

```bash
data/
└── fractal_data_v2_filtered.npy  # Required
```

Expected format:
- Shape: `(N_samples, N_points, N_features)`
- Features: `[X, Y, Z, R, G, B, IR, ReturnNum, NumReturns, Label]`

## ▶️ Run Training (1 minute)

### Default Configuration

```bash
python scripts/train.py
```

### Custom Parameters

```bash
python scripts/train.py \
    --epochs 100 \
    --batch-size 32 \
    --lr 1e-3 \
    --k 20 \
    --dropout 0.5
```

### Check GPU

```bash
python -c "import torch; print('GPU Available:', torch.cuda.is_available())"
```

## 📊 View Results

After training, check outputs:

```bash
outputs/
├── checkpoints/best_model.pth
├── plots/
│   ├── confusion_matrix.png
│   ├── training_curves.png
│   ├── per_class_iou.png
│   └── per_class_f1.png
└── results/
    ├── test_metrics.json
    └── training_log.csv
```

## 🧪 Evaluate Pre-trained Model

```bash
python scripts/evaluate.py --model-path outputs/checkpoints/best_model.pth
```

## 🐛 Troubleshooting

### Out of Memory (OOM)

```python
# In src/config.py
BATCH_SIZE = 16  # Reduce from 32
```

### Slow Training

```bash
# Verify GPU is being used
nvidia-smi

# Check CPU usage
top  # Linux/macOS
tasklist  # Windows
```

### Data Not Found

```bash
# Verify data file exists
ls -la data/fractal_data_v2_filtered.npy  # Linux/macOS
dir data\                                 # Windows
```

## 📖 Key Files

| File | Purpose |
|------|---------|
| `scripts/train.py` | Main training script |
| `scripts/evaluate.py` | Evaluation script |
| `src/config.py` | Configuration parameters |
| `src/data.py` | Data loading |
| `src/model.py` | Model architecture |
| `README.md` | Full documentation |

## ⚙️ Configuration Quick Reference

### Hyperparameters (`src/config.py`)

```python
# Model
K = 20                    # KNN neighbors
DROPOUT = 0.5            # Dropout rate
N_POINTS = 2048          # Points per cloud

# Training
BATCH_SIZE = 32
EPOCHS = 100
LR = 1e-3               # Learning rate
WEIGHT_DECAY = 1e-4

# Hardware
USE_AMP = True          # Mixed precision
NUM_WORKERS = 4         # Data loading workers
```

## 🔧 Git Commands Quick Reference

### Initial Setup

```bash
# Initialize and first push
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/user/repo.git
git push -u origin main
```

### Development Workflow

```bash
# Create feature branch
git checkout -b feature/my-feature

# Make changes
git add .
git commit -m "feat: add new feature"

# Push and create PR
git push -u origin feature/my-feature
```

### Useful Commands

```bash
# Check status
git status

# View history
git log --oneline -10

# Undo last commit (keep changes)
git reset --soft HEAD~1

# Sync with upstream
git fetch upstream
git rebase upstream/main

# Clean up branches
git branch -d feature/done
```

## 📝 Next Steps

1. ✅ **Setup complete!** Run `python scripts/train.py`
2. 📖 Read [README.md](README.md) for detailed documentation
3. 🔧 Configure hyperparameters in [src/config.py](src/config.py)
4. 📤 Push to GitHub following [GITHUB_SETUP.md](GITHUB_SETUP.md)
5. 👥 Contribute following [CONTRIBUTING.md](CONTRIBUTING.md)

## 💬 Need Help?

- 📖 Check [README.md](README.md)
- 🐛 Open an issue on GitHub
- 💬 Discuss in GitHub Discussions
- 📧 Contact: your.email@example.com

---

**Happy training! 🚀**
