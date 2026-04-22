# Contributing to DGCNN-MidFusion-Model

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing.

## 🎯 Code of Conduct

- Be respectful and inclusive
- Welcome diverse perspectives
- Focus on constructive feedback
- Help others learn

## 📋 How to Contribute

### Reporting Bugs

1. **Check existing issues** before reporting
2. **Use the bug template**:
   ```markdown
   ## Description
   Brief description of the bug
   
   ## Steps to Reproduce
   1. Load model with config X
   2. Run training for N epochs
   3. Observe error
   
   ## Expected Behavior
   What should happen
   
   ## Actual Behavior
   What actually happened
   
   ## Environment
   - OS: Windows/Linux/macOS
   - Python: 3.10
   - PyTorch: 2.0
   - GPU: (if applicable)
   
   ## Error Log
   ```
   paste full error traceback
   ```
   ```

3. **Use clear, descriptive titles**
4. **Label appropriately** (bug, urgent, etc.)

### Suggesting Enhancements

1. **Use the feature request template**:
   ```markdown
   ## Feature Description
   What feature would you like?
   
   ## Motivation
   Why is this important?
   
   ## Implementation Ideas
   How might this be implemented?
   
   ## Additional Context
   References, examples, etc.
   ```

2. **Search for existing feature requests** first

### Code Contributions

#### Step 1: Fork and Clone

```bash
# Fork on GitHub
# Clone your fork
git clone https://github.com/YOUR_USERNAME/DGCNN-MidFusion-Model.git
cd DGCNN-MidFusion-Model

# Add upstream remote
git remote add upstream https://github.com/ORIGINAL_OWNER/DGCNN-MidFusion-Model.git
```

#### Step 2: Create Feature Branch

```bash
# Keep main updated
git fetch upstream
git checkout main
git merge upstream/main

# Create feature branch
git checkout -b feature/your-feature-name

# Good branch names:
# feature/add-attention-mechanism
# fix/gradient-clipping-issue
# docs/update-installation-guide
# refactor/simplify-data-loading
```

#### Step 3: Make Changes

**Code Style:**
- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/)
- Use meaningful variable names
- Add docstrings to functions and classes
- Keep functions focused and small

**Format your code:**
```bash
# Install formatters
pip install black flake8 isort

# Format code
black src/
isort src/

# Check for issues
flake8 src/
```

**Example contribution structure:**

```python
def compute_weighted_iou(predictions: np.ndarray, labels: np.ndarray, 
                         weights: np.ndarray) -> float:
    """
    Compute weighted Intersection-over-Union.
    
    Weights classes by frequency to handle class imbalance.
    
    Args:
        predictions: (N,) predicted class indices
        labels: (N,) ground truth labels
        weights: (num_classes,) class weights
    
    Returns:
        weighted_iou: float, weighted IoU score
    
    Raises:
        ValueError: if shapes don't match
    
    Example:
        >>> preds = np.array([0, 1, 1, 2])
        >>> labels = np.array([0, 1, 2, 2])
        >>> weights = np.array([1.0, 1.0, 1.0])
        >>> iou = compute_weighted_iou(preds, labels, weights)
    """
    if predictions.shape != labels.shape:
        raise ValueError("Predictions and labels must have same shape")
    
    # Implementation
    return iou
```

#### Step 4: Write Tests

```python
# tests/test_metrics.py
import numpy as np
import pytest
from src.metrics import compute_weighted_iou

def test_compute_weighted_iou_perfect():
    """Test IoU with perfect predictions."""
    preds = np.array([0, 1, 2, 0, 1])
    labels = np.array([0, 1, 2, 0, 1])
    weights = np.array([1.0, 1.0, 1.0])
    
    iou = compute_weighted_iou(preds, labels, weights)
    assert iou == 1.0, "Perfect predictions should have IoU=1.0"

def test_compute_weighted_iou_shape_mismatch():
    """Test error handling for shape mismatch."""
    preds = np.array([0, 1, 2])
    labels = np.array([0, 1])
    weights = np.array([1.0, 1.0, 1.0])
    
    with pytest.raises(ValueError):
        compute_weighted_iou(preds, labels, weights)

# Run tests
# pytest tests/test_metrics.py -v
```

#### Step 5: Document Changes

Update relevant documentation:
- `README.md` - If new features
- `CHANGELOG.md` - If significant changes
- Docstrings - For new functions/classes

#### Step 6: Commit

```bash
# Stage changes
git add src/metrics.py
git add tests/test_metrics.py

# Commit with clear message
git commit -m "feat: add weighted IoU metric for handling class imbalance

- Implements weighted intersection-over-union calculation
- Useful for imbalanced datasets
- Includes comprehensive tests
- Tested on FRACTAL dataset

Closes #42"
```

**Commit message guidelines:**
```
<type>(<scope>): <subject>

<body>

<footer>
```

Types: `feat`, `fix`, `refactor`, `perf`, `docs`, `test`, `chore`

Example:
```
feat(metrics): add weighted IoU metric

Implements weighted IoU for imbalanced datasets:
- Calculates per-class IoU
- Applies class weights
- Returns single weighted score

Breaking change: compute_metrics() now requires weights parameter

Closes #42
```

#### Step 7: Push and Create PR

```bash
# Push to your fork
git push -u origin feature/your-feature-name

# On GitHub: Create pull request
```

**PR Template:**
```markdown
## Description
What does this PR do?

## Related Issues
Closes #123

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Tests added/updated
- [ ] Manual testing on [GPU/CPU]
- [ ] Results match expectations

## Checklist
- [ ] Code follows style guide
- [ ] Docstrings added
- [ ] Tests pass locally
- [ ] No breaking changes
- [ ] CHANGELOG updated
```

#### Step 8: Address Review Comments

```bash
# Make requested changes
vim src/metrics.py

# Commit and push
git add src/metrics.py
git commit -m "refactor: improve code clarity based on review feedback"
git push origin feature/your-feature-name

# PR auto-updates with new commits
```

## 🧪 Testing Requirements

### Run All Tests

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run all tests
pytest tests/ -v

# Generate coverage report
pytest tests/ --cov=src --cov-report=html
```

### Test Coverage

Aim for >80% coverage. To check:

```bash
# Coverage report
pytest tests/ --cov=src --cov-report=term-missing

# View in HTML
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
start htmlcov/index.html  # Windows
```

## 📖 Documentation

### Adding Docstrings

```python
def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Train model for one epoch.
    
    Args:
        model (nn.Module): Neural network model
        loader (DataLoader): Training data loader
        criterion (nn.Module): Loss function
        optimizer (torch.optim.Optimizer): Optimizer
        device (torch.device): Device to train on
    
    Returns:
        float: Average loss over epoch
    
    Raises:
        RuntimeError: If GPU out of memory
    
    Example:
        >>> model = create_model(config)
        >>> loss = train_one_epoch(model, loader, criterion, 
        ...                       optimizer, torch.device('cuda'))
    """
    # Implementation
    return loss
```

### README Updates

Keep examples and install instructions current.

### CHANGELOG Updates

```markdown
## [Unreleased]

### Added
- New weighted IoU metric for imbalanced datasets

### Fixed
- Fixed gradient clipping for long sequences

### Changed
- Refactored data loading pipeline for better performance

### Removed
- Deprecated `compute_metrics_v1` function
```

## 🚀 Development Setup

```bash
# Clone and setup
git clone https://github.com/yourusername/DGCNN-MidFusion-Model.git
cd DGCNN-MidFusion-Model

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or: venv\Scripts\activate

# Install in development mode
pip install -e .
pip install -r requirements.txt
pip install pytest pytest-cov black flake8 isort mypy
```

## 📊 Performance Benchmarking

When optimizing code, include benchmarks:

```python
import time
import numpy as np

# Before optimization
t0 = time.time()
for _ in range(1000):
    result = slow_function()
t_old = time.time() - t0

# After optimization
t0 = time.time()
for _ in range(1000):
    result = fast_function()
t_new = time.time() - t0

print(f"Speedup: {t_old / t_new:.2f}x")  # Should be > 1.0
```

## 🔗 Resources

- [GitHub Flow](https://guides.github.com/introduction/flow/)
- [PEP 8 Style Guide](https://www.python.org/dev/peps/pep-0008/)
- [Conventional Commits](https://www.conventionalcommits.org/)
- [NumPy Docstring Guide](https://numpydoc.readthedocs.io/en/latest/format.html)

## ✅ Contributor Checklist

Before submitting PR:

- [ ] Tests pass: `pytest tests/ -v`
- [ ] Code formatted: `black src/`
- [ ] No linting issues: `flake8 src/`
- [ ] Docstrings added for new functions
- [ ] Commit messages follow convention
- [ ] CHANGELOG.md updated
- [ ] No hardcoded paths or credentials
- [ ] Branch is up to date with main

## 🎉 Recognition

Contributors will be recognized in:
- CHANGELOG.md
- GitHub contributors page
- Project website (if applicable)

---

**Thank you for contributing to make this project better! 💙**
