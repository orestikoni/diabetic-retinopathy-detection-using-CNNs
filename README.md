# Diabetic Retinopathy Severity Classification using CNNs

## Overview

This project implements a comparative study of deep learning models for **diabetic retinopathy (DR) severity classification** using convolutional neural networks. The study compares a standard **ResNet-50 baseline** against an experimental **SE-ResNet-50 (Squeeze-and-Excitation ResNet-50)** architecture to evaluate the impact of channel attention mechanisms on retinal fundus image classification.

## Project Objectives

- Classify diabetic retinopathy severity into 5 grades using retinal fundus images
- Compare baseline ResNet-50 with SE-ResNet-50 attention-enhanced model
- Achieve reproducible results through fixed seeding and identical hyperparameters
- Generate comprehensive evaluation metrics and visualizations for both models

## Dataset

**Dataset:** APTOS 2019 (Asian Pacific Tele-Ophthalmology Society)
- **Image Size:** 224×224 pixels (pre-processed)
- **Number of Classes:** 5
  - 0: No DR (No Diabetic Retinopathy)
  - 1: Mild
  - 2: Moderate
  - 3: Severe
  - 4: Proliferate DR (Proliferative Diabetic Retinopathy)
- **Data Split:** 
  - Training: 80%
  - Validation: 10%
  - Testing: 10%
  - Stratified split with fixed seed (saved to `split_indices.json` for consistency)

## Model Architectures

### 1. ResNet-50 Baseline (`resnet50-baseline.ipynb`)

Standard ResNet-50 pre-trained on ImageNet with the following configuration:
- **Architecture:** 50-layer residual network
- **Input:** 224×224 RGB images
- **Output:** 5-class probability distribution
- **Backbone:** PyTorch torchvision ResNet50
- **Fine-tuning:** Last fully connected layer adapted for 5 classes

**Key Features:**
- Deterministic training with fixed seed (SEED = 42)
- CosineAnnealingLR scheduler for smooth learning rate decay
- WeightedRandomSampler for handling class imbalance
- Comprehensive evaluation metrics and visualizations

### 2. SE-ResNet-50 Experimental (`se-resnet50-experimental.ipynb`)

Custom SE-ResNet-50 with Squeeze-and-Excitation blocks inserted into each bottleneck:
- **Architecture:** ResNet-50 + Squeeze-and-Excitation modules
- **Attention Mechanism:** Channel attention to recalibrate feature responses
- **Input:** 224×224 RGB images
- **Output:** 5-class probability distribution

**Key Differences from Baseline:**
- SE blocks add learnable channel-wise attention weights
- Identical hyperparameters to baseline for fair comparison
- Same data split (loaded from `split_indices.json`)
- Identical scheduler and optimizer configuration

**SE Block Details:**
- Reduces channel dimension by a reduction ratio (typically 16)
- Learns which channels are more important for prediction
- Applied after each residual block's convolution

## Training Configuration

Both models share identical hyperparameters for fair comparison:

```
Image Size:      224×224
Batch Size:      32
Number of Epochs: 50
Learning Rate:   1e-4
Weight Decay:    1e-4
LR Min (scheduler): 1e-7
Optimizer:       Adam
Scheduler:       CosineAnnealingLR
Loss Function:   Cross-Entropy Loss
Device:          GPU (CUDA if available)
Seed:            42 (for reproducibility)
```

## Project Structure

```
diabetic-retinopathy-detection-using-CNNs/
├── README.md                          # This file
├── resnet50-baseline.ipynb            # ResNet-50 baseline model training
├── se-resnet50-experimental.ipynb     # SE-ResNet-50 experimental model
└── outputs/                           # Generated during training
    ├── resnet50_final/
    │   ├── best_model.pth
    │   ├── final_model.pth
    │   ├── training_history.json
    │   └── metrics.json
    ├── se_resnet50_final/
    │   ├── best_model.pth
    │   ├── final_model.pth
    │   ├── training_history.json
    │   └── metrics.json
    └── split_indices.json             # Train/val/test split (shared)
```

## Notebook Structure

### resnet50-baseline.ipynb

1. **Environment & Reproducibility** - Fixed seed configuration for deterministic results
2. **Configuration** - Hyperparameter and path setup
3. **Imports** - Required libraries and dependencies
4. **Dataset Preparation** - Image loading, preprocessing, and augmentation
5. **Class Imbalance Handling** - WeightedRandomSampler implementation
6. **Model Definition** - ResNet-50 initialization
7. **Training Pipeline** - Training loop with validation
8. **Model Evaluation** - Metrics calculation and analysis
9. **Visualization** - Confusion matrices, ROC curves, and loss/accuracy plots
10. **Predictions & Inference** - Testing on validation/test sets

### se-resnet50-experimental.ipynb

Follows the same structure as the baseline notebook with the addition of:
- **SE Block Implementation** - Squeeze-and-Excitation module definition
- **SE-ResNet-50 Construction** - Integration of SE blocks into ResNet-50
- **Attention Visualization** - Channel attention weight analysis

## Key Metrics & Evaluation

Both models are evaluated using:

- **Classification Metrics:**
  - Accuracy
  - Precision (per-class and macro-averaged)
  - Recall (per-class and macro-averaged)
  - F1-Score (per-class and macro-averaged)
  - Matthews Correlation Coefficient (MCC)
  - Cohen's Kappa

- **Ranking Metrics:**
  - Mean Absolute Error (MAE) - measures severity ordering
  - Weighted Kappa (accounts for ordinal nature of severity)

- **ROC Analysis:**
  - One-vs-Rest ROC curves
  - Area Under Curve (AUC) per class
  - Micro and Macro AUC

- **Confusion Analysis:**
  - Confusion matrices (count and normalized)
  - Per-class performance breakdown

## Dependencies

```
Python >= 3.8
PyTorch >= 1.9.0
torchvision >= 0.10.0
NumPy >= 1.19.0
Pandas >= 1.1.0
Matplotlib >= 3.3.0
Seaborn >= 0.11.0
scikit-learn >= 0.24.0
Pillow >= 8.0.0
opencv-python >= 4.5.0
```

## Installation & Setup

### Local Installation

```bash
# Clone the repository
git clone https://github.com/orestikoni/diabetic-retinopathy-detection-using-CNNs.git
cd diabetic-retinopathy-detection-using-CNNs

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Or install manually
pip install torch torchvision numpy pandas matplotlib seaborn scikit-learn opencv-python pillow
```

### Kaggle Notebook

The notebooks are designed to run on Kaggle with the APTOS 2019 dataset available in the Kaggle datasets:
- Dataset: `sovitrath/diabetic-retinopathy-224x224-2019-data`
- Split indices: `orestikoni/dr-split-indices` (created by baseline notebook)

## Usage

### Running the Baseline Model

1. Open `resnet50-baseline.ipynb` in Jupyter Notebook or VS Code
2. Ensure the dataset path is correct (Kaggle or local path)
3. Run all cells sequentially
4. This will generate:
   - Trained model weights
   - Training history
   - Evaluation metrics
   - `split_indices.json` for consistent splits

### Running the Experimental SE-ResNet-50 Model

1. Open `se-resnet50-experimental.ipynb`
2. Ensure the dataset path and split indices path are correct
3. Run all cells sequentially
4. This will:
   - Load the identical data split from `split_indices.json`
   - Train SE-ResNet-50 with identical hyperparameters
   - Generate evaluation metrics using the same test set

### Comparing Models

Both notebooks generate metrics and visualizations that can be compared:
- Training/validation loss curves
- Accuracy progression
- Confusion matrices
- ROC curves
- Per-class performance breakdown

## Key Implementation Details

### Reproducibility

- **Fixed Seed:** SEED = 42 set for Python, NumPy, PyTorch, and CUDA
- **Deterministic Mode:** `torch.backends.cudnn.deterministic = True`
- **Benchmark Disabled:** `torch.backends.cudnn.benchmark = False`
- **Shared Data Split:** Both models use identical train/val/test split saved in JSON

### Data Augmentation

- Horizontal flips
- Random rotations (±20°)
- Color jittering (brightness, contrast, saturation)
- Normalization using ImageNet statistics

### Handling Class Imbalance

- **WeightedRandomSampler:** Assigns higher sampling probability to minority classes
- **Stratified Split:** Maintains class distribution across train/val/test sets

### Learning Rate Scheduling

- **Scheduler:** CosineAnnealingLR
  - Smoothly anneals learning rate from `lr` to `lr_min` over `T_max` epochs
  - Formula: $\eta_t = \eta_{min} + \frac{1}{2}(\eta_0 - \eta_{min})(1 + \cos(\pi t / T_{max}))$
  - Provides warm restarts without explicit restart mechanism

## Expected Results

The SE-ResNet-50 model with channel attention is expected to:
- Show improved classification accuracy on difficult cases
- Demonstrate better generalization through attention mechanisms
- Provide interpretability via channel attention weights
- Maintain computational efficiency while adding selective refinement

## Visualization Outputs

Both notebooks generate:
- Training/validation loss and accuracy curves
- Confusion matrices (raw counts and normalized percentages)
- ROC curves for each severity class
- Per-class precision, recall, and F1-score bar plots
- Feature importance/attention weight visualizations (SE-ResNet-50)

## References & Citations

**Dataset:** 
- Kaggle APTOS 2019 Blindness Detection Challenge
- https://www.kaggle.com/c/aptos2019-blindness-detection

**Model Architectures:**
- ResNet: He, K., Zhang, X., Ren, S., & Sun, J. (2016). "Deep Residual Learning for Image Recognition"
- SE-ResNet: Hu, J., Shen, L., & Sun, G. (2018). "Squeeze-and-Excitation Networks"

## Author

Orestikoni  
GitHub: [@orestikoni](https://github.com/orestikoni)

## License

This project is available for academic and research purposes.

## Future Improvements

- [ ] Implement other attention mechanisms (CBAM, ECA-Net)
- [ ] Experiment with different backbone architectures (Vision Transformer, DenseNet)
- [ ] Add ensemble methods combining both models
- [ ] Implement explainability techniques (Grad-CAM, attention visualization)
- [ ] Cross-validate with other DR datasets
- [ ] Deploy as web API for real-time predictions

## Contact & Support

For issues, questions, or contributions, please open an issue on the GitHub repository.

---

**Last Updated:** September 2026
