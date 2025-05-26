# MFBP: Multi-Feature Bilinear Pooling for Spatial Transcriptomics

MFBP is a deep learning model for spatial transcriptomics gene expression prediction, capable of predicting gene expression profiles from histopathology images.

## 🚀 Key Features

- **Efficient Gene Expression Prediction**: Predict spatial transcriptomics data from histopathology images
- **Multi-GPU Training Support**: Distributed training for improved efficiency
- **Data Augmentation**: 7x data augmentation support for better model generalization
- **STEm-Compatible Evaluation Metrics**: Output evaluation metrics in the same format as STEm
- **WSI-based Spatial Visualization**: Generate spatial gene expression maps on tissue images
- **Comprehensive Visualization Suite**: Gene variation curves, correlation analysis, and summary reports
- **Flexible Configuration System**: Simplified command-line parameter configuration

## 📊 Evaluation Metrics

MFBP outputs exactly the same evaluation metrics as STEm:

- **PCC-10**: Average Pearson correlation coefficient for top 10 genes
- **PCC-50**: Average Pearson correlation coefficient for top 50 genes
- **PCC-200**: Average Pearson correlation coefficient for top 200 genes
- **MSE**: Mean Squared Error
- **MAE**: Mean Absolute Error
- **RVD**: Relative Variance Difference

## 🛠️ Environment Requirements

### System Requirements
- Linux system (Ubuntu 18.04+ recommended)
- Python 3.8+
- CUDA 11.0+ (for GPU training)

### Dependencies
```bash
# Core dependencies
torch>=1.12.0
pytorch-lightning>=1.8.0
torchmetrics>=0.10.0
numpy>=1.21.0
pandas>=1.3.0
scipy>=1.7.0
scikit-learn>=1.0.0

# Data processing
scanpy>=1.8.0
anndata>=0.8.0
h5py>=3.6.0

# Configuration and utilities
addict>=2.4.0
tqdm>=4.62.0
matplotlib>=3.5.0
seaborn>=0.11.0

# Optional: for visualization
tensorboard>=2.8.0
wandb>=0.12.0
```

## 📁 Project Structure

```
MFBP/
├── src/                          # Source code
│   ├── dataset/                  # Dataset processing
│   │   ├── hest_dataset.py      # Main dataset class
│   │   └── data_interface.py    # Data interface
│   ├── model/                    # Model definitions
│   │   ├── MFBP/                # MFBP model
│   │   │   └── MFBP.py         # Model implementation
│   │   └── model_interface.py   # Model interface
│   ├── visualization/            # Visualization module
│   │   ├── __init__.py          # Module initialization
│   │   └── gene_visualizer.py   # WSI-based spatial visualization
│   └── main.py                  # Main training script

├── logs/                        # Logs and results (organized by dataset/model)
│   ├── {dataset}/               # Dataset-specific logs
│   │   └── {model}/             # Model-specific logs
│   │       ├── evaluation_results/ # Evaluation metrics
│   │       ├── vis/             # Visualization outputs
│   │       └── lightning_logs/  # Training logs
└── README.md                    # Project documentation
```

## 🚀 Quick Start

### 1. Environment Setup

First, ensure your environment has the required dependencies:

```bash
# Activate conda environment
conda activate stem_env

# Install missing dependencies if needed
pip install pytorch-lightning torchmetrics addict
```

### 2. Data Preparation

Ensure your data is organized in the following structure:

```
/data/ouyangjiarui/stem/hest1k_datasets/PRAD/
├── st/                          # Spatial transcriptomics data (.h5ad files)
├── processed_data/              # Preprocessed data
│   ├── selected_gene_list.txt   # Selected gene list
│   └── all_slide_lst.txt       # All slide list
└── 1spot_uni_ebd/              # Image embedding features
    └── {slide_id}_uni.pt       # Feature files for each slide
```

### 3. Basic Training

#### Using Simplified Configuration (Recommended)

```bash
# Basic training - PRAD dataset
python src/main.py --dataset PRAD --gpus 1

# Multi-GPU training
python src/main.py --dataset PRAD --gpus 4

# Custom parameters
python src/main.py --dataset PRAD --model MFBP --encoder uni \
    --gpus 4 --epochs 200 --batch-size 256 --lr 1e-4

# her2st dataset
python src/main.py --dataset her2st --gpus 2
```

#### Available Parameters

| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| `--dataset` | Dataset name | Required | PRAD, her2st |
| `--model` | Model name | MFBP | MFBP |
| `--encoder` | Encoder type | Auto-select | uni, conch |
| `--gpus` | Number of GPUs | 1 | 1, 2, 4, 8 |
| `--epochs` | Training epochs | 200 | Any positive integer |
| `--batch-size` | Batch size | 256 | Any positive integer |
| `--lr` | Learning rate | 1e-4 | Any float |
| `--weight-decay` | Weight decay | 1e-4 | Any float |
| `--use-augmented` | Use data augmentation | True | True/False |
| `--expand-augmented` | Expand augmented data | True | True/False |
| `--mode` | Run mode | train | train, test |

### 4. Test Mode

```bash
# Test mode
python src/main.py --dataset PRAD --gpus 1 --mode test
```

## 📊 Evaluation Metrics

MFBP automatically calculates and outputs evaluation metrics during training. The metrics are saved to `logs/{dataset}/{model}/evaluation_results/` directory.

## 🎨 Visualization Features

MFBP provides comprehensive visualization capabilities for spatial transcriptomics analysis.

**Key Feature**: Visualizations are generated **only once at the end of training** to avoid cluttering the output directory with intermediate files.

### 1. WSI-based Spatial Gene Expression Maps

Visualize gene expression patterns overlaid on tissue images (like STEm):

**Features:**
- **Ground Truth vs Prediction**: Side-by-side comparison on tissue images
- **Multiple Gene Support**: Visualize marker genes for different datasets
- **Automatic WSI Loading**: Supports .tif, .jpg, .png, .jpeg formats
- **Correlation Display**: Shows correlation coefficient in plot titles
- **Fallback Mode**: Works without tissue images if unavailable

### 2. Gene Variation Analysis

Comprehensive analysis of gene expression patterns:

**Includes:**
- **Gene Variation Curves**: Mean and variance analysis (normalized and absolute)
- **Correlation Analysis**: Gene-wise correlation distribution and ranking
- **Performance Summary**: Comprehensive model performance report

### 3. Automatic Visualization After Training

Visualizations are automatically generated **once at the end of training** (not every epoch):

```bash
# Training with final visualization (enabled by default)
python src/main.py --dataset PRAD --gpus 4

# Disable visualization if needed (faster training, no visualization files)
python src/main.py --dataset PRAD --gpus 4 --enable_visualization False
```

**Output Structure:**
```
logs/PRAD/MFBP/vis/
├── val_final/                      # Final validation visualizations
│   ├── val_final_gene_variation_curves.png
│   ├── val_final_correlation_analysis.png
│   ├── val_final_summary_report.png
│   ├── marker_gene_KLK3.png        # Spatial expression maps
│   ├── marker_gene_AR.png
│   └── ...
└── test_final/                     # Final test visualizations
    └── ...
```

**Benefits:**
- **Cleaner output**: No intermediate visualization files during training
- **Faster training**: No visualization overhead during epochs
- **Final results**: Only the best/final model results are visualized

### 4. Dataset-Specific Marker Genes

MFBP automatically selects appropriate marker genes for visualization:

- **PRAD**: KLK3, AR, FOLH1, ACPP, KLK2, STEAP2, PSMA, NKX3-1
- **her2st**: ERBB2, ESR1, PGR, MKI67, TP53, BRCA1, BRCA2
- **Default**: CD3E, CD4, CD8A, CD19, CD68, PTPRC, VIM, KRT19

## 📈 Training Monitoring

### Training Process Output

During training, the system automatically outputs:
- Basic training metrics (loss, pearson, etc.)
- Detailed evaluation metrics (every validation epoch)
- Comprehensive visualizations (generated once at training completion)
- Auto-save to organized log directories

### Output Structure

```bash
logs/
├── PRAD/                        # Dataset name
│   └── MFBP/                    # Model name
│       ├── evaluation_results/   # Evaluation metrics
│       ├── vis/                 # Visualizations
│       └── lightning_logs/      # Training logs
└── her2st/
    └── MFBP/
        ├── evaluation_results/
        ├── vis/
        └── lightning_logs/
```

### Log Files

```bash
# View training logs
ls logs/

# View evaluation results
ls logs/evaluation_results/
cat logs/evaluation_results/val_metrics_epoch_10_*.txt
```

## 🔧 Advanced Configuration

### Multi-GPU Training Optimization

```bash
# DDP strategy (recommended)
python src/main.py --dataset PRAD --gpus 4 --strategy ddp

# Sync BatchNorm
python src/main.py --dataset PRAD --gpus 4 --sync-batchnorm
```

### Data Augmentation Options

```bash
# Disable data augmentation
python src/main.py --dataset PRAD --gpus 1 --use-augmented False

# Use augmentation but don't expand
python src/main.py --dataset PRAD --gpus 1 --expand-augmented False
```

## 📋 Supported Datasets

### PRAD (Prostate Adenocarcinoma)
- **Path**: `/data/ouyangjiarui/stem/hest1k_datasets/PRAD/`
- **Validation set**: MEND139
- **Test set**: MEND140
- **Recommended encoder**: uni

### her2st (Breast Cancer)
- **Path**: `/data/ouyangjiarui/stem/hest1k_datasets/her2st/`
- **Validation set**: A1,B1
- **Test set**: C1,D1
- **Recommended encoder**: conch

## 🐛 Common Issues

### 1. Environment Issues

```bash
# Missing dependencies
ModuleNotFoundError: No module named 'torch'
# Solution: Activate the correct conda environment
conda activate stem_env

# Missing torchmetrics
ModuleNotFoundError: No module named 'torchmetrics'
# Solution: Install missing packages
pip install torchmetrics
```

### 2. Data Issues

```bash
# Data files not found
FileNotFoundError: ST file does not exist
# Solution: Check if data path is correct
ls /data/ouyangjiarui/stem/hest1k_datasets/PRAD/st/
```

### 3. GPU Issues

```bash
# CUDA out of memory
RuntimeError: CUDA out of memory
# Solution: Reduce batch size
python src/main.py --dataset PRAD --batch-size 128
```

## 📊 Performance Benchmarks

### Training Time (PRAD Dataset)

| GPU Configuration | Batch Size | Time per Epoch | Total Training Time (200 epochs) |
|-------------------|------------|----------------|-----------------------------------|
| 1x RTX 3090 | 256 | ~15 minutes | ~50 hours |
| 4x RTX 3090 | 256 | ~4 minutes | ~13 hours |
| 8x A100 | 512 | ~2 minutes | ~7 hours |

### Memory Usage

| Configuration | GPU Memory | System Memory |
|---------------|------------|---------------|
| Batch 256 | ~12GB | ~16GB |
| Batch 512 | ~20GB | ~24GB |

## 🤝 Comparison with STEm

| Feature | STEm | MFBP |
|---------|------|------|
| Evaluation Metrics | PCC-10/50/200, MSE, MAE, RVD | ✅ Exactly the same |
| Gene Normalization | log2(+1) | ✅ Exactly the same |
| Data Format | .h5ad | ✅ Compatible |
| Output Format | Text files | ✅ Same format |

## 📝 Changelog

### v1.0.0 (Current Version)
- ✅ Implemented MFBP model
- ✅ Added STEm-compatible evaluation metrics
- ✅ Multi-GPU training support
- ✅ Simplified configuration system
- ✅ Data augmentation support

## 📄 License

This project is licensed under the MIT License.

## 🙋‍♂️ Support

If you encounter issues:
1. Check the Common Issues section in this README
2. Run test scripts to verify your environment
3. Check log files for detailed error information

---

**Quick Start Example**:
```bash
# 1. Activate environment
conda activate stem_env

# 2. Install dependencies
pip install pytorch-lightning torchmetrics addict

# 3. Start training
python src/main.py --dataset PRAD --gpus 1

# 4. View results
ls logs/PRAD/MFBP/evaluation_results/
``` 