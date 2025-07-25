# RepViT Semantic Segmentation API Documentation

This document provides comprehensive documentation for the RepViT semantic segmentation module, built on top of MMSegmentation framework.

## Table of Contents

1. [Overview](#overview)
2. [Model Architecture](#model-architecture)
3. [Training API](#training-api)
4. [Evaluation API](#evaluation-api)
5. [Configuration System](#configuration-system)
6. [Utilities](#utilities)
7. [Usage Examples](#usage-examples)

---

## Overview

The RepViT segmentation module provides efficient semantic segmentation models based on the RepViT backbone. It integrates with the MMSegmentation framework for comprehensive segmentation pipeline support.

### Key Features

- **Efficient Backbone**: RepViT models optimized for mobile deployment
- **MMSegmentation Integration**: Full compatibility with MMSeg ecosystem
- **Multiple Decoders**: Support for various segmentation heads
- **Flexible Training**: Knowledge distillation and advanced training strategies

---

## Model Architecture

### RepViT Backbone (`segmentation/repvit.py`)

#### `RepViT(nn.Module)`
RepViT backbone adapted for segmentation tasks.

**Constructor Parameters:**
- `cfgs`: Model configuration defining layer architecture
- `num_classes`: Number of output classes (not used in backbone mode)
- `distillation`: Enable distillation mode

**Key Methods:**
- `forward(x)`: Forward pass returning multi-scale features
- `init_weights(pretrained=None)`: Initialize weights from pretrained checkpoint

**Supported Variants:**
- `repvit_m0_9`: Ultra-fast variant for real-time segmentation
- `repvit_m1_1`: Balanced accuracy/speed tradeoff
- `repvit_m1_5`: Higher accuracy variant
- `repvit_m2_3`: Best accuracy variant

**Usage:**
```python
# Configuration example
backbone = dict(
    type='RepViT',
    cfgs=repvit_m1_1_cfgs,
    pretrained='path/to/pretrained.pth'
)
```

### AlignResize Transform (`segmentation/align_resize.py`)

#### `AlignResize(object)`
Custom resize transformation that aligns dimensions for efficient processing.

**Constructor Parameters:**
- `size`: Target size for resizing
- `interpolation`: Interpolation method

**Methods:**
- `__call__(img)`: Apply resize transformation

**Usage:**
```python
from align_resize import AlignResize

transform = AlignResize(size=(512, 512))
resized_image = transform(image)
```

---

## Training API

### Training Script (`segmentation/tools/train.py`)

The main training script for RepViT segmentation models.

**Key Arguments:**
- `--config`: Path to configuration file
- `--work-dir`: Working directory for outputs
- `--load-from`: Load model from checkpoint
- `--resume-from`: Resume training from checkpoint
- `--no-validate`: Skip validation during training
- `--gpus`: Number of GPUs to use
- `--seed`: Random seed for reproducibility

**Usage:**
```bash
# Single GPU training
python tools/train.py configs/repvit/repvit_m1_1_upernet_512x512_ade20k.py

# Multi-GPU training
bash tools/dist_train.sh configs/repvit/repvit_m1_1_upernet_512x512_ade20k.py 8

# Resume training
python tools/train.py configs/repvit/repvit_m1_1_upernet_512x512_ade20k.py \
    --resume-from work_dirs/repvit_m1_1/latest.pth
```

### Distributed Training

#### `dist_train.sh`
Script for distributed training across multiple GPUs.

**Parameters:**
- `CONFIG`: Path to config file
- `GPUS`: Number of GPUs
- `PORT`: Communication port (optional)

**Usage:**
```bash
# 8-GPU training
bash tools/dist_train.sh configs/repvit/repvit_m1_1_upernet_512x512_ade20k.py 8

# With custom port
bash tools/dist_train.sh configs/repvit/repvit_m1_1_upernet_512x512_ade20k.py 8 29500
```

---

## Evaluation API

### Test Script (`segmentation/tools/test.py`)

#### Key Functions

##### `single_gpu_test(model, data_loader, show=False, out_dir=None)`
Test model on single GPU.

**Parameters:**
- `model`: Model to test
- `data_loader`: Test data loader
- `show`: Whether to show results
- `out_dir`: Output directory for results

##### `multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False)`
Test model on multiple GPUs.

**Usage:**
```bash
# Single GPU testing
python tools/test.py configs/repvit/repvit_m1_1_upernet_512x512_ade20k.py \
    checkpoints/repvit_m1_1_upernet_ade20k.pth \
    --eval mIoU

# Multi-GPU testing
bash tools/dist_test.sh configs/repvit/repvit_m1_1_upernet_512x512_ade20k.py \
    checkpoints/repvit_m1_1_upernet_ade20k.pth 8 \
    --eval mIoU

# Generate visualization results
python tools/test.py configs/repvit/repvit_m1_1_upernet_512x512_ade20k.py \
    checkpoints/repvit_m1_1_upernet_ade20k.pth \
    --show-dir results/
```

### Evaluation Metrics

The framework supports various evaluation metrics:

- **mIoU**: Mean Intersection over Union
- **mAcc**: Mean pixel accuracy
- **aAcc**: All pixel accuracy
- **mDice**: Mean Dice coefficient
- **mFscore**: Mean F-score

---

## Configuration System

### Configuration Structure

RepViT segmentation uses MMSegmentation's config system. Key components:

#### Model Configuration
```python
# Example config structure
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='RepViT',
        cfgs=repvit_m1_1_cfgs,
        pretrained='pretrained/repvit_m1_1_distill_300e.pth'
    ),
    decode_head=dict(
        type='UPerHead',
        in_channels=[64, 128, 256, 512],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=150,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0
        )
    ),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=256,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=150,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=0.4
        )
    ),
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)
```

#### Dataset Configuration
```python
# ADE20K dataset example
dataset_type = 'ADE20KDataset'
data_root = 'data/ade/ADEChallengeData2016'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], 
    std=[58.395, 57.12, 57.375], 
    to_rgb=True
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='Resize', img_scale=(2048, 512), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=(512, 512), pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
```

### Common Configurations

#### Available Configs
- `repvit_m0_9_upernet_512x512_ade20k.py`: Ultra-fast variant
- `repvit_m1_1_upernet_512x512_ade20k.py`: Recommended variant
- `repvit_m1_5_upernet_512x512_ade20k.py`: High accuracy variant
- `repvit_m2_3_upernet_512x512_ade20k.py`: Best accuracy variant

---

## Utilities

### Data Preprocessing

#### AlignResize
Ensures input dimensions are properly aligned for efficient processing.

```python
from align_resize import AlignResize

# Create align resize transform
align_resize = AlignResize(size=(512, 512))

# Apply to image
aligned_image = align_resize(original_image)
```

### Model Utilities

#### Weight Initialization
```python
# Initialize backbone from pretrained classification model
backbone.init_weights(pretrained='pretrained/repvit_m1_1_distill_300e.pth')
```

#### Feature Extraction
```python
# Extract multi-scale features
features = backbone(input_tensor)
# features is a list of feature maps at different scales
```

---

## Usage Examples

### 1. Training Custom Segmentation Model

```python
# config.py
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='RepViT',
        cfgs=repvit_m1_1_cfgs,
        pretrained='pretrained/repvit_m1_1_distill_300e.pth'
    ),
    decode_head=dict(
        type='UPerHead',
        in_channels=[64, 128, 256, 512],
        channels=512,
        num_classes=19,  # Custom number of classes
        # ... other parameters
    )
)

# Training command
python tools/train.py config.py --work-dir work_dirs/custom_model
```

### 2. Fine-tuning on Custom Dataset

```python
# Modify dataset configuration
dataset_type = 'CustomDataset'
data_root = 'data/custom_dataset'

# Update data pipeline for custom classes
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),  # Keep all classes
    # ... rest of pipeline
]

# Training with custom dataset
python tools/train.py configs/custom/repvit_m1_1_custom.py
```

### 3. Inference on Single Image

```python
from mmseg.apis import inference_segmentor, init_segmentor

# Initialize model
config_file = 'configs/repvit/repvit_m1_1_upernet_512x512_ade20k.py'
checkpoint_file = 'checkpoints/repvit_m1_1_upernet_ade20k.pth'
model = init_segmentor(config_file, checkpoint_file, device='cuda:0')

# Inference
img = 'demo/demo.jpg'
result = inference_segmentor(model, img)

# Visualize result
model.show_result(img, result, out_file='result.jpg')
```

### 4. Batch Inference

```python
import os
from mmseg.apis import inference_segmentor, init_segmentor

# Initialize model
model = init_segmentor(config_file, checkpoint_file, device='cuda:0')

# Process directory of images
img_dir = 'input_images/'
out_dir = 'output_results/'

for img_name in os.listdir(img_dir):
    img_path = os.path.join(img_dir, img_name)
    result = inference_segmentor(model, img_path)
    
    out_path = os.path.join(out_dir, img_name)
    model.show_result(img_path, result, out_file=out_path)
```

### 5. Model Deployment

```python
# Convert to ONNX
from mmseg.apis import init_segmentor
import torch

model = init_segmentor(config_file, checkpoint_file, device='cpu')
model.eval()

# Export to ONNX
dummy_input = torch.randn(1, 3, 512, 512)
torch.onnx.export(
    model,
    dummy_input,
    'repvit_segmentation.onnx',
    export_params=True,
    opset_version=11,
    input_names=['input'],
    output_names=['output']
)
```

### 6. Performance Benchmarking

```bash
# Measure inference speed
python tools/benchmark.py \
    configs/repvit/repvit_m1_1_upernet_512x512_ade20k.py \
    checkpoints/repvit_m1_1_upernet_ade20k.pth \
    --log-interval 100

# Test FLOPs and parameters
python tools/get_flops.py \
    configs/repvit/repvit_m1_1_upernet_512x512_ade20k.py \
    --shape 512 512
```

---

## Performance Optimization

### Best Practices

1. **Input Size**: Use appropriate input sizes (512x512 recommended for RepViT)
2. **Batch Size**: Optimize based on GPU memory
3. **Mixed Precision**: Enable for faster training
4. **Data Loading**: Use sufficient workers for data loading

### Mobile Deployment

```python
# Optimize for mobile deployment
import utils

# Replace batch normalization for inference
utils.replace_batchnorm(model.backbone)

# Use CPU for mobile deployment
model = model.cpu()
model.eval()
```

---

This documentation covers the comprehensive segmentation API for RepViT models. For specific implementation details and advanced configurations, refer to the MMSegmentation documentation and the provided example configs.