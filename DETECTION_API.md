# RepViT Object Detection API Documentation

This document provides comprehensive documentation for the RepViT object detection module, built on top of MMDetection framework.

## Table of Contents

1. [Overview](#overview)
2. [Model Architecture](#model-architecture)
3. [Training API](#training-api)
4. [Testing and Evaluation](#testing-and-evaluation)
5. [Configuration System](#configuration-system)
6. [Custom Components](#custom-components)
7. [Usage Examples](#usage-examples)
8. [Performance Optimization](#performance-optimization)

---

## Overview

The RepViT detection module provides efficient object detection models using RepViT backbones. It integrates seamlessly with the MMDetection framework, supporting various detection heads and training strategies.

### Key Features

- **Efficient Backbone**: RepViT models optimized for mobile deployment
- **MMDetection Integration**: Full compatibility with MMDet ecosystem
- **Multiple Detection Heads**: Support for RetinaNet, FCOS, FPN, etc.
- **Advanced Training**: Knowledge distillation and mixed precision training
- **Flexible Deployment**: ONNX and TensorRT export support

---

## Model Architecture

### RepViT Backbone (`detection/repvit.py`)

#### `RepViT(nn.Module)`
RepViT backbone adapted for object detection tasks.

**Constructor Parameters:**
- `cfgs`: Model configuration defining layer architecture
- `out_indices`: Output layer indices for FPN (default: (0, 1, 2, 3))
- `frozen_stages`: Number of frozen stages (default: -1)
- `norm_eval`: Whether to freeze BN statistics (default: False)

**Key Methods:**
- `forward(x)`: Forward pass returning multi-scale features
- `init_weights(pretrained=None)`: Initialize weights from pretrained checkpoint
- `train(mode=True)`: Set training mode with frozen stage support

**Supported Variants:**
- `repvit_m0_9`: Ultra-fast variant for real-time detection
- `repvit_m1_1`: Balanced accuracy/speed tradeoff
- `repvit_m1_5`: Higher accuracy variant
- `repvit_m2_3`: Best accuracy variant

**Usage:**
```python
# Configuration example
backbone = dict(
    type='RepViT',
    cfgs=repvit_m1_1_cfgs,
    out_indices=(0, 1, 2, 3),
    frozen_stages=1,
    init_cfg=dict(type='Pretrained', checkpoint='pretrained/repvit_m1_1_distill_300e.pth')
)
```

### Feature Pyramid Network Integration

RepViT backbones work seamlessly with FPN for multi-scale feature extraction:

```python
neck = dict(
    type='FPN',
    in_channels=[64, 128, 256, 512],  # RepViT output channels
    out_channels=256,
    num_outs=5
)
```

---

## Training API

### Training Script (`detection/train.py`)

The main training script for RepViT detection models.

**Key Arguments:**
- `--config`: Path to configuration file
- `--work-dir`: Working directory for outputs
- `--load-from`: Load model from checkpoint
- `--resume-from`: Resume training from checkpoint
- `--no-validate`: Skip validation during training
- `--gpus`: Number of GPUs to use
- `--gpu-ids`: Specific GPU IDs to use
- `--seed`: Random seed for reproducibility
- `--deterministic`: Use deterministic algorithms

**Usage:**
```bash
# Single GPU training
python tools/train.py configs/repvit/retinanet_repvit_m1_1_fpn_1x_coco.py

# Multi-GPU training
bash tools/dist_train.sh configs/repvit/retinanet_repvit_m1_1_fpn_1x_coco.py 8

# Resume training
python tools/train.py configs/repvit/retinanet_repvit_m1_1_fpn_1x_coco.py \
    --resume-from work_dirs/retinanet_repvit_m1_1/latest.pth
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
bash tools/dist_train.sh configs/repvit/retinanet_repvit_m1_1_fpn_1x_coco.py 8

# With custom port
bash tools/dist_train.sh configs/repvit/retinanet_repvit_m1_1_fpn_1x_coco.py 8 29500
```

### SLURM Training

#### `slurm_train.sh`
Script for training on SLURM clusters.

**Usage:**
```bash
# SLURM training
bash tools/slurm_train.sh partition_name job_name \
    configs/repvit/retinanet_repvit_m1_1_fpn_1x_coco.py \
    work_dirs/retinanet_repvit_m1_1
```

---

## Testing and Evaluation

### Test Script (`detection/test.py`)

#### Key Functions

##### `single_gpu_test(model, data_loader, show=False, out_dir=None, show_score_thr=0.3)`
Test model on single GPU.

**Parameters:**
- `model`: Model to test
- `data_loader`: Test data loader
- `show`: Whether to show results
- `out_dir`: Output directory for results
- `show_score_thr`: Score threshold for visualization

##### `multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False)`
Test model on multiple GPUs.

**Usage:**
```bash
# Single GPU testing
python tools/test.py configs/repvit/retinanet_repvit_m1_1_fpn_1x_coco.py \
    checkpoints/retinanet_repvit_m1_1_fpn_1x_coco.pth \
    --eval bbox

# Multi-GPU testing
bash tools/dist_test.sh configs/repvit/retinanet_repvit_m1_1_fpn_1x_coco.py \
    checkpoints/retinanet_repvit_m1_1_fpn_1x_coco.pth 8 \
    --eval bbox

# Generate visualization results
python tools/test.py configs/repvit/retinanet_repvit_m1_1_fpn_1x_coco.py \
    checkpoints/retinanet_repvit_m1_1_fpn_1x_coco.pth \
    --show-dir results/ --show-score-thr 0.5
```

### Evaluation Metrics

The framework supports various COCO evaluation metrics:

- **bbox**: Bounding box detection metrics
- **AP**: Average Precision at IoU=0.50:0.95
- **AP50**: Average Precision at IoU=0.50
- **AP75**: Average Precision at IoU=0.75
- **APs/APm/APl**: AP for small/medium/large objects

---

## Configuration System

### Configuration Structure

RepViT detection uses MMDetection's config system. Key components:

#### Model Configuration
```python
# RetinaNet with RepViT backbone example
model = dict(
    type='RetinaNet',
    backbone=dict(
        type='RepViT',
        cfgs=repvit_m1_1_cfgs,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        init_cfg=dict(
            type='Pretrained', 
            checkpoint='pretrained/repvit_m1_1_distill_300e.pth'
        )
    ),
    neck=dict(
        type='FPN',
        in_channels=[64, 128, 256, 512],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5
    ),
    bbox_head=dict(
        type='RetinaHead',
        num_classes=80,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64, 128]
        ),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]
        ),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0
        ),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)
    ),
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1
        ),
        allowed_border=-1,
        pos_weight=-1,
        debug=False
    ),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100
    )
)
```

#### Dataset Configuration
```python
# COCO dataset example
dataset_type = 'CocoDataset'
data_root = 'data/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], 
    std=[58.395, 57.12, 57.375], 
    to_rgb=True
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ]
    )
]
```

### Common Configurations

#### Available Configs
- `retinanet_repvit_m0_9_fpn_1x_coco.py`: Ultra-fast RetinaNet
- `retinanet_repvit_m1_1_fpn_1x_coco.py`: Recommended RetinaNet
- `fcos_repvit_m1_1_fpn_1x_coco.py`: FCOS with RepViT
- `faster_rcnn_repvit_m1_5_fpn_1x_coco.py`: Two-stage detector

---

## Custom Components

### Custom Runner (`detection/mmcv_custom/runner/`)

#### `EpochBasedRunnerAmp`
Enhanced runner with automatic mixed precision support.

**Features:**
- Automatic mixed precision training
- Gradient scaling
- Loss scaling optimization

#### `DistOptimizerHook`
Distributed optimizer hook for multi-GPU training.

### Checkpoint Management (`detection/checkpoint.py`)

#### `CheckpointLoader`
Advanced checkpoint loading with flexible mapping.

**Key Features:**
- Automatic key mapping
- Partial loading support
- Resume from different architectures

**Usage:**
```python
from checkpoint import CheckpointLoader

loader = CheckpointLoader()
checkpoint = loader.load_checkpoint(
    'path/to/checkpoint.pth',
    map_location='cpu'
)
```

---

## Usage Examples

### 1. Training Custom Detection Model

```python
# Custom config for RepViT + RetinaNet
model = dict(
    type='RetinaNet',
    backbone=dict(
        type='RepViT',
        cfgs=repvit_m1_1_cfgs,
        out_indices=(0, 1, 2, 3),
        init_cfg=dict(
            type='Pretrained',
            checkpoint='pretrained/repvit_m1_1_distill_300e.pth'
        )
    ),
    # ... rest of configuration
)

# Training command
python tools/train.py config.py --work-dir work_dirs/custom_detector
```

### 2. Fine-tuning on Custom Dataset

```python
# Modify dataset configuration
dataset_type = 'CustomDataset'
data_root = 'data/custom_dataset'
classes = ('class1', 'class2', 'class3')  # Custom classes

# Update model for custom classes
model = dict(
    bbox_head=dict(
        num_classes=len(classes)  # Number of custom classes
    )
)

# Training with custom dataset
python tools/train.py configs/custom/retinanet_repvit_m1_1_custom.py
```

### 3. Inference on Single Image

```python
from mmdet.apis import inference_detector, init_detector

# Initialize model
config_file = 'configs/repvit/retinanet_repvit_m1_1_fpn_1x_coco.py'
checkpoint_file = 'checkpoints/retinanet_repvit_m1_1_fpn_1x_coco.pth'
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# Inference
img = 'demo/demo.jpg'
result = inference_detector(model, img)

# Visualize result
model.show_result(img, result, out_file='result.jpg', score_thr=0.5)
```

### 4. Batch Inference

```python
import os
from mmdet.apis import inference_detector, init_detector

# Initialize model
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# Process directory of images
img_dir = 'input_images/'
out_dir = 'output_results/'

for img_name in os.listdir(img_dir):
    img_path = os.path.join(img_dir, img_name)
    result = inference_detector(model, img_path)
    
    out_path = os.path.join(out_dir, img_name)
    model.show_result(img_path, result, out_file=out_path, score_thr=0.5)
```

### 5. Model Deployment

```python
# Convert to ONNX
from mmdet.apis import init_detector
import torch

model = init_detector(config_file, checkpoint_file, device='cpu')
model.eval()

# Export to ONNX
dummy_input = torch.randn(1, 3, 800, 1333)
torch.onnx.export(
    model,
    dummy_input,
    'repvit_detector.onnx',
    export_params=True,
    opset_version=11,
    input_names=['input'],
    output_names=['dets', 'labels']
)
```

### 6. Performance Benchmarking

```bash
# Measure inference speed
python tools/benchmark.py \
    configs/repvit/retinanet_repvit_m1_1_fpn_1x_coco.py \
    checkpoints/retinanet_repvit_m1_1_fpn_1x_coco.pth \
    --log-interval 100

# Test FLOPs and parameters
python tools/get_flops.py \
    configs/repvit/retinanet_repvit_m1_1_fpn_1x_coco.py \
    --shape 800 1333
```

### 7. Mixed Precision Training

```python
# Enable mixed precision in config
fp16 = dict(loss_scale=512.)

# Training with mixed precision
python tools/train.py configs/repvit/retinanet_repvit_m1_1_fpn_1x_coco.py \
    --cfg-options fp16.loss_scale=512
```

---

## Performance Optimization

### Best Practices

1. **Input Size**: Use appropriate input sizes for speed/accuracy tradeoff
2. **Batch Size**: Optimize based on GPU memory
3. **Mixed Precision**: Enable FP16 for faster training
4. **Data Loading**: Use sufficient workers for data loading
5. **Frozen Stages**: Freeze early stages for faster training

### Mobile Deployment

```python
# Optimize for mobile deployment
import utils

# Replace batch normalization for inference
utils.replace_batchnorm(model.backbone)

# Use CPU for mobile deployment
model = model.cpu()
model.eval()

# Export to mobile formats
# ONNX for general deployment
torch.onnx.export(model, dummy_input, 'detector.onnx')

# Core ML for iOS (requires additional tools)
# TensorRT for NVIDIA deployment
```

### Speed Optimization

```python
# Optimize model for speed
test_cfg = dict(
    nms_pre=1000,        # Reduce pre-NMS candidates
    max_per_img=100,     # Reduce post-NMS detections
    score_thr=0.05,      # Increase score threshold
    nms=dict(type='nms', iou_threshold=0.5)
)
```

---

## Advanced Features

### Custom Loss Functions

```python
# Custom focal loss configuration
loss_cls = dict(
    type='FocalLoss',
    use_sigmoid=True,
    gamma=2.0,
    alpha=0.25,
    loss_weight=1.0
)

# Custom IoU loss
loss_bbox = dict(
    type='IoULoss',
    mode='linear',
    eps=1e-6,
    reduction='mean',
    loss_weight=2.0
)
```

### Data Augmentation

```python
# Advanced augmentation pipeline
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=[(1333, 480), (1333, 800)], keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Mixup', alpha=0.2, beta=0.2),  # Advanced augmentation
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
```

### Multi-Scale Training

```python
# Multi-scale training configuration
img_scale = [(1333, 480), (1333, 512), (1333, 544), (1333, 576), 
             (1333, 608), (1333, 640), (1333, 672), (1333, 704), 
             (1333, 736), (1333, 768), (1333, 800)]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=img_scale, multiscale_mode='value', keep_ratio=True),
    # ... rest of pipeline
]
```

---

This documentation covers the comprehensive detection API for RepViT models. For specific implementation details and advanced configurations, refer to the MMDetection documentation and the provided example configs.