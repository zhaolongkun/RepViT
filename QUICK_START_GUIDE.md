# RepViT and RepViT-SAM Quick Start Guide

This guide provides step-by-step instructions to quickly get started with RepViT for classification, segmentation, detection, and the RepViT-SAM for real-time segmentation.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start - Classification](#quick-start---classification)
3. [Quick Start - RepViT-SAM](#quick-start---repvit-sam)
4. [Quick Start - Segmentation](#quick-start---segmentation)
5. [Quick Start - Detection](#quick-start---detection)
6. [Model Zoo](#model-zoo)
7. [Performance Benchmarks](#performance-benchmarks)
8. [Common Issues](#common-issues)

---

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 1.9+
- CUDA 11.0+ (for GPU support)

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/THU-MIG/RepViT.git
cd RepViT

# Create conda environment
conda create -n repvit python=3.8
conda activate repvit

# Install dependencies
pip install -r requirements.txt

# Install timm (for model creation)
pip install timm
```

### Extended Installation (for SAM, Detection, Segmentation)

```bash
# For RepViT-SAM
cd sam
pip install -r requirements.txt

# For Detection (MMDetection)
cd ../detection
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html
pip install mmdet

# For Segmentation (MMSegmentation)
cd ../segmentation
pip install mmsegmentation
```

---

## Quick Start - Classification

### 1. Load Pretrained Model

```python
from timm.models import create_model
import torch
import utils

# Create RepViT model
model = create_model('repvit_m1_1', pretrained=True)

# Optimize for inference
utils.replace_batchnorm(model)
model.eval()

print(f"Model created: {model}")
```

### 2. Image Classification

```python
import torch
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

# Load and preprocess image
config = resolve_data_config({}, model=model)
transform = create_transform(**config)

img = Image.open('path/to/your/image.jpg')
tensor = transform(img).unsqueeze(0)

# Inference
with torch.no_grad():
    output = model(tensor)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    
# Get top 5 predictions
top5_prob, top5_catid = torch.topk(probabilities, 5)
for i in range(top5_prob.size(0)):
    print(f"Class {top5_catid[i]}: {top5_prob[i].item():.4f}")
```

### 3. Training Custom Model

```bash
# Download ImageNet dataset first
# Then train RepViT-M1.1
python main.py --model repvit_m1_1 --data-path /path/to/imagenet --batch-size 256

# For evaluation only
python main.py --eval --model repvit_m1_1 --resume /path/to/checkpoint.pth --data-path /path/to/imagenet
```

### 4. Export for Mobile Deployment

```bash
# Export to Core ML (iOS)
python export_coreml.py --model repvit_m1_1 --ckpt /path/to/checkpoint.pth

# Export to ONNX
python -c "
import torch
from timm.models import create_model
import utils

model = create_model('repvit_m1_1', pretrained=True)
utils.replace_batchnorm(model)
model.eval()

dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy_input, 'repvit_m1_1.onnx')
"
```

---

## Quick Start - RepViT-SAM

### 1. Basic Setup

```python
import sys
sys.path.append('sam')

from repvit_sam import SamPredictor, build_sam_repvit
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Build RepViT-SAM model
sam = build_sam_repvit(checkpoint="path/to/repvit_sam_checkpoint.pth")
predictor = SamPredictor(sam)
```

### 2. Interactive Point-based Segmentation

```python
# Load image
image = np.array(Image.open("path/to/image.jpg"))
predictor.set_image(image)

# Define point prompt (x, y coordinates)
input_point = np.array([[500, 375]])  # foreground point
input_label = np.array([1])  # 1 for foreground, 0 for background

# Predict masks
masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,
)

# Visualize results
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

# Display results
for i, (mask, score) in enumerate(zip(masks, scores)):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    show_mask(mask, plt.gca())
    plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
    plt.axis('off')
    plt.show()
```

### 3. Box-based Segmentation

```python
# Define bounding box [x1, y1, x2, y2]
input_box = np.array([425, 600, 700, 875])

masks, scores, logits = predictor.predict(
    box=input_box,
    multimask_output=False,
)

# Display result
plt.figure(figsize=(10, 10))
plt.imshow(image)
show_mask(masks[0], plt.gca())
plt.title(f"Box Segmentation, Score: {scores[0]:.3f}", fontsize=18)
plt.axis('off')
plt.show()
```

### 4. Automatic Mask Generation

```python
from repvit_sam import SamAutomaticMaskGenerator

# Create automatic mask generator
mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,
    pred_iou_thresh=0.86,
    stability_score_thresh=0.92,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=100,
)

# Generate masks for entire image
masks = mask_generator.generate(image)

print(f"Generated {len(masks)} masks")

# Visualize all masks
def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], 
                   sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

plt.figure(figsize=(20, 20))
plt.imshow(image)
show_anns(masks)
plt.axis('off')
plt.show()
```

### 5. Web Application

```bash
# Launch Gradio web interface
cd sam/app
python app.py

# Open browser and go to http://localhost:7860
```

---

## Quick Start - Segmentation

### 1. Setup MMSegmentation Environment

```bash
cd segmentation

# Install dependencies
pip install mmsegmentation
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html
```

### 2. Training on ADE20K

```bash
# Download ADE20K dataset
# Then train RepViT-M1.1 with UPerNet
python tools/train.py configs/repvit/repvit_m1_1_upernet_512x512_ade20k.py

# Multi-GPU training
bash tools/dist_train.sh configs/repvit/repvit_m1_1_upernet_512x512_ade20k.py 8
```

### 3. Inference on Custom Images

```python
from mmseg.apis import inference_segmentor, init_segmentor
import mmcv

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

### 4. Evaluation

```bash
# Test on ADE20K validation set
python tools/test.py configs/repvit/repvit_m1_1_upernet_512x512_ade20k.py \
    checkpoints/repvit_m1_1_upernet_ade20k.pth \
    --eval mIoU
```

---

## Quick Start - Detection

### 1. Setup MMDetection Environment

```bash
cd detection

# Install dependencies
pip install mmdet
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html
```

### 2. Training on COCO

```bash
# Download COCO dataset
# Then train RepViT-M1.1 with RetinaNet
python tools/train.py configs/repvit/retinanet_repvit_m1_1_fpn_1x_coco.py

# Multi-GPU training
bash tools/dist_train.sh configs/repvit/retinanet_repvit_m1_1_fpn_1x_coco.py 8
```

### 3. Inference on Custom Images

```python
from mmdet.apis import inference_detector, init_detector
import mmcv

# Initialize model
config_file = 'configs/repvit/retinanet_repvit_m1_1_fpn_1x_coco.py'
checkpoint_file = 'checkpoints/retinanet_repvit_m1_1_fpn_1x_coco.pth'
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# Inference
img = 'demo/demo.jpg'
result = inference_detector(model, img)

# Visualize result
model.show_result(img, result, out_file='result.jpg', score_thr=0.3)
```

### 4. Evaluation

```bash
# Test on COCO validation set
python tools/test.py configs/repvit/retinanet_repvit_m1_1_fpn_1x_coco.py \
    checkpoints/retinanet_repvit_m1_1_fpn_1x_coco.pth \
    --eval bbox
```

---

## Model Zoo

### Classification Models (ImageNet-1K)

| Model | Top-1 Acc | Params | MACs | Latency* | Download |
|-------|-----------|--------|------|----------|----------|
| RepViT-M0.6 | 74.1% | 5.1M | 0.6G | 0.6ms | [model](https://github.com/THU-MIG/RepViT/releases/download/v1.0/repvit_m0_6_distill_300e.pth) |
| RepViT-M0.9 | 78.7% | 5.1M | 0.8G | 0.9ms | [model](https://github.com/THU-MIG/RepViT/releases/download/v1.0/repvit_m0_9_distill_300e.pth) |
| RepViT-M1.0 | 80.0% | 6.8M | 1.1G | 1.0ms | [model](https://github.com/THU-MIG/RepViT/releases/download/v1.0/repvit_m1_0_distill_300e.pth) |
| RepViT-M1.1 | 80.7% | 8.2M | 1.3G | 1.1ms | [model](https://github.com/THU-MIG/RepViT/releases/download/v1.0/repvit_m1_1_distill_300e.pth) |
| RepViT-M1.5 | 82.3% | 14.0M | 2.3G | 1.5ms | [model](https://github.com/THU-MIG/RepViT/releases/download/v1.0/repvit_m1_5_distill_300e.pth) |
| RepViT-M2.3 | 83.3% | 22.9M | 4.5G | 2.3ms | [model](https://github.com/THU-MIG/RepViT/releases/download/v1.0/repvit_m2_3_distill_300e.pth) |

*Latency measured on iPhone 12 with Core ML

### RepViT-SAM Models

| Model | Backbone | Size | Performance | Download |
|-------|----------|------|-------------|----------|
| RepViT-SAM | RepViT-M0.9 | 6.8M | 10x faster than MobileSAM | [model](link) |
| RepViT-SAM | RepViT-M1.1 | 9.4M | Best accuracy/speed trade-off | [model](link) |

### Segmentation Models (ADE20K)

| Model | Backbone | mIoU | Params | Download |
|-------|----------|------|--------|----------|
| UPerNet | RepViT-M0.9 | 40.5% | 14.2M | [model](link) |
| UPerNet | RepViT-M1.1 | 42.9% | 17.3M | [model](link) |
| UPerNet | RepViT-M1.5 | 45.2% | 23.1M | [model](link) |

### Detection Models (COCO)

| Model | Backbone | AP | AP50 | AP75 | Download |
|-------|----------|----|----- |------|----------|
| RetinaNet | RepViT-M0.9 | 37.4 | 57.9 | 39.8 | [model](link) |
| RetinaNet | RepViT-M1.1 | 39.1 | 59.7 | 42.1 | [model](link) |
| RetinaNet | RepViT-M1.5 | 41.2 | 62.1 | 44.5 | [model](link) |

---

## Performance Benchmarks

### Speed Measurement

```bash
# Measure classification model speed
python speed_gpu.py --model repvit_m1_1 --batch-size 2048

# Measure FLOPs
python flops.py
```

### Memory Usage

```python
import torch
from timm.models import create_model

model = create_model('repvit_m1_1')
dummy_input = torch.randn(1, 3, 224, 224)

# Measure model parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params / 1e6:.2f}M")

# Measure inference memory
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
with torch.no_grad():
    output = model(dummy_input.cuda())
peak_memory = torch.cuda.max_memory_allocated() / 1024**2
print(f"Peak memory usage: {peak_memory:.2f}MB")
```

---

## Common Issues

### 1. Installation Issues

```bash
# If CUDA version mismatch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# If mmcv installation fails
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu118/torch1.13.0/index.html
```

### 2. Memory Issues

```python
# Reduce batch size for training
python main.py --model repvit_m1_1 --batch-size 128  # instead of 256

# Use gradient accumulation
python main.py --model repvit_m1_1 --batch-size 64 --accumulation-steps 4
```

### 3. Performance Optimization

```python
# For inference optimization
import utils
model = create_model('repvit_m1_1', pretrained=True)
utils.replace_batchnorm(model)  # Replace BN with Identity
model.eval()

# Use mixed precision
with torch.cuda.amp.autocast():
    output = model(input_tensor)
```

### 4. Model Loading Issues

```python
# If checkpoint loading fails
checkpoint = torch.load('model.pth', map_location='cpu')
model.load_state_dict(checkpoint['model'])  # Try different keys

# For distributed training checkpoints
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in checkpoint['model'].items():
    name = k[7:] if k.startswith('module.') else k  # remove 'module.' prefix
    new_state_dict[name] = v
model.load_state_dict(new_state_dict)
```

---

## Next Steps

1. **Explore Advanced Features**: Check out the full API documentation for advanced usage
2. **Custom Training**: Adapt models for your specific datasets and tasks
3. **Mobile Deployment**: Export models for iOS/Android deployment
4. **Research Applications**: Use RepViT as backbone for your research projects

For more detailed information, refer to:
- [API Documentation](API_DOCUMENTATION.md)
- [Segmentation API](SEGMENTATION_API.md) 
- [Detection API](DETECTION_API.md)
- [Original Paper](https://arxiv.org/abs/2307.09283)
- [RepViT-SAM Paper](https://arxiv.org/abs/2312.05760)

---

**Happy coding with RepViT! 🚀**