# RepViT and RepViT-SAM API Documentation

This comprehensive documentation covers all public APIs, functions, and components of the RepViT and RepViT-SAM project, providing real-time segmentation and efficient mobile computer vision capabilities.

## Table of Contents

1. [Main Classification API](#main-classification-api)
2. [RepViT Model Architecture](#repvit-model-architecture)
3. [RepViT-SAM API](#repvit-sam-api)
4. [Training and Evaluation](#training-and-evaluation)
5. [Utility Functions](#utility-functions)
6. [Data Processing](#data-processing)
7. [Loss Functions](#loss-functions)
8. [Performance Utilities](#performance-utilities)
9. [Model Export](#model-export)

---

## Main Classification API

### Main Training Script (`main.py`)

#### `get_args_parser()`
Creates and returns an argument parser for training and evaluation configuration.

**Returns:**
- `ArgumentParser`: Configured argument parser with all training options

**Usage:**
```python
parser = get_args_parser()
args = parser.parse_args()
```

#### `main(args)`
Main training and evaluation function that orchestrates the entire pipeline.

**Parameters:**
- `args`: Command line arguments containing configuration

**Key Arguments:**
- `--model`: Model name (e.g., 'repvit_m0_9', 'repvit_m1_1', 'repvit_m2_3')
- `--batch-size`: Batch size for training (default: 256)
- `--epochs`: Number of training epochs (default: 300)
- `--data-path`: Path to ImageNet dataset
- `--eval`: Enable evaluation mode only
- `--resume`: Path to checkpoint for resuming training

**Usage:**
```python
# Training
python main.py --model repvit_m1_1 --data-path ~/imagenet --batch-size 256

# Evaluation
python main.py --eval --model repvit_m1_1 --resume pretrain/repvit_m1_1_distill_300e.pth --data-path ~/imagenet
```

#### `export_onnx(model, output_dir)`
Exports a trained model to ONNX format for deployment.

**Parameters:**
- `model`: PyTorch model to export
- `output_dir`: Directory to save the ONNX model

---

## RepViT Model Architecture

### Core Components (`model/repvit.py`)

#### `RepViT(nn.Module)`
Main RepViT model class implementing the efficient mobile CNN architecture.

**Constructor Parameters:**
- `cfgs`: Configuration list defining layer architecture
- `num_classes`: Number of output classes (default: 1000)
- `distillation`: Enable distillation mode (default: False)

**Methods:**
- `forward(x)`: Forward pass through the network

**Usage:**
```python
from timm.models import create_model
import model

# Create RepViT model
model = create_model('repvit_m1_1')

# For inference, replace batchnorm for better performance
import utils
utils.replace_batchnorm(model)
```

#### Model Variants

##### `repvit_m0_6(pretrained=False, num_classes=1000, distillation=False)`
Smallest RepViT model (~0.6ms latency).
- **Parameters:** 5.1M
- **Top-1 Accuracy:** 74.1%
- **Latency:** ~0.6ms on iPhone 12

##### `repvit_m0_9(pretrained=False, num_classes=1000, distillation=False)`
Ultra-fast RepViT model.
- **Parameters:** 5.1M 
- **Top-1 Accuracy:** 78.7% (300e) / 79.1% (450e)
- **Latency:** 0.9ms on iPhone 12

##### `repvit_m1_0(pretrained=False, num_classes=1000, distillation=False)`
Balanced RepViT model.
- **Parameters:** 6.8M
- **Top-1 Accuracy:** 80.0% (300e) / 80.3% (450e)
- **Latency:** 1.0ms on iPhone 12

##### `repvit_m1_1(pretrained=False, num_classes=1000, distillation=False)`
Recommended RepViT model.
- **Parameters:** 8.2M
- **Top-1 Accuracy:** 80.7% (300e) / 81.2% (450e)
- **Latency:** 1.1ms on iPhone 12

##### `repvit_m1_5(pretrained=False, num_classes=1000, distillation=False)`
High-accuracy RepViT model.
- **Parameters:** 14.0M
- **Top-1 Accuracy:** 82.3% (300e) / 82.5% (450e)
- **Latency:** 1.5ms on iPhone 12

##### `repvit_m2_3(pretrained=False, num_classes=1000, distillation=False)`
Largest RepViT model.
- **Parameters:** 22.9M
- **Top-1 Accuracy:** 83.3% (300e) / 83.7% (450e)
- **Latency:** 2.3ms on iPhone 12

**Usage:**
```python
# Create specific model variants
model = repvit_m1_1(pretrained=True)
model = repvit_m2_3(num_classes=100)  # Custom number of classes
```

#### Building Blocks

##### `RepViTBlock(nn.Module)`
Core RepViT block combining efficient convolutions and attention mechanisms.

**Constructor Parameters:**
- `inp`: Input channels
- `hidden_dim`: Hidden dimension
- `oup`: Output channels
- `kernel_size`: Kernel size
- `stride`: Stride
- `use_se`: Use Squeeze-and-Excitation
- `use_hs`: Use hard swish activation

##### `Conv2d_BN(torch.nn.Sequential)`
Fused convolution and batch normalization layer.

**Constructor Parameters:**
- `a`: Input channels
- `b`: Output channels
- `ks`: Kernel size (default: 1)
- `stride`: Stride (default: 1)
- `pad`: Padding (default: 0)

**Methods:**
- `fuse()`: Fuse conv and BN for inference

##### `Residual(torch.nn.Module)`
Residual connection with optional dropout.

**Constructor Parameters:**
- `m`: Module to wrap
- `drop`: Dropout probability (default: 0.0)

---

## RepViT-SAM API

### Main SAM Components (`sam/repvit_sam/`)

#### Model Building (`build_sam.py`)

##### `build_sam_repvit(checkpoint=None)`
Builds RepViT-SAM model for real-time segmentation.

**Parameters:**
- `checkpoint`: Path to pretrained checkpoint (optional)

**Returns:**
- `Sam`: RepViT-SAM model ready for inference

**Usage:**
```python
from repvit_sam import build_sam_repvit

# Build model
sam = build_sam_repvit(checkpoint="path/to/checkpoint.pth")
```

##### Other SAM Variants
- `build_sam_vit_h(checkpoint=None)`: SAM with ViT-H backbone
- `build_sam_vit_l(checkpoint=None)`: SAM with ViT-L backbone
- `build_sam_vit_b(checkpoint=None)`: SAM with ViT-B backbone
- `build_sam_vit_t(checkpoint=None)`: Mobile SAM with TinyViT

#### `Sam(nn.Module)`
Main SAM model class.

**Constructor Parameters:**
- `image_encoder`: Image encoder (RepViT, ViT, or TinyViT)
- `prompt_encoder`: Prompt encoder for various input types
- `mask_decoder`: Mask decoder for generating segmentation masks
- `pixel_mean`: Pixel normalization mean values
- `pixel_std`: Pixel normalization std values

**Methods:**
- `forward(batched_input, multimask_output)`: End-to-end mask prediction

#### `SamPredictor(torch.nn.Module)`
Interactive predictor for efficient mask generation with prompts.

**Constructor Parameters:**
- `sam_model`: SAM model for prediction

**Key Methods:**

##### `set_image(image, image_format="RGB")`
Sets the image for mask prediction.

**Parameters:**
- `image`: Input image as numpy array (HWC uint8)
- `image_format`: Color format ("RGB" or "BGR")

##### `predict(point_coords=None, point_labels=None, box=None, mask_input=None, multimask_output=True, return_logits=False)`
Predicts masks given prompts.

**Parameters:**
- `point_coords`: Point coordinates as numpy array
- `point_labels`: Point labels (1 for foreground, 0 for background)
- `box`: Bounding box as [x1, y1, x2, y2]
- `mask_input`: Previous mask input for refinement
- `multimask_output`: Return multiple mask candidates
- `return_logits`: Return raw logits instead of binary masks

**Returns:**
- `masks`: Predicted masks
- `scores`: Confidence scores
- `logits`: Raw prediction logits (if requested)

**Usage:**
```python
from repvit_sam import SamPredictor, build_sam_repvit
import numpy as np

# Initialize predictor
sam = build_sam_repvit(checkpoint="checkpoint.pth")
predictor = SamPredictor(sam)

# Set image
predictor.set_image(image)

# Predict with point prompt
input_point = np.array([[500, 375]])
input_label = np.array([1])
masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,
)

# Predict with box prompt
input_box = np.array([425, 600, 700, 875])
masks, scores, logits = predictor.predict(
    box=input_box,
    multimask_output=False,
)
```

#### `SamAutomaticMaskGenerator`
Generates masks for entire images without prompts.

**Constructor Parameters:**
- `model`: SAM model
- `points_per_side`: Grid density for point sampling (default: 32)
- `pred_iou_thresh`: IoU threshold for filtering (default: 0.88)
- `stability_score_thresh`: Stability threshold (default: 0.95)
- `box_nms_thresh`: NMS threshold for boxes (default: 0.7)
- `crop_n_layers`: Number of crop layers (default: 0)
- `min_mask_region_area`: Minimum mask area (default: 0)
- `output_mode`: Output format ("binary_mask", "uncompressed_rle", "coco_rle")

**Methods:**

##### `generate(image)`
Generates masks for the entire image.

**Parameters:**
- `image`: Input image as numpy array

**Returns:**
- List of mask dictionaries with keys: 'segmentation', 'area', 'bbox', 'predicted_iou', 'point_coords', 'stability_score', 'crop_box'

**Usage:**
```python
from repvit_sam import SamAutomaticMaskGenerator, build_sam_repvit

# Initialize automatic mask generator
sam = build_sam_repvit(checkpoint="checkpoint.pth")
mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,
    pred_iou_thresh=0.86,
    stability_score_thresh=0.92,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=100,
)

# Generate masks
masks = mask_generator.generate(image)
```

---

## Training and Evaluation

### Engine Functions (`engine.py`)

#### `train_one_epoch(model, criterion, data_loader, optimizer, device, epoch, loss_scaler, **kwargs)`
Trains the model for one epoch.

**Parameters:**
- `model`: Model to train
- `criterion`: Loss function
- `data_loader`: Training data loader
- `optimizer`: Optimizer
- `device`: Training device
- `epoch`: Current epoch number
- `loss_scaler`: Loss scaler for mixed precision

**Returns:**
- Training statistics dictionary

#### `evaluate(data_loader, model, device)`
Evaluates the model on validation set.

**Parameters:**
- `data_loader`: Validation data loader
- `model`: Model to evaluate
- `device`: Evaluation device

**Returns:**
- Evaluation statistics including top-1 and top-5 accuracy

**Usage:**
```python
from engine import train_one_epoch, evaluate

# Training loop
for epoch in range(args.epochs):
    train_stats = train_one_epoch(
        model, criterion, data_loader_train, 
        optimizer, device, epoch, loss_scaler
    )
    
    test_stats = evaluate(data_loader_val, model, device)
```

---

## Utility Functions

### Core Utilities (`utils.py`)

#### `SmoothedValue(object)`
Tracks and smooths a series of values.

**Constructor Parameters:**
- `window_size`: Size of smoothing window (default: 20)
- `fmt`: Format string for printing

**Methods:**
- `update(value, n=1)`: Add new value
- `avg`: Get average over window
- `global_avg`: Get global average
- `median`: Get median value

#### `MetricLogger(object)`
Logs and tracks multiple metrics during training.

**Constructor Parameters:**
- `delimiter`: String delimiter for printing (default: "\t")

**Methods:**
- `update(**kwargs)`: Update multiple meters
- `add_meter(name, meter)`: Add new meter
- `log_every(iterable, print_freq, header)`: Log every N iterations

#### `replace_batchnorm(net)`
Replaces batch normalization with identity for inference optimization.

**Parameters:**
- `net`: Network to modify

**Usage:**
```python
# Optimize model for inference
utils.replace_batchnorm(model)
```

#### Distributed Training Utilities

##### `init_distributed_mode(args)`
Initializes distributed training.

##### `is_main_process()`
Returns True if current process is main process.

##### `save_on_master(*args, **kwargs)`
Save checkpoint only on master process.

---

## Data Processing

### Data Loading (`data/datasets.py`)

#### `build_dataset(is_train, args)`
Builds dataset for training or validation.

**Parameters:**
- `is_train`: Whether to build training dataset
- `args`: Arguments containing dataset configuration

**Returns:**
- Dataset object

#### `build_transform(is_train, args)`
Builds data transformations.

**Parameters:**
- `is_train`: Whether to build training transforms
- `args`: Arguments containing transform configuration

**Returns:**
- Transform composition

### Data Augmentation (`data/threeaugment.py`)

#### `new_data_aug_generator(args=None)`
Creates advanced data augmentation pipeline.

**Returns:**
- Data augmentation transform

**Available Augmentations:**
- `GaussianBlur`: Gaussian blur augmentation
- `Solarization`: Solarization augmentation
- `gray_scale`: Grayscale conversion
- `horizontal_flip`: Horizontal flipping

### Sampling (`data/samplers.py`)

#### `RASampler(torch.utils.data.Sampler)`
Repeated augmentation sampler for improved training.

**Constructor Parameters:**
- `dataset`: Dataset to sample from
- `num_replicas`: Number of distributed processes
- `rank`: Rank of current process
- `shuffle`: Whether to shuffle data

---

## Loss Functions

### `DistillationLoss(torch.nn.Module)` (`losses.py`)
Knowledge distillation loss for model training.

**Constructor Parameters:**
- `base_criterion`: Base loss function (e.g., CrossEntropyLoss)
- `teacher_model`: Teacher model for distillation
- `distillation_type`: Type of distillation ('soft', 'hard', 'none')
- `alpha`: Distillation loss weight
- `tau`: Temperature for soft distillation

**Methods:**
- `forward(inputs, outputs, labels)`: Compute distillation loss

**Usage:**
```python
from losses import DistillationLoss
import torch.nn as nn

# Create distillation loss
base_criterion = nn.CrossEntropyLoss()
distillation_loss = DistillationLoss(
    base_criterion=base_criterion,
    teacher_model=teacher_model,
    distillation_type='soft',
    alpha=0.5,
    tau=3.0
)

# Use in training
loss = distillation_loss(inputs, outputs, targets)
```

---

## Performance Utilities

### GPU Throughput Testing (`speed_gpu.py`)

#### `throughput(name, model, device, batch_size, resolution=224)`
Measures model throughput on GPU.

**Parameters:**
- `name`: Model name for logging
- `model`: Model to benchmark
- `device`: Device for testing
- `batch_size`: Batch size for testing
- `resolution`: Input resolution

**Usage:**
```bash
# Measure RepViT-M1.1 throughput
python speed_gpu.py --model repvit_m1_1 --batch-size 2048 --resolution 224
```

### FLOP Calculation (`flops.py`)
Calculates model computational complexity.

**Usage:**
```python
from fvcore.nn import FlopCountAnalysis

# Calculate FLOPs
inputs = torch.randn(1, 3, 224, 224)
flops = FlopCountAnalysis(model, inputs)
print(f"FLOPs: {flops.total() / 1e9:.2f}G")
```

---

## Model Export

### Core ML Export (`export_coreml.py`)

#### `export_coreml(model, output_dir)`
Exports model to Core ML format for iOS deployment.

**Usage:**
```bash
# Export RepViT to Core ML
python export_coreml.py --model repvit_m1_1 --ckpt pretrain/repvit_m1_1_distill_300e.pth
```

---

## Application Examples

### Gradio Web Application (`sam/app/app.py`)

The SAM application provides an interactive web interface for segmentation.

**Key Functions:**
- `segment_with_points()`: Segment using point prompts
- `get_points_with_draw()`: Interactive point selection
- `clear()`: Clear current session

**Usage:**
```bash
# Launch Gradio app
cd sam/app
python app.py
```

### Utility Tools (`sam/app/utils/tools.py`)

#### `fast_process(annotations, image, device, scale, better_quality=False)`
Fast mask processing and visualization.

#### `fast_show_mask(annotation, ax, bbox_color, text_color, **kwargs)`
Efficient mask visualization.

---

## Advanced Usage Examples

### 1. Training a Custom RepViT Model

```python
import torch
from timm.models import create_model
import model
from engine import train_one_epoch, evaluate
from losses import DistillationLoss

# Create model
model = create_model('repvit_m1_1', num_classes=100)  # Custom classes

# Setup training
criterion = DistillationLoss(...)
optimizer = torch.optim.AdamW(model.parameters())

# Training loop
for epoch in range(epochs):
    train_stats = train_one_epoch(model, criterion, train_loader, optimizer, device, epoch, scaler)
    test_stats = evaluate(val_loader, model, device)
```

### 2. Custom SAM Pipeline

```python
from repvit_sam import build_sam_repvit, SamPredictor
import numpy as np

# Build custom SAM
sam = build_sam_repvit()
predictor = SamPredictor(sam)

# Process multiple images
for image in image_list:
    predictor.set_image(image)
    
    # Multiple prompts
    masks = []
    for point, label in zip(points, labels):
        mask, score, logit = predictor.predict(
            point_coords=point.reshape(1, -1),
            point_labels=label.reshape(-1),
            multimask_output=False
        )
        masks.append(mask)
```

### 3. Model Optimization for Deployment

```python
import torch
from timm.models import create_model
import utils

# Load and optimize model
model = create_model('repvit_m1_1')
model.load_state_dict(torch.load('checkpoint.pth'))

# Optimize for inference
utils.replace_batchnorm(model)
model.eval()

# Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    "repvit_m1_1.onnx",
    opset_version=11,
    input_names=['input'],
    output_names=['output']
)
```

---

## Error Handling and Best Practices

### Common Issues and Solutions

1. **Memory Issues**: Use smaller batch sizes and gradient accumulation
2. **Convergence Problems**: Adjust learning rate and warmup schedule
3. **Performance Optimization**: Use `replace_batchnorm()` for inference

### Recommended Configurations

#### For Training:
- Batch size: 256-1024 (depending on GPU memory)
- Learning rate: 1e-3 with cosine annealing
- Epochs: 300 for best accuracy, 450 for marginal improvements

#### For Inference:
- Always call `model.eval()`
- Use `replace_batchnorm()` for mobile deployment
- Consider mixed precision with `torch.cuda.amp`

---

This documentation provides comprehensive coverage of all public APIs and components in the RepViT and RepViT-SAM project. For additional examples and implementation details, refer to the training scripts and example applications in the respective directories.