---
layout: post
title: Precision Weeding in Sugarbeets - End-to-End Real-Time Computer Vision System
date: 2024-12-25 21:00:00
description: Spot Spraying Project
tags: spotspraying, precision agriculture
categories: ai
thumbnail: assets/img/spot_spraying.png
---
## Introduction

Conventional spraying methods apply herbicides uniformly across fields, resulting in excessive chemical use, environmental risks, and increased operational costs. This project implements a precision weeding system for sugarbeet fields by combining semantic segmentation, depth sensing, and advanced model optimization techniques. The system targets only weed-infested areas, reducing herbicide usage and minimizing environmental impact.

## System Overview

The system pipeline is summarized in the following flowchart:

```
RGB + Depth Images
       │
       ▼
Semantic Segmentation 
(DeepLabV3+ with Channel Attention)
       │
       ▼
Depth Projection & 3D Processing
       │
       ▼
Field Actuation (Coordinate Transformation & Control)
```

The process begins with the acquisition of synchronized RGB and depth images. It then uses semantic segmentation to distinguish sugarbeets from weeds, followed by processing to obtain 3D information. Finally, the system translates these 3D coordinates to guide precise actuation in the field.

## Semantic Segmentation with Channel Attention

### Overview

The segmentation network is built on a modified DeepLabV3+ architecture that classifies each pixel as sugarbeet, weed, or background. A channel attention module enhances the network's ability to differentiate between crops and weeds, even under challenging conditions such as variable illumination, occlusions, or subtle texture differences.

### Benefits of Channel Attention

- **Adaptive Feature Recalibration:**  
  Global average pooling generates a channel descriptor that is then processed to produce channel-specific weights. This allows the network to emphasize important features while suppressing less informative ones.

- **Enhanced Discrimination:**  
  Recalibrated feature maps improve the network’s ability to distinguish between visually similar classes, leading to a  
  $$\textbf{19\% increase in Intersection over Union (IoU)}$$.

- **Robustness to Variability:**  
  Dynamic adjustment of feature importance helps maintain consistent performance despite changes in weather, soil, or crop growth stages.

### Mathematical Formulation

The channel attention module computes a channel descriptor via global average pooling:

$$
z_c = \frac{1}{H \times W} \sum_{i=1}^{H} \sum_{j=1}^{W} x_c(i,j),
$$

which is transformed into channel-specific weights using:

$$
s = \sigma(W_2 \cdot \text{ReLU}(W_1 \cdot z)),
$$

and applied to rescale the feature maps:

$$
\tilde{x}_c = s_c \cdot x_c.
$$

### PyTorch Implementation

```python
import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y
```

## Real-Time Deployment and Model Optimization

### Embedded System Architecture

Due to remote field conditions and limited internet connectivity, the entire inference pipeline is implemented in C++ and deployed on an NXP i.MX8 board running Yocto Linux with a Hailo-8 accelerator. **libtorch** is used to integrate the PyTorch models into the C++ environment, ensuring seamless execution in the field.

### Model Optimization Techniques

- **INT8 Quantization:**  
  The model is optimized for real-time performance using INT8 precision. This involves:
  
  - **Dynamic Quantization:**

    ```python
    import torch.quantization as quant

    model_fp32 = CustomSegNet(num_classes=3)
    model_int8 = quant.quantize_dynamic(model_fp32, {nn.Conv2d, nn.Linear}, dtype=torch.qint8)
    ```

  - **Quantization Aware Training (QAT):**  
    During QAT, fake quantization layers simulate INT8 precision. The activation quantization function is defined as:

    $$
    \hat{x} = Q(x) = s \cdot \text{clip}\left(\left\lfloor \frac{x}{s} \right\rceil, -q_{\min}, q_{\max}\right),
    $$

    where $$s$$ is the scale factor and $$\lfloor \cdot \rceil$$ denotes rounding. The gradient is approximated via:

    $$
    \frac{\partial L}{\partial x} \approx \frac{\partial L}{\partial \hat{x}}.
    $$

    The optimal scale $$s^*$$ minimizes the Kullback-Leibler divergence between the full-precision gradient distribution $$P(g)$$ and the quantized distribution $$Q(g; s)$$:

    $$
    s^* = \arg\min_{s} \, \mathrm{KL}(P(g) \parallel Q(g; s)) = \arg\min_{s} \sum_{i} P(g_i) \log \frac{P(g_i)}{Q(g_i; s)}.
    $$

- **Model Freezing and Tracing:**  
  The optimized model is frozen and traced using **libtorch**, resulting in a static computation graph that is exported to the ONNX format:

    ```python
    import torch

    model.eval()  # Freeze layers like batch normalization and dropout
    example_input = torch.randn(1, 3, 512, 512)
    traced_model = torch.jit.trace(model, example_input)
    traced_model.save("model_traced.pt")

    torch.onnx.export(
        model, 
        example_input, 
        "model_int8.onnx", 
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output']
    )
    ```

- **Integration with Hailo AI:**  
  The traced ONNX model is integrated into the C++ inference pipeline and executed using the Hailo AI inference server, providing optimized INT8 performance for real-time processing.

## Field Actuation and 3D Processing

After real-time inference, the system must translate the segmentation output into actionable data for field actuation. This section integrates depth projection, 3D localization, and coordinate transformation.

### Depth Projection and 3D Localization

The 2D segmentation mask is projected into 3D space using the intrinsic camera matrix $$K$$:

$$
\begin{bmatrix}
X \\
Y \\
Z
\end{bmatrix} = d(u,v) \cdot K^{-1} 
\begin{bmatrix}
u \\
v \\
1
\end{bmatrix},
$$

where $$d(u,v)$$ represents the depth at pixel $$(u,v)$$. Clustering the resulting 3D points enables the estimation of weed density and the computation of average spatial coordinates for each weed cluster.

### Coordinate Transformation for Field Actuation

To accurately guide the mechanical weeding nib, the computed 3D coordinates (initially defined in the camera frame) must be transformed into the machine’s coordinate system. This alignment is performed using a pre-computed transformation matrix.

#### C++ Implementation

```cpp
#include <Eigen/Dense>

Eigen::Matrix4f getTransformMatrix();  // Retrieves the transformation matrix

Eigen::Vector4f getPlantCoordinates(float x, float y, float z) {
    Eigen::Matrix4f camToActuator = getTransformMatrix();
    Eigen::Vector4f plantCoordCam(x, y, z, 1);
    return camToActuator * plantCoordCam;
}
```

This transformation ensures that the machine’s location in the field is accurately adjusted for effective actuation.

## Weedicide Usage Calculation and Motivation

### Why Precision Spot Spraying?

Conventional systems apply herbicides uniformly, often wasting chemicals and affecting non-target crops. By precisely identifying weed-infested areas using real-time semantic segmentation and depth estimation, the system applies herbicides only where necessary, reducing chemical waste and environmental impact.

### Calculating Weedicide Requirements

1. **Segmentation Map $$S(u,v)$$:**

    $$
    S(u,v) = 
    \begin{cases}
    1, & \text{if pixel } (u,v) \text{ is classified as weed} \\
    0, & \text{otherwise}
    \end{cases}
    $$

2. **Real-World Area Calculation:**  
   The area corresponding to a pixel is approximated by:

    $$
    A_{pixel}(u,v) = \left(\frac{d(u,v)}{f}\right)^2,
    $$

    where $$d(u,v)$$ is the depth at pixel $$(u,v)$$ and $$f$$ is the focal length.

3. **Total Weed-Covered Area $$A_w$$:**

    $$
    A_w = \sum_{u,v} S(u,v) \cdot \left(\frac{d(u,v)}{f}\right)^2.
    $$

4. **Herbicide Amount:**  
   Given $$\beta$$ as the application rate (liters per square meter), the total herbicide required is:

    $$
    \text{Weedicide Amount} = \beta \cdot A_w.
    $$

This calculation guarantees that herbicide is applied only where needed.

## Key Results

- **Improved Segmentation Accuracy:**  
  Integration of the channel attention module resulted in a  
  $$\textbf{19\% increase in IoU}$$.

- **Enhanced Processing Speed:**  
  Optimizations increased the frame rate from **1.3 fps to 32 fps** on the NVIDIA Xavier AGX platform.

- **Real-Time Deployment:**  
  The embedded system on an NXP board achieved **23 fps** with INT8 quantization.


## Conclusion

This project demonstrates a comprehensive approach to precision weeding in sugarbeet fields. By combining advanced semantic segmentation with robust model optimizations and efficient embedded deployment, the system offers a scalable solution for targeted weed treatment. The integrated field actuation module—comprising depth projection, 3D localization, and coordinate transformation—ensures that the machine's location is accurately adjusted for precise herbicide application. This method not only reduces herbicide waste and environmental impact but also offers significant cost savings and improved operational efficiency—key benefits for modern precision agriculture.