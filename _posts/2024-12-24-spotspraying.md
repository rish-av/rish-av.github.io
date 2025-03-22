---
layout: post
title: Precision Weeding in Sugarbeets - A Scientific Exploration of a Real-Time Computer Vision System
date: 2024-12-25 21:00:00
description: Spot Spraying Project
tags: spotspraying, precision agriculture
categories: ai
thumbnail: assets/img/spot_spraying.png
---

**Figure 1: System Flowchart**  
```
RGB + Depth Images
       │
       ▼
Semantic Segmentation 
(DeepLabV3+ with Channel Attention)
       │
       ▼
Depth Projection 
(using Intrinsic Matrix K⁻¹)
       │
       ▼
3D Localization 
(Clustering for Weed Density Estimation)
       │
       ▼
Actuation 
(Coordinate Transformation & Control)
```

This project was developed to address the need for targeted weed and crop treatment in sugarbeet fields. I designed the system to distinguish sugarbeets from weeds, estimate weed density using depth data, and compute precise 3D coordinates to control a mechanical weeding nib. Due to field deployment constraints—including the lack of reliable internet connectivity and strict cost limitations—the solution was fully embedded and optimized for efficiency.

The system began with the acquisition of synchronized RGB and depth images. The RGB images were processed using a semantic segmentation network based on a modified DeepLabV3+ architecture that classified each pixel as sugarbeet, weed, or background. To improve the network's ability to differentiate between crop growth stages and weed types, I integrated a channel attention module. This module refined the feature representation by computing a channel descriptor via global average pooling:

$$z_c = \frac{1}{H \times W} \sum_{i=1}^{H} \sum_{j=1}^{W} x_c(i,j)$$

where $x_c$ is the feature map for channel $c$. The descriptor was then processed through two fully connected layers—with a ReLU activation followed by a sigmoid function—to generate channel-specific weights:

$$s = \sigma(W_2 \cdot \text{ReLU}(W_1 \cdot z)),$$

which were used to rescale the original feature maps:

$$\tilde{x}_c = s_c \cdot x_c.$$

The following PyTorch code implemented the channel attention module:

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

Since the data from Hydac were limited and slow to acquire, I initially trained the base model on a larger University of Bonn sugarbeet dataset and later used transfer learning to fine-tune the model on the Hydac data.

After semantic segmentation, the depth data were used to project the 2D segmentation mask into 3D space using the intrinsic camera matrix $K$:

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

where $d(u,v)$ is the depth at pixel $(u,v)$. Clustering these 3D points yielded estimates of weed density and calculated the average spatial coordinates for each weed cluster, which were then used for actuation.

For the coordinate transformation from the camera frame to the actuator frame, I used a pre-computed transformation matrix. In C++, the transformation was implemented as follows:

```cpp
#include <Eigen/Dense>

Eigen::Matrix4f getTransformMatrix();  // Function that retrieves the transformation matrix

Eigen::Vector4f getPlantCoordinates(float x, float y, float z) {
    Eigen::Matrix4f camToActuator = getTransformMatrix();
    Eigen::Vector4f plantCoordCam(x, y, z, 1);
    return camToActuator * plantCoordCam;
}
```

Due to remote field conditions where online processing was not feasible, I implemented the entire inference pipeline in C++ and deployed it on an NXP i.MX8 board running Yocto Linux with a Hailo-8 accelerator. For model deployment, I used **libtorch** to integrate my PyTorch models into the C++ inference pipeline.

To achieve real-time performance, I optimized the deep learning model using INT8 precision. Initially, I applied dynamic quantization in PyTorch:

```python
import torch.quantization as quant

model_fp32 = CustomSegNet(num_classes=3)
model_int8 = quant.quantize_dynamic(model_fp32, {nn.Conv2d, nn.Linear}, dtype=torch.qint8)
```

In addition to dynamic quantization, I employed Quantization Aware Training (QAT) to further reduce quantization error. During QAT, fake quantization layers simulated INT8 precision during training. The quantization function for an activation \(x\) was defined as:

$$
\hat{x} = Q(x) = s \cdot \text{clip}\left(\left\lfloor \frac{x}{s} \right\rceil, -q_{\min}, q_{\max}\right),
$$

where $s$ is the scale factor and $\lfloor \cdot \rceil$ denotes rounding. The Straight-Through Estimator (STE) approximated the gradient:

$$
\frac{\partial L}{\partial x} \approx \frac{\partial L}{\partial \hat{x}}.
$$

A crucial part of QAT was calibrating the quantization parameters using the histogram of gradients $`H(g)`$, where $`g = \frac{\partial L}{\partial x}`$. By minimizing the Kullback-Leibler divergence between the full-precision gradient distribution $P(g)$ and the quantized distribution $Q(g; s)$:

$$
s^* = \arg\min_{s} \, \mathrm{KL}(P(g) \parallel Q(g; s)) = \arg\min_{s} \sum_{i} P(g_i) \log \frac{P(g_i)}{Q(g_i; s)},
$$

I determined the optimal scale $s^*$ that minimized quantization error.

To facilitate efficient deployment in the C++ inference pipeline, I froze the weights and set the model to trace mode using **libtorch**. This process created a static computation graph optimized for inference. An example of tracing a model in PyTorch is shown below:

```python
import torch

# Assume 'model' is an instance of CustomSegNet with loaded weights.
model.eval()  # Freeze batch normalization, dropout, etc.
example_input = torch.randn(1, 3, 512, 512)  # Example input matching expected shape
traced_model = torch.jit.trace(model, example_input)
traced_model.save("model_traced.pt")
```

The traced model, with frozen weights and a static graph, was then exported to ONNX format and integrated into the C++ pipeline.

After exporting the model to ONNX, I integrated Triton for further optimization of the INT8 inference pipeline. With Triton, the system achieved high throughput while maintaining accuracy. The following C++ pseudocode illustrates the inference loop using Triton:

```cpp
// Pseudo-code for Triton-based INT8 inference
#include "triton_inference.h"  // Hypothetical header for Triton integration

TritonEngine engine("model_int8.onnx");
auto inputTensor = prepareInputTensor(inputImageInt8); // Preprocess image data
auto output = engine.runInference(inputTensor);          // Execute inference via Triton
auto segmentationMask = processOutput(output);           // Post-process to obtain segmentation mask
```

**Key results include:**  
- **A 19% increase in IoU** after integrating the channel attention module.  
- **An improvement from 1.3 fps to 32 fps** on the NVIDIA Xavier AGX platform using knowledge distillation and TensorRT optimization.  
- **Deployment on an NXP board achieving 23 fps** with 8-bit quantization.

Field tests were conducted to evaluate the system under realistic conditions, and the results confirmed its ability to perform real-time targeted weed and crop treatment in sugarbeet fields.

In summary, this project described a method for precision weeding that combined semantic segmentation with channel attention, 3D localization using depth data, and several model optimizations including INT8 quantization, Quantization Aware Training (with gradient histogram calibration), and model freezing and tracing using libtorch. Transfer learning was employed to fine-tune the model on limited Hydac data after pre-training on a larger University of Bonn sugarbeet dataset. This approach addressed challenges associated with data scarcity and slow data acquisition, and the system was validated through field tests.