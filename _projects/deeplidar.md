---
layout: page
title: DeepLiDARFlow
description: Bachelor thesis; worked on scene flow estimation using monocular camera & sparse LiDAR. Work accepted at two conferences.
img: assets/img/deeplidarflow.png
importance: 1
category: research
related_publications: true
---
### Abstract

Scene flow is the dense 3D reconstruction of motion and geometry of a scene. Most state-of-the-art methods use a pair of stereo images as input for full scene reconstruction. These methods depend a lot on the quality of the RGB images and perform poorly in regions with reflective objects, shadows, ill-conditioned light environment and so on. LiDAR measurements are much less sensitive to the aforementioned conditions but LiDAR features are in general unsuitable for matching tasks due to their sparse nature. Hence, using both LiDAR and RGB can potentially overcome the individual disadvantages of each sensor by mutual improvement and yield robust features which can improve the matching process. In this paper, we present DeepLiDARFlow, a novel deep learning architecture which fuses high level RGB and LiDAR features at multiple scales in a monocular setup to predict dense scene flow. Its performance is much better in the critical regions where image-only and LiDAR-only methods are inaccurate. We verify our DeepLiDARFlow using the established data sets KITTI and FlyingThings3D and we show strong robustness compared to several state-of-the-art methods which used other input modalities.

### Resources
- **Code Repository:** [GitHub Link](https://github.com/dfki-av/DeepLiDARFlow)
- **Paper:** [ArXiV](https://arxiv.org/abs/2008.08136)

