# VisualBackProp
VisualBackProp - visualization method for convolutional neural networks

# Description
Detailed description of the VisualBackProp can be found in:
https://arxiv.org/abs/1611.05418

Content:

* inputImages - folder with input images
* outputImages - folder for output images
* vis.lua - implementation of VisualBackProp 
* run.lua - loads images, runs visualization method, saves images
* model.t7b - model trained to predict steering wheel angle

# Usage
`qlua run.lua`

Script generates set of 3 images for each input image. Each set contain:

* Input image with visualization mask overlaid in red
* Averaged feature maps
* Intermediate masks
