# Advanced Training Concepts

## Late Assignment on Time

## Group Members
- Harshita Sharma - harshita23sharma@gmail.com
- Pankaj S K - pankajsureshkumar7@gmail.com
- Avichal Jain - avichaljain1997@gmail.com
- Aman Kumar - aman487jain@gmail.com

## Table of Contents
- [Group Members](https://github.com/amanjain487/tsai-eva6/blob/main/Assignments/S8/README.md#group-members)
- [Table of Contents](https://github.com/amanjain487/tsai-eva6/blob/main/Assignments/S8/README.md#table-of-contents)
- [Objective](https://github.com/amanjain487/tsai-eva6/blob/main/Assignments/S8/README.md#objective)
- [Parameters and Hyperparameters](https://github.com/amanjain487/tsai-eva6/blob/main/Assignments/S8/README.md#parameters-and-hyperparameters)
- [Results](https://github.com/amanjain487/tsai-eva6/blob/main/Assignments/S8/README.md#results)
- [Graphs for Losses and Accuracies](https://github.com/amanjain487/tsai-eva6/blob/main/Assignments/S8/README.md#graphs-for-losses-and-accuracies)
- [Correctly Classified Images](https://github.com/amanjain487/tsai-eva6/blob/main/Assignments/S8/README.md#correctly-classified-images)
- [GradCAM for Correctly Classified Images](https://github.com/amanjain487/tsai-eva6/blob/main/Assignments/S8/README.md#gradcam-for-correctly-classified-images)
- [Incorrectly Classified Images](https://github.com/amanjain487/tsai-eva6/blob/main/Assignments/S8/README.md#incorrectly-classified-images)
- [GradCAM for Incorrectly Classified Images](https://github.com/amanjain487/tsai-eva6/blob/main/Assignments/S8/README.md#gradcam-for-incorrectly-classified-images)

## Objective
- Train for 40 Epochs
- Display 20 misclassified images
- Display 20 GradCam output on the SAME misclassified images
- Apply these transforms while training:
  - RandomCrop(32, padding=4)
  - CutOut(16x16)
  - Rotate(±5°)
- Must use ReduceLROnPlateau
- Must use LayerNormalization ONLY.

## Parameters and Hyperparameters
- Loss Function: Cross Entropy Loss
- Optimizer: SGD
- Scheduler: ReduceLROnPlateau
- Batch Size: 128
- Learning Rate: lr=0.01
- Epochs: 40

## Results
- Best Accracy is 89.17% at 39th Epoch.

## Graphs for Losses and Accuracies
![image](https://user-images.githubusercontent.com/46129975/124268705-7e262980-db57-11eb-831e-fa0f1ab2c045.png)

## Correctly Classified Images
![image](https://user-images.githubusercontent.com/46129975/124268846-b7f73000-db57-11eb-8597-234d7b5bc587.png)


## GradCAM for Correctly Classified Images
|GradCAM Output|GradCAM Output|
|--------------|--------------|
|![image](https://user-images.githubusercontent.com/46129975/124269298-57b4be00-db58-11eb-8e36-0062fb611871.png)|![image](https://user-images.githubusercontent.com/46129975/124269336-613e2600-db58-11eb-862e-71788c1180ef.png)|
|![image](https://user-images.githubusercontent.com/46129975/124269373-6c915180-db58-11eb-86bc-272c6469424d.png)|![image](https://user-images.githubusercontent.com/46129975/124269387-70bd6f00-db58-11eb-9f71-25ca29d0b5e4.png)|
|![image](https://user-images.githubusercontent.com/46129975/124269402-74e98c80-db58-11eb-8af6-02985e844618.png)|![image](https://user-images.githubusercontent.com/46129975/124269810-fe00c380-db58-11eb-9c40-3bfd8af241f6.png)|
|![image](https://user-images.githubusercontent.com/46129975/124269420-7a46d700-db58-11eb-9c38-af9f940a5b4b.png)|![image](https://user-images.githubusercontent.com/46129975/124269426-7e72f480-db58-11eb-8910-3cda2657ebaa.png)|
|![image](https://user-images.githubusercontent.com/46129975/124269446-83d03f00-db58-11eb-9459-7e53bbc9e284.png)|![image](https://user-images.githubusercontent.com/46129975/124269603-b5e1a100-db58-11eb-99fb-ffb2d154b396.png)|
|![image](https://user-images.githubusercontent.com/46129975/124269588-b11ced00-db58-11eb-9b1e-53bc7b55725a.png)|![image](https://user-images.githubusercontent.com/46129975/124269580-acf0cf80-db58-11eb-8e09-737368d0ffc8.png)|
|![image](https://user-images.githubusercontent.com/46129975/124269565-a82c1b80-db58-11eb-896b-976429fa558c.png)|![image](https://user-images.githubusercontent.com/46129975/124269555-a4989480-db58-11eb-8ca8-d0764450998f.png)|
|![image](https://user-images.githubusercontent.com/46129975/124269543-a1050d80-db58-11eb-89c2-bdf3a430a7e6.png)|![image](https://user-images.githubusercontent.com/46129975/124269532-9cd8f000-db58-11eb-9e2d-54b8ab7c475e.png)|
|![image](https://user-images.githubusercontent.com/46129975/124269520-98acd280-db58-11eb-8937-66c240ef9b13.png)|![image](https://user-images.githubusercontent.com/46129975/124269504-9480b500-db58-11eb-8565-55bbcf7e060e.png)|
|![image](https://user-images.githubusercontent.com/46129975/124269486-8fbc0100-db58-11eb-9415-be437cd81c8c.png)|![image](https://user-images.githubusercontent.com/46129975/124269464-8af74d00-db58-11eb-9998-09f54175c6e4.png)|

## Incorrectly Classified Images
![image](https://user-images.githubusercontent.com/46129975/124268884-c6dde280-db57-11eb-8c34-33f12de6bdce.png)


## GradCAM for Incorrectly Classified Images
|GradCAM Output|GradCAM Output|
|--------------|--------------|
|![image](https://user-images.githubusercontent.com/46129975/124270531-f1c93600-db59-11eb-8cb4-72f25a32a735.png)|![image](https://user-images.githubusercontent.com/46129975/124270515-ee35af00-db59-11eb-8619-313f6ef7774c.png)|
|![image](https://user-images.githubusercontent.com/46129975/124270497-e8d86480-db59-11eb-83c2-9188182e6702.png)|![image](https://user-images.githubusercontent.com/46129975/124270487-e4ac4700-db59-11eb-98b5-918db2a33655.png)|
|![image](https://user-images.githubusercontent.com/46129975/124270476-e0802980-db59-11eb-820e-02f3027a35bc.png)|![image](https://user-images.githubusercontent.com/46129975/124270454-db22df00-db59-11eb-99a1-8ffd29a14bab.png)|
|![image](https://user-images.githubusercontent.com/46129975/124270444-d65e2b00-db59-11eb-9336-9b0833fb6214.png)|![image](https://user-images.githubusercontent.com/46129975/124270429-d2320d80-db59-11eb-9d8e-eeac2973f1ba.png)|
|![image](https://user-images.githubusercontent.com/46129975/124270413-ccd4c300-db59-11eb-8340-76e98b9f00a5.png)|![image](https://user-images.githubusercontent.com/46129975/124270399-c9413c00-db59-11eb-8825-6c599e5074ec.png)|
|![image](https://user-images.githubusercontent.com/46129975/124270391-c5151e80-db59-11eb-838c-e93e6dbc0c72.png)|![image](https://user-images.githubusercontent.com/46129975/124270382-c21a2e00-db59-11eb-87f4-99d8742dc53c.png)|
|![image](https://user-images.githubusercontent.com/46129975/124270367-bd557a00-db59-11eb-97c2-2b017203c9ac.png)|![image](https://user-images.githubusercontent.com/46129975/124270357-b9295c80-db59-11eb-8f87-537c88cef7b5.png)|
|![image](https://user-images.githubusercontent.com/46129975/124270342-b4fd3f00-db59-11eb-9dca-80e951e9d7fc.png)|![image](https://user-images.githubusercontent.com/46129975/124270322-b0d12180-db59-11eb-8c53-115fe75aabe9.png)|
|![image](https://user-images.githubusercontent.com/46129975/124270309-ac0c6d80-db59-11eb-96df-968662cbac63.png)|![image](https://user-images.githubusercontent.com/46129975/124270292-a6af2300-db59-11eb-8b1a-b55bd7bcc760.png)|
|![image](https://user-images.githubusercontent.com/46129975/124270277-a0b94200-db59-11eb-9b2a-150e7b82e390.png)|![image](https://user-images.githubusercontent.com/46129975/124270260-9c8d2480-db59-11eb-873e-917b6dd2fbb7.png)|
