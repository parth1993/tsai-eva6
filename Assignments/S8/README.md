# Advanced Training Concepts

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
|![image](https://user-images.githubusercontent.com/46129975/124269402-74e98c80-db58-11eb-8af6-02985e844618.png)|![image](https://user-images.githubusercontent.com/46129975/124269420-7a46d700-db58-11eb-9c38-af9f940a5b4b.png)|
|![image](https://user-images.githubusercontent.com/46129975/124269426-7e72f480-db58-11eb-8910-3cda2657ebaa.png)|![image](https://user-images.githubusercontent.com/46129975/124269446-83d03f00-db58-11eb-9459-7e53bbc9e284.png)|
|![image](https://user-images.githubusercontent.com/46129975/124269616-b9752800-db58-11eb-8c8f-ec6c25db64bd.png)|![image](https://user-images.githubusercontent.com/46129975/124269603-b5e1a100-db58-11eb-99fb-ffb2d154b396.png)|
|![image](https://user-images.githubusercontent.com/46129975/124269588-b11ced00-db58-11eb-9b1e-53bc7b55725a.png)|![image](https://user-images.githubusercontent.com/46129975/124269580-acf0cf80-db58-11eb-8e09-737368d0ffc8.png)|
|![image](https://user-images.githubusercontent.com/46129975/124269565-a82c1b80-db58-11eb-896b-976429fa558c.png)|![image](https://user-images.githubusercontent.com/46129975/124269555-a4989480-db58-11eb-8ca8-d0764450998f.png)|
|![image](https://user-images.githubusercontent.com/46129975/124269543-a1050d80-db58-11eb-89c2-bdf3a430a7e6.png)|![image](https://user-images.githubusercontent.com/46129975/124269532-9cd8f000-db58-11eb-9e2d-54b8ab7c475e.png)|
|![image](https://user-images.githubusercontent.com/46129975/124269520-98acd280-db58-11eb-8937-66c240ef9b13.png)|![image](https://user-images.githubusercontent.com/46129975/124269504-9480b500-db58-11eb-8565-55bbcf7e060e.png)|
|![image](https://user-images.githubusercontent.com/46129975/124269486-8fbc0100-db58-11eb-9415-be437cd81c8c.png)|![image](https://user-images.githubusercontent.com/46129975/124269464-8af74d00-db58-11eb-9998-09f54175c6e4.png)|

## Incorrectly Classified Images
![image](https://user-images.githubusercontent.com/46129975/124268884-c6dde280-db57-11eb-8c34-33f12de6bdce.png)


## GradCAM for Incorrectly Classified Images
