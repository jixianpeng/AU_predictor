# Note
AU[1,2,4,6,12,15,20,25]

### 1
Due to the limition of file_size  and the off-line operation in this project,we give a cloud disk url https://pan.baidu.com/s/1rOgVD-5Lzz5gRqQt0f0t0g ,in which the segmented faces,border images and the landmarks stored.

### 2
All of the preprocessing about the boder image and segmented face also have been implemented in the data-seg-store.py

### 3
With all the pre-calculated data already,AU_Test.py could be used for prediction and the model could be re-trained with Aspect_based_train.py 
### 4
Same path structure as ./test/ are needed for the ./train/ and ./valid/ if you would like to retrain the model while the landmarks should be detected by a certain tool to which you have access. In addition, the label files need to be add to the path ./Training_Set/, so do the path ./Validation_Set/. 

# Dependency
    pytoch==1.1.0
    numpy==1.16.4
    opencv-python==4.1.1.26
    Pillow==6.1.0
    torchvision==0.4.1

# Usage for test
### 1. Clone the repository
    git clone https://github.com/jixianpeng/AU_predictor.git 
    cd AU_predictor
### 2. Download the landmarks,segmented faces and the border images from the [url](https://pan.baidu.com/s/1rOgVD-5Lzz5gRqQt0f0t0g). Decompress and place the data in them into the corresponding path in this project by the name.
### 3. Adding the cropped_aligned frames and the original videos to the ./test/cropped_aligned/ and ./test/video/ without any change,Manually.
### 4.test
    python3 ./Au_Test.py
    
the results will be stored into ./prediction/, each file corrresponding to a file in the dataset by the name
