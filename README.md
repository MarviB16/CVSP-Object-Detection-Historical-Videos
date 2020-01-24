# CVSP-Object-Detection-Historical-Videos

## Darknet

To install darknet and its prerequisites please follow the instructions of the original repo:
https://github.com/AlexeyAB/darknet#requirements

### Training

After installing darknet and preparing the images and groundtruth, one must prepare the dataset. 

After the final dataset preparation, one must download the pretrained weights for the convolutional layers from: 

https://pjreddie.com/media/files/darknet53.conv.74 

into the darknet/ folder. After this is done 

./darknet detector train x64/Release/data/obj.data cfg/cvsp_hist.cfg darknet53.conv.74 -map -dont_show -mjpeg_port 8090

## SSD

## RetinaNet
