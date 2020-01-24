# CVSP-Object-Detection-Historical-Videos

## Conda
## Custom scripts

Under custom_scripts/ one can find the script for the dataset preparation (copyImages.ipynb). Open it by using Jupyter on a remote host run (Port forwarding must be enabled):

> jupyter notebook --no-browser --port=8081

On a local machine run:

> jupyter notebook

Then open custom_scripts/copyImages.ipynb.
### First, set the paths for the input images and groundtruth:
**Note**: use the following folder structure:

    path/to/some_folder
    path/to/some_folder/images				<-- Contains the images (Input of OpenLabeling)
    path/to/some_folder/groundtruth			<-- Contains the darknet groundruth (Output of OpenLabeling)
    path/to/some_folder/groundtruth_voc		<-- Contains the voc groundtruth (Output of OpenLabeling)

**Optionally**: The standard image input format is .png, if .jpg is used, adapt the code
**Optionally**: By standard the darknet groundtruth is filtered, so only civilian, soldier and civil vehicle is annotated, if you use a different dataset or want to train with all classes, remove or comment this section.

### Second, run the script:

Depending on the script run two (train,val) or one (test) new folder will appear in the custom_scripts/ folder.

### Third, copy images:
####  (train/val):

**Darknet**: 

Copy:

    darknet_train/train.txt --> darknet/x64/Release/data
    darknet_train/images/*.jpg --> darknet/x64/Release/data/img_train
    darknet_train/groundtruth/*.txt --> darknet/x64/Release/data/img_train
    
    darknet_val/val.txt --> darknet/x64/Release/data
    darknet_val/images/*.jpg --> darknet/x64/Release/data/img_val
    darknet_val/groundtruth/*.txt --> darknet/x64/Release/data/img_val

**VOC**: 

Make a new folder called VOCTemplate with the following subfolders:

    VOCTemplate/VOC2019/Annotations
    VOCTemplate/VOC2019/ImageSets/Main
    VOCTemplate/VOC2019/JPEGImages

Copy:

	darknet_train/trainval_voc.txt --> VOCTemplate/VOC2019/ImageSets/Main/trainval.txt
    darknet_train/images/*.jpg --> VOCTemplate/VOC2019/JPEGImages
    darknet_train/groundtruth_voc/*.txt --> VOCTemplate/VOC2019/Annotations

	darknet_val/val_voc_voc.txt --> VOCTemplate/VOC2019/ImageSets/Main/val.txt
    darknet_val/images/*.jpg --> VOCTemplate/VOC2019/JPEGImages
    darknet_val/groundtruth_voc/*.txt --> VOCTemplate/VOC2019/Annotations

####  (test):

**Darknet**: 
Copy:

    darknet_test/test.txt --> darknet/x64/Release/data
    darknet_test/images/*.jpg --> darknet/x64/Release/data/img_test
    darknet_test/groundtruth/*.txt --> darknet/x64/Release/data/img_test

**VOC**: 
Copy:

	darknet_test/test_voc.txt --> VOCTemplate/VOC2019/ImageSets/Main/test.txt
    darknet_test/images/*.jpg --> VOCTemplate/VOC2019/JPEGImages
    darknet_test/groundtruth_voc/*.txt --> VOCTemplate/VOC2019/Annotations

## Darknet

To install darknet and its prerequisites please follow the instructions of the original repo:
https://github.com/AlexeyAB/darknet#requirements

### Training

After installing darknet and preparing the images and groundtruth, one must prepare the dataset. Instructions for this can be found here:

    https://github.com/AlexeyAB/darknet#how-to-train-to-detect-your-custom-objects

**Note**: A dummy dataset structure is already prepared under darknet/x64. If you just want to redo the training  on the civilian, soldier and civil vehicle, just copy the training/validation **images** into 

    darknet/x64/Release/data/img_train

and the **train.txt** and **val.txt** into 

    darknet/x64/Release/data/

After the final dataset preparation, one must download the pretrained weights for the convolutional layers from: 

    https://pjreddie.com/media/files/darknet53.conv.74 

into the darknet/ folder. After this is done run:

    ./darknet detector train x64/Release/data/obj.data cfg/cvsp_hist.cfg darknet53.conv.74 -map -dont_show -mjpeg_port 8090

to train the network. The following custom training configs have been made:

| CFG Name | Setting |
|--|--|
| cvsp_hist_no_aug.cfg | Training without augmentation |
| cvsp_hist.cfg | Training with augmentation |
| cvsp_hist_freeze_1.cfg | Training with augmentation, freeze before first Yolo-Layer |
| cvsp_hist_freeze_2.cfg | Training with augmentation, freeze before second Yolo-Layer |
| cvsp_hist_freeze_3.cfg | Training with augmentation, freeze before third Yolo-Layer |

## SSD

## RetinaNet

## Problems and solutions

 - Darknet can't read input image
	 - Solution: Don't use OpenCV to save the image, use Scipy
 - 

