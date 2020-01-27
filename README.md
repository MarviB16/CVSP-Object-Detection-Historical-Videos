# CVSP-Object-Detection-Historical-Videos

## Conda

### Installation
To install Anaconda (Not miniconda) use this guide:

https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html

For this evaluation Anaconda version 4.8.0, has been used.
### Environments

#### Util

The util evironment is used for the custom scripts.

#### SSD

The SSD environtment is used for the training and evaluation of SSD.

With Anaconda installed the installation is quit simple. First create the environment by running:

    conda create -n SSD --file ssd_clean.txt

then install the kernel into Jupyter by first activating the environment and then by running:

    conda activate SSD
    python -m ipykernel install --user --name SSD --display-name "SSD"


#### RetinaNet

The RetinaNet environment is used for training and evaluating RetinaNet.

With Anaconda installed the installation is quit simple. First create the environment by running:

    conda create -n RetinaNet --file retinanet_clean.txt

then activate the environment by running:

    conda activate RetinaNet

then install all the dependencies which are not available in Conda, by running:

	pip install -r retinaNet_pip_Requirements.txt

**Note**: If you encounter the Error 13 "Permission denied" see the problems section.
And finally run:

	python -m ipykernel install --user --name RetinaNet --display-name "RetinaNet"

to install the kernel to Jupyter.

#### Darknet

As darknet doesn't use python (for training) no conda environment is needed.

## Custom scripts

Under custom_scripts/ one can find the script for the dataset preparation (copyImages.ipynb). Open it by using Jupyter, on a remote host run (Port forwarding must be enabled):

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

to train the network. **-map** calculates the mean average presision every 1000 steps, **-dont_show** hides the loss curve (which would crash the training when running on a remote pc) and **-mjpeg_port** allows watching the loss/map-chart on the local pc under localhost:8090 (SSH Port forwarding is needed for this, see https://www.ssh.com/ssh/tunneling/example for instructions). The following custom training configs have been made:

| CFG Name | Setting |
|--|--|
| cvsp_hist_no_aug.cfg | Training without augmentation |
| cvsp_hist.cfg | Training with augmentation |
| cvsp_hist_freeze_1.cfg | Training with augmentation, freeze before first Yolo-Layer |
| cvsp_hist_freeze_2.cfg | Training with augmentation, freeze before second Yolo-Layer |
| cvsp_hist_freeze_3.cfg | Training with augmentation, freeze before third Yolo-Layer |

## SSD

Under ssd_keras/ one can find the scripts for training (cvsp_ssd300_training_custom.ipynb) and evaluating (cvsp_ssd300_evaluation.ipynb) SSD. Open them by using Jupyter, on a remote host run (Port forwarding must be enabled):

> jupyter notebook --no-browser --port=8081

On a local machine run:

> jupyter notebook

Download the pretrained weights from:

    https://drive.google.com/open?id=1sBmajn6vOE7qJ8GnxUJt4fGPuffVUZox

For a detailed description on how to train the network read the readme from the orginial repo, which you can find under ssd_keras/README.md.

If you just want to redo the training follow these steps:

 1. Change the path in **line 23** in **cell 2.1** to the just downloaded weights (but then don't run run cell 2.2).
 2. **OR:** Change the path in **line 2** in **cell 2.2** to previously trained weights to continue training.
 3. Change the paths in **lines 13, 16, 20 and 21** in **cell 3** to the respective dataset paths.
 4. In the second cell of **4** change **line 4** to the folder were the weights should be saved in.
 5. Change **initial epoch**, **final epoch** and **steps per epoch** to your desired numbers.

If you just want to redo the evaluation follow these steps:
 1. Change the path in **line 32** in **cell 1.1** to the trained weights.
 2. Change **lines 4 - 6** in **cell 2** to your dataset.
 4. Run all sections **until cell 5**. This gives you the mAP and the Precision-Recall-Curve for civilian and soldier.

## RetinaNet

RetinaNet doesn't use a Jupyter notebook for training. For a detailed guide how to train RetinaNet take a look at the readme under keras_retinanet/README.md. 

If you just want to redo the training use this command:

MobileNet224:

    python keras_retinanet/bin/train.py --backbone=mobilenet224_1.0 --gpu=1 pascal /path/to/VOCTemplate/VOC2019/

ResNet50:
    
    python keras_retinanet/bin/train.py --backbone=resnet50 --gpu=1 pascal /path/to/VOCTemplate/VOC2019/

ResNet152
    
    python keras_retinanet/bin/train.py --backbone=resnet152--gpu=1 pascal /path/to/VOCTemplate/VOC2019/

**Note:** With the --gpu flag you can specify a gpu

If you want to evaluate run:

    python keras_retinanet/bin/evaluate.py --gpu=1 --backbone=resnet152 --score-threshold=0.05 --iou-threshold=0.5  pascal /path/to/VOCTemplateTest/VOC2019/ snapshots/resnet152_pascal_05.h5
**Note:** Exchange the backbone to fit the weights (mobilenet224_1.0, resnet50, resnet152)

## Problems and solutions

 - Darknet can't read input image
	 - Solution: Don't use OpenCV to save the image, use Scipy
 - While installing the pip requirements with RetinaNet you encounter the Error 13 Permission denied?
	 - Check which pip is used, by running "which pip". You should see that it uses the system python and pip. To fix this modify the PATH system variable, such that the path to anaconda is first. (`echo $PATH` `export PATH=/path/to/anaconda3/envs/RetinaNet/bin:/path/to/python/bin`)
 - How do i show the code line in Jupyter?
	 - ESC then l (small L), has to be done per cell.
