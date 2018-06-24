# perceptual-reflection-removal


Code and data for paper [Single Image Reflection Removal with Perceptual Losses](https://arxiv.org/abs/1806.05376)

This code is based on tensorflow. It has been tested on Ubuntu 16.04 LTS.

##![Our result compared against CEILNet on real images.](./teaser/teaser.png)

## Setup
  * Clone/Download this repo
  * `$ cd perceptual-reflection-removal`
  * `$ mkdir VGG_Model`
  * Download [VGG-19](http://www.vlfeat.org/matconvnet/pretrained/#downloading-the-pre-trained-models). Search imagenet-vgg-verydeep-19 in this page and download imagenet-vgg-verydeep-19.mat. We need the pre-trained VGG-19 model for our hypercolumn input and feature loss
  * move the downloaded vgg model to `VGG_Model`

# Dataset
[Here](https://drive.google.com/drive/folders/1NYGL3wQ2pRkwfLMcV2zxXDV8JRSoVxwA?usp=sharing) is the link to real dataset. 

You can also try with your own dataset. For example, to generate your own synthetic dataset, prepare for two folders of images, one used for transmission layer and the other used for reflection layer. In order to use our data loader, please follow the instructions below to organize your dataset directories. Overall, we name the directory of input image with reflection `blended`, the ground truth transmission layer `transmission_layer` and the ground truth reflection layer `reflection_layer`.

For synthetic data, since we generate images on the fly, there is no need to have the `blended` subdirectory.
>+-- `root_training_synthetic_data`<br>
>>+-- `reflection_layer`<br>
>>+-- `transmission_layer`<br>

For real data, since the ground truth reflection layer is not available, there is no need to have the `reflection_layer` subdirectory.
>+-- `root_training_real_data`<br>
>>+-- `blended`<br>
>>+-- `transmission_layer`<br>

## Training
###Quick Start
(assume you have the dataset paths set up):

`python3 main.py` for triaining from scratch

`python3 main.py --continue_training` for training with existing checkpoint (path specified by the `--task` argument)

### Other Arguments
`--task`: the checkpoint directory path. For example, for `--task experiment_1`, the checkpoints are saved inside `./experiment_1/`

`--data_syn_dir`: root path to the images to generate synthetic data

`--data_real_dir`: root path to the real images

`--save_model_freq`: frequency to save model and the output images

`--is_hyper`: whether to use hypercolumn features as input, all our trained models uses hypercolumn features as input


<!--## Testing
Testing and evaluation codes coming soon.-->

## Acknowledgement
Part of the code is based upon [FastImageProcessing](https://github.com/CQFIO/FastImageProcessing)

## Citation
If you find this work useful for your research, please cite:

```
@inproceedings{zhang2018single,
  title = {Single Image Reflection Separation with Perceptual Losses},
  author = {Zhang, Xuaner and Ng, Ren and Chen, Qifeng}
  booktitle = {IEEE Conference on Computer Vision and Pattern Recognition},
  month = june,
  year = {2018}
}
```

## Contact
Please contact me if there is any question (Cecilia Zhang <cecilia77@berkeley.edu>)