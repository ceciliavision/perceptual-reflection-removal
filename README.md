## perceptual-reflection-removal


Code and data for paper [Single Image Reflection Removal with Perceptual Losses](https://arxiv.org/abs/1806.05376)

This code is based on tensorflow. It has been tested on Ubuntu 16.04 LTS.

Part of this code is based upon [FastImageProcessing](https://github.com/CQFIO/FastImageProcessing)

#![Our result compared against CEILNet on real images.](./teaser/teaser.png)

# Setup
  * Clone/Download this repo
  * `$ cd reflection-removal`
  * `$ mkdir VGG_Model`
  * Download [VGG-19](http://www.vlfeat.org/matconvnet/pretrained/#downloading-the-pre-trained-models). Search imagenet-vgg-verydeep-19 in this page and download imagenet-vgg-verydeep-19.mat
  * move the downloaded vgg model to VGG_Model


# Training
`python3 main.py`