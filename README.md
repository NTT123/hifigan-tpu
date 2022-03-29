# hifigan-tpu
Train HiFi-GAN on TPU and other fixes.

## Introduction

[HiFi-GAN](https://arxiv.org/abs/2010.05646) is a popular GAN vocoder that achieves very good audio quality and real-time speech generation on CPU. The official HiFi-GAN implementation is at https://github.com/jik876/hifi-gan

This repo tried different things to improve the official implementation:

1. It uses JAX library so you can train your GAN vocoder on TPUs. It can run on Google Colab TPUv2 with a similar speed to a V100 GPU. It can run x3 faster (than a V100) on a TPUv3 (tested on Kaggle TPU).

2. Even though the HiFi-GAN paper claims that its generator is a fully convolutional neural network (FCN), its official implementation uses padding at conv layers to keep the dimension the same. So it is not a FCN. This repo provides a FCN generator which leads to an improvement in the mel-spectrogram loss.

3. In this repo, the learning rate is reduced by a factor of 0.999 for every 1000 steps. This is different from the original implementation which reduces the learning rate for every epoch. For a small dataset, this can cause the learning rate to be reduced too fast.


## Instructions

```sh
pip3 install -r requirements.txt
python3 ljs.py
python3 prepare_data.py --wav-dir=/path/to/wav/dir
python3 train.py --data-dir=/path/to/wav/dir
```
