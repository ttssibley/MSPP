# Micrograph Segmentation using PixelPick

This repository implements PixelPick for low-annotation micrograph segmentation. The code is based on the original 
PixelPick paper (https://arxiv.org/abs/2104.06394).

## Setup
```shell
pip install -r requirements.txt
```

## PixelPick query configuration
A sample query configuration can be found in `configs/uhcs/default.yaml`. The parameters for making queries are:
- `max_pixels`: maximum number of pixels to query excluding the starting pixels
- `n_pixels_per_round`: number of pixels to query in each round for each class
- `n_init_pixels_per_class`: number of pixels to query for each class at the beginning
- `query_strategy`: the query strategy to use, can be `random`, `entropy`, `margin`, `least_confidence`.
- `top_n_percent`: the top n percent of pixels to randomly sample from using the query strategy. For example, 
if `top_n_percent` is 0.05 and `query_strategy` is `entropy`, then the top 5% pixels with the highest entropy will be 
randomly sampled from.

## Data preparation and training configuration
The details of data preparation and training configuration can be found at https://github.com/leibo-cmu/MatSeg.

## Training
The following command 
trains segmentation models with incremental queries using the `entropy` query strategy:
```shell
python train_pixelpick.py --config entropy.yaml
```