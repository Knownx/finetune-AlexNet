**Updated 03.25.2018**

# Finetune AlexNet with TensorFlow 1.1.0
In this project, AlexNet finetuning was implemented with tensorflow. The final accuracy was just 65.57% with only 2 epochs. Didn't upload the dogs-vs-cats datasets and pretrained files (bvlc_alexnet.npy). Please download them from below attached urls.

## Requirements
- Python3
- TensorFlow 1.1.0
- Numpy
- OpenCV
- Datasets: https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data
- Pretrained parameters (bvlc_alexnet.npy): http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/

## Codes structure
- `alexnet.py` --- The implementation of AlexNet model
- `config.py` --- Configuration for some parameters, not all
- `dataset.py` --- Data processing calss that contains data split, data preprocessing and getNextBatch functions
- `finetune.py` --- Main codes of finetuning AlexNet with dogs-vs-cats datasets

## Points for post optimizing
There are some TensorBoard usage issues need to be completed

## Reference:
https://github.com/kratzert/finetune_alexnet_with_tensorflow/tree/5d751d62eb4d7149f4e3fd465febf8f07d4cea9d
https://kratzert.github.io/2017/02/24/finetuning-alexnet-with-tensorflow.html

