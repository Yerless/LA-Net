## Installation

Install [TensorFlow](https://www.tensorflow.org/). The code has been tested with Python 3.6.9, Tensorflow 1.13.2, CUDA 9.2 and cuDNN 7.6.1 on Ubuntu 18.04

Install h5py for Python

```shell
sudo apt-get install libhdf5-dev
sudo pip install h5py 
```

## Usage

### Classification

To train LA-Net (with sample size 32 in this example):

```shell
python train.py --num_out_points 32--log_dir log/LA-Net/32
```

To evaluate the classifier over LA-Net's sampled points:

```shell
python evaluate.py --sampler_model_path log/LA-Net/32/model.ckpt --dump_dir log/LA-Net/32/eval
```

### Retrieval

To train LA-Net (with sample size 32 in this example):

```shell
python train.py --num_out_points 32--log_dir log/LA-Net/32
```

To evaluate the classifier over LA-Net's sampled points and save data for retrieval:

```shell
python evaluate.py --sampler_model_path log/LA-Net/32/model.ckpt --num_out_points 32 --dump_dir log/LA-Net/32/eval --save_retrieval_vectors 1
```

To analyze retrieval data:

```shell
python analyze_precision_recall.py --num_out_points 32 --dump_dir log/LA-Net/32/retrieval/
```

## Acknowledgment

Our code builds upon the code provided by [Qi et al](https://github.com/charlesq34/pointnet) and [orendv et al](https://github.com/orendv/learning_to_sample). We would like to thank the authors for sharing their code.

