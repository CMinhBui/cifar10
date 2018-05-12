# Bottleneck Resnet
A implementation of bottle neck resnet for cifar10 with 87.53% accuracy. Implemented in Keras

## Requirements

```
keras
tensorflow
python3
```

## Usage

Train model:
```
python train.py
```
Optional argument: `python train.py -h`

Evaluate model:
```
python train.py --mode eval --from-pretrain output/resnet/model.h5
```