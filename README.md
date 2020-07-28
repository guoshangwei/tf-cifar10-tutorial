# tf-cifar10-tutorial

## Train models with different initializers

### With RandomNormal initializer

python main.py

### With Orthogonal initializer

python main.py --init_para 1

## Evaluate trained models

### Model with RandomNormal initializer
python main.py --train 0

### Model with Orthogonal initializer
python main.py --train 0 --init_para 1
