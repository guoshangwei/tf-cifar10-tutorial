# tf-cifar10-tutorial

## Train models with different initializers

- With RandomNormal initializer

```{r, engine='bash', count_lines}
python main.py
```
- With Orthogonal initializer
```{r, engine='bash', count_lines}
python main.py --init_para 1
```
## Evaluate trained model

- Model with RandomNormal initializer
```{r, engine='bash', count_lines}
python main.py --train 0
```
- Model with Orthogonal initializer

```{r, engine='bash', count_lines}
python main.py --train 0 --init_para 1
```
