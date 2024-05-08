# ResNet compression

To compress the ResNet by merging BN to adjacent former Conv layer, for both of them are linear layers. To be more specific, 
```python
# separate conv and bn
# conv_w: [C1, C, h, w]
# 
x = conv(x, conv_w, conv_b)
x = batchnorm(x, bn_w, bn_b, mean, var, eps)

# merge the adjacent two layers
conv_w1 = conv_w * (bn_w / torch.sqrt(var + eps))[:, None, None, None]
conv_b1 = bn_w * (conv_b - running_mean) / torch.sqrt(var + eps) + bn_b
x = conv(x, conv_w1, conv_b1)
```
