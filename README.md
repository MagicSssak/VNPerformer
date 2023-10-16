# VNPerformer

Created by <a href="xf2219@columbia.edu" target="_blank">Xuande Feng</a> and <a href="zl3101@columbia.edu" target="_blank">Zonglin Lyu</a>.

We reproduced the VNTransformer based on <a href="https://arxiv.org/pdf/2206.04176.pdf" target="_blank">VN-Transformer: Rotation-Equivariant Attention for Vector Neurons</a>.

## Overview
This is a Performerized VNTransformer which introduces softmax kernel to relax the quadratic time complexity of VNTransformer.
## Data Preparation

+ Classification: Download [ScanObjectNN](https://hkust-vgd.github.io/scanobjectnn/).

## Usage

### Training Classification on ModelNet40 with Transformer
```
python train_cls.py
```
### With Performer with antithetic random feature sampling
```
python train_cls.py --kernel --antithetic --num_random 20
```
### Training Classification on ScanObjectNN with Transformer
```
python train_cls.py --data_name 'Scan'
```
### Training Classification on ScanObjectNN with Performer
```
python train_cls.py --data_name 'Scan' --kernel --antithetic --num_random 20
```

### Uptrainning Performer on ScanObjectNN(ModelNet40 by default) with stored Transformer checkpoints
```
python train_cls.py --data_name 'Scan' --kernel --antithetic --num_random 20 restore --'path to your restored transformer'
```



## License
MIT License

## Acknowledgement
The structure of this codebase is borrowed from this pytorch implementataion of [Vector Neurons](https://github.com/FlyingGiraffe/vnn).
