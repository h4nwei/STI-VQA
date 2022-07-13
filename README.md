# STI-VQA
This repository contains the offical implementations along with the experimental splits for the paper "Learning Spatiotemporal Interactions for User-Generated Video Quality Assessment",
 Hanwei Zhu, Baoliang Chen, Lingyu Zhu, and [Shiqi Wang](https://www.cs.cityu.edu.hk/~shiqwang/).


## Framework
![framework](./data/framework.png)

### Prerequisites
The release codes were implemented and have been tested in Ubuntu 18.04 with
- Python = 3.6.13
- PyTorch = 1.8.1
- torchvision = 0.9.0 

## Feature extraction
More details can be found in README.md in the folder of `feature`.

## Training and Testing on VQA Databases
You can change the paramers in `param.py` to train and test each dataset with intra-/cross-dataset settings
```bash
# Training
python main.py --test_only False
# Test
python main.py --test_only True
```


## Acknowledgement
The authors would like to thank Dingquan Li for his implementation of [VSFA](https://github.com/lidq92/VSFA), [Yang Li](https://github.com/sherlockyy) for his code architecture, the [BVQA_Benchmark](https://github.com/vztu/BVQA_Benchmark), and the implementation of [ViT](https://github.com/lucidrains/vit-pytorch).

## Citation
```bibtex
@article{zhu2022learing,
title={Learning Spatiotemporal Interactions for User-Generated Video Quality Assessment},
author={Zhu, Hanwei and Chen, Baoliang and Zhu, lingyu and Wang, Shiqi},
journal={coming soon},
volume={},
number={},
pages={},
month={},
year={}
}
