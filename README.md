# [IJCAI 2025] AKBR: Learning Adaptive Kernel-based Representations for Graph Classification
## Dependencies

- Python 3.7+
- torch >= 1.8
- torch-geometric
- numpy
- scikit-learn

## Quick Start
If you want to use all graphs in the dataset as sample graphs, you can run experiments:
```python
bash run.sh
```
If you want to select some prototype graphs in the dataset as sample graphs, you can run experiments:
```python
bash run_sample.sh
```

## Datasets
 Datasets can be download from https://chrsmrrs.github.io/datasets/

## Switch Methods and Datasets
#### Change dataset:
```
--dataset PROTEINS
```
or any supported dataset.
#### Change feature extraction:
WL, SP, RW, or hybrid
```
--method RW
```
#### Change attention type:
```
--attn_type sa
```

## Citing
If you find this work is helpful to your research, please consider citing our paper
```bibtex
@inproceedings{AKBR,
  author       = {Lu Bai and
                  Feifei Qian and
                  Lixin Cui and
                  Ming Li and
                  Hangyuan Du and
                  Yue Wang and
                  Edwin R. Hancock},
  title        = {AKBR: Learning Adaptive Kernel-based Representations for Graph Classification},
  booktitle    = {Proceedings of IJCAI},
  year         = {2025}
}
```
