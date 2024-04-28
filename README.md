# Mambaformer-in-Time-Series
This repo is the official Pytorch implementation of paper: "[Integrating Mamba and Transformer for Long-Short Range Time Series Forecasting](https://arxiv.org/pdf/2404.14757)".

## Code
The description of the core code files.
* models/MambaFormer.py: implementation of of Mambaformer
* models/AttMam.py: implementation of  Attention-Mamba hybrid
* models/MamAtt.py: implementation of Mamba-Attention hybrid
* models/Mamba.py: implementation of Mamba
* models/DecoderOnly.py: implementation of decoder-only Transformer

## Getting Started
## Environment
* python            3.10.13
* torch             1.12.1+cu116
* mamba-ssm         1.2.0.post1
* numpy             1.26.4
* transformers      4.38.2
The installation of mamba-ssm package can refer to https://github.com/state-spaces/mamba. 

## Run
To get the result of Table 2, run the scripts etth1.sh, electricity.sh, and exchange_rate.sh. For exmaple, run etth1.sh:
`./etth1.sh`

## Acknowledgement
We would like to greatly thank 

## Cite
If you find this repository useful for your work, please consider citing the paper as follows:

```bibtex
@article{xu2024integrating,
  title={Integrating Mamba and Transformer for Long-Short Range Time Series Forecasting},
  author={Xu, Xiongxiao and Liang, Yueqing and Huang, Baixiang and Lan, Zhiling and Shu, Kai},
  journal={arXiv preprint arXiv:2404.14757},
  year={2024}
}
```
