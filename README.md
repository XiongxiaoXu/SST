# Mambaformer-in-Time-Series
This repo is the official Pytorch implementation of paper: "[Integrating Mamba and Transformer for Long-Short Range Time Series Forecasting](https://arxiv.org/pdf/2404.14757)".

<img width="525" alt="image" src="https://github.com/XiongxiaoXu/Mambaformer-in-Time-Series/assets/34889516/bb84159b-4a49-41f4-9ae3-e16606b9d742">

## Models and Core Codes
* **Mambaformer** leverages a pre-processing Mamba block and Mambaformer layer without a positional encoding. The architecture is at models/MambaFormer.py, and the layer is at layers/Mamba_Family.py -> AM_Layer
* **Attention-Mamba** adopts a Attention-Mamba layer where an attention layer is followed by a Mamba layer with a positional encoding. The architecture is at models/AttMam.py, and the layer is at layers/Mamba_Family.py -> AM_Layer
* **Mamba-Attention** adopts a Mamba-Attention layer where a Mamba block layer is followed by an attention layer without a positional encoding. The architecture is at models/MamAtt.py, and the layer is at layers/Mamba_Family.py -> MA_Layer
* **Mamba** adopts two Mamba block as a layer. The architecture is at models/Mamba.py, and the layer is at layers/Mamba_Family.py -> Mamba_Layer
* **Transformer** is a decoder-only Transformer architecture. the architecture is at models/DecoderOnly.py, and the layer is at layers/Transformer_EncDec.py -> Decoder_wo_cross_Layer

<img width="1308" alt="image" src="https://github.com/XiongxiaoXu/Mambaformer-in-Time-Series/assets/34889516/3cdd9d58-e8bc-4aa9-a836-16045554e927">

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
