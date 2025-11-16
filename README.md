# SST
The SST (State Space Transformer) paper has been accepted into CIKM'25. This repo is the code for the paper "[SST: Multi-Scale Hybrid Mamba-Transformer Experts for Time Series Forecasting](https://arxiv.org/abs/2404.14757)".

<img width="1075" alt="image" src="https://github.com/user-attachments/assets/93128514-7ada-4f3e-9c5e-3fad8bde8ae1">

## Contributions
* We propose to **decompose time series into global patterns and local variations according to ranges**. We identify that global patterns as the focus of long range and local variations should be captured in short range.
* To effectively capture long-term patterns and short-term variations, we leverage the patching to create coarser PTS in long range and finer PTS in short range. Moreover, we introduce **a new metric to precisely quantify the resolution of PTS**.
* We propose a **novel hybrid Mamba-Transformer experts architecture SST**, with Mamba as a global patterns expert in long range, and LWT as a local variations expert in short range. A long-short router is designed to adaptively integrate the global patterns and local variations. **With Mamba and LWT, SST is highly scalable with linear complexity O(L) on time series length L**.

## Getting Started
### Environment
* python            3.10.13
* torch             1.12.1+cu116
* mamba-ssm         1.2.0.post1
* numpy             1.26.4
* transformers      4.38.2

The installation of mamba-ssm package can refer to https://github.com/state-spaces/mamba. 

### Run
To run SST on various dataset, run corrrsponidng dataset `.sh` files in the scripts folder. 

For exmaple, run SST on the Weather dataset: `./weather.sh`

### Dataset
You can download all the datasets from the "[Autoformer](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy)" project. Creatae a `dataset` folder in the current directory and put the downloaded datasets into `dataset` folder.

## Acknowledgement
We would like to greatly thank the following awesome projects:

Mamba (https://github.com/state-spaces/mamba)

PatchTST (https://github.com/yuqinie98/PatchTST)

LTSF-Linear (https://github.com/cure-lab/LTSF-Linear)

Autoformer (https://github.com/thuml/Autoformer)

## Cite
If you find this repository useful for your work, please consider citing the paper as follows:

```bibtex
@inproceedings{xu2025sst,
  title={SST: Multi-Scale Hybrid Mamba-Transformer Experts for Time Series Forecasting},
  author={Xu, Xiongxiao and Chen, Canyu and Liang, Yueqing and Huang, Baixiang and Bai, Guangji and Zhao, Liang and Shu, Kai},
  booktitle={Proceedings of the 34th ACM International Conference on Information and Knowledge Management},
  pages={3655--3665},
  year={2025}
}
```
