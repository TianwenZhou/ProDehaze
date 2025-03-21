## ProDehaze: Prompting Diffusion Models Toward Faithful Image Dehazing

[Paper](https://arxiv.org/abs/2308.10510)|[Project Page](https://zhoutianwen.com)

This is the official repo of ProDehaze: Prompting Diffusion Models Toward Faithful Image Dehazing by Pytorch.
<img src="asset/overview.png" alt="show" style="zoom:90%;" />

News: Our paper has been accepted to ICME 2025!

## Getting started
### Dependency Installation
```shell
# git clone this repository
git clone https://github.com/TianwenZhou/ProDehaze.git
cd ProDehaze

# Create the conda environment 
conda env create --file environment.yaml
conda activate prodehaze

# Install taming & clip
pip install -e git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers
pip install -e git+https://github.com/openai/CLIP.git@main#egg=clip
pip install -e .
```

### Data Prepare

Download train/eval data from the following links:

Training: [*RESIDE*](https://sites.google.com/view/reside-dehaze-datasets/reside-v0)

Testing:
[*I-Haze*](https://data.vision.ee.ethz.ch/cvl/ntire18//i-haze/#:~:text=To%20overcome%20this%20issue%20we%20introduce%20I-HAZE%2C%20a,real%20haze%20produced%20by%20a%20professional%20haze%20machine.) / 
[*O-Haze*](https://data.vision.ee.ethz.ch/cvl/ntire18/o-haze/) /
[*Dense-Haze*](https://arxiv.org/abs/1904.02904#:~:text=To%20address%20this%20limitation%2C%20we%20introduce%20Dense-Haze%20-,introducing%20real%20haze%2C%20generated%20by%20professional%20haze%20machines.) /
[*Nh-Haze*](https://data.vision.ee.ethz.ch/cvl/ntire20/nh-haze/) /
[*RTTS*](https://sites.google.com/view/reside-dehaze-datasets/reside-standard?authuser=0) 

## Pretrained Model

We prepared the pretrained model at:

| Type                                                        | Weights                                        |
| ----------------------------------------------------------- | ------------------------------------------------------------ |
| SPR                                                 | [OneDrive](https://1drv.ms/u/s!AsqtTP8eWS-penA8AqrU8c_I4jU) |
| HCR                                                 | [OneDrive](https://1drv.ms/u/s!AsqtTP8eWS-penA8AqrU8c_I4jU) |