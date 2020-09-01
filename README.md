## Towards Physics-informed Deep Learning for Turbulent Flow Prediction
## Paper: 
Rui Wang, Karthik Kashinath, Mustafa Mustafa, Adrian Albert, Rose Yu [Towards Physics-informed Deep Learning for Turbulent Flow Prediction](https://ucsdml.github.io/jekyll/update/2020/08/23/TF-Net.html), KDD 2020

## [DataSet](https://drive.google.com/drive/folders/1VOtLjfAkCWJePiacoDxC-nrgCREKvrpE?usp=sharing.)
2000 velocity fields (![formula](https://render.githubusercontent.com/render/math?math=2000\times2\times256\times1792))

### Abstract:
While deep learning has shown tremendous success in a wide range of domains, it remains a grand challenge to incorporate physical principles in a systematic manner to the design, training, and inference of such models. In this paper, we aim to predict turbulent flow by learning its highly nonlinear dynamics from spatiotemporal velocity fields of large-scale fluid flow simulations of relevance to turbulence modeling and climate modeling. We adopt a hybrid approach by marrying two well-established turbulent flow simulation techniques with deep learning. Specifically, we introduce trainable spectral filters in a coupled model of Reynolds-averaged Navier-Stokes (RANS) and Large Eddy Simulation (LES), followed by a specialized U-net for prediction. Our approach, which we call turbulent-Flow Net (TF-Net), is grounded in a principled physics model, yet offers the flexibility of learned representations. We compare our model, TF-Net, with state-of-the-art baselines and observe significant reductions in error for predictions 60 frames ahead. Most importantly, our method predicts physical fields that obey desirable physical characteristics, such as conservation of mass, whilst faithfully emulating the turbulent kinetic energy field and spectrum, which are critical for accurate prediction of turbulent flows.

### Model Architecture
<img src="./model.png" width="700" height="300">


### Velocity U & V Prediction and Ablation Study
![](Videos/all.gif)


## Description
1. Baselines/: Six baseline modules included in the paper.
2. TF-Net/: 
   1. model.py: TF-net pytorch implementation.
   2. penalty.py: a few regularizers we have tried.
   3. train.py: data loaders, train epoch, validation epoch, test epoch functions.
   4. run_model.py: Scripts to train TF-Net
   ```
   python run_model.py
   ```
3. Evaluation/:
   1. Evaluation.ipynb: contains the functions of four evaluation metrics.
   2. radialProfile.py: a helper function for calculating energy spectrum.
4. Videos/: Videos of velocity u, v predictions and ablation study.

## Requirement 
* python 3.6
* pytorch 10.1
* matplotlib

## Cite
```
@article{Wang2020TF,
   title={Towards Physics-informed Deep Learning for Turbulent Flow Prediction},
   author={Rui Wang, Karthik Kashinath, Mustafa Mustafa, Adrian Albert, Rose Yu},
   journal={Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining},
   Year = {2020}
}	
```
