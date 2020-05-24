## Towards Physics-informed Deep Learning for Turbulent Flow Prediction
### Paper: 
[Towards Physics-informed Deep Learning for Turbulent Flow Prediction](https://arxiv.org/abs/1911.08655)

### Abstract:
While deep learning has shown tremendous success in a wide range of domains, it remains a grand challenge to incorporate physical principles in a systematic manner to the design, training, and inference of such models. In this paper, we aim to predict turbulent flow by learning its highly nonlinear dynamics from spatiotemporal velocity fields of large-scale fluid flow simulations of relevance to turbulence modeling and climate modeling. We adopt a hybrid approach by marrying two well-established turbulent flow simulation techniques with deep learning. Specifically, we introduce trainable spectral filters in a coupled model of Reynolds-averaged Navier-Stokes (RANS) and Large Eddy Simulation (LES), followed by a specialized U-net for prediction. Our approach, which we call turbulent-Flow Net (TF-Net), is grounded in a principled physics model, yet offers the flexibility of learned representations. We compare our model, TF-Net, with state-of-the-art baselines and observe significant reductions in error for predictions 60 frames ahead. Most importantly, our method predicts physical fields that obey desirable physical characteristics, such as conservation of mass, whilst faithfully emulating the turbulent kinetic energy field and spectrum, which are critical for accurate prediction of turbulent flows.

### Model Architecture
![Alt text](model.png?raw=true "Title")

### Velocity U predictions by TF-net and three best baselines.
s[![](https://img.youtube.com/vi/80U8lcIZYe4/hqdefault.jpg)](https://www.youtube.com/watch?v=80U8lcIZYe4)

### Ablation Study
#####(includes the predictions of TF-net, and the outputs of each small U-net while the other two encoders
are zeroed out.)
s[![](https://img.youtube.com/vi/ysdrMUfdhe0/hqdefault.jpg)](https://www.youtube.com/watch?v=ysdrMUfdhe0)
