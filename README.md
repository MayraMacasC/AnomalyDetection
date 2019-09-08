# An Unsupervised Framework for Anomaly Detection in a Water Treatment System 
In this framework, we used the Statistical Correlation Analysis module to characterize the system status at different time steps.  The generated correlation matrices are fed into the core module of the framework - the one with the attentionbased Spatio-Temporal Autoencoder  for anomaly detection (STAE-AD). In greater detail, the convolutional encoder is employed to learn the spatial information that is hidden in the correlation matrices, whereas ConvLSTM captures the temporal dependencies. In addition, the soft-Attention technique  is applied in ConvLSTM. Specifically, we introduce input attention mechanism to select the most relevant input features adaptively. Finally, the convolutional decoder reconstructs the correlation matrices and use a mean square loss to perform end-to-end learning.


# Matrix Correlation
We use Bayesian Pearson Correlation Analysis in order to generate the statistical correlation matrices

* Bayes correlation in [a link](AnomalyDetection/Matrix Correlation/BayesCorrelation.py)
* Genereting the statistical correlation matrices in 

# Attention-based Spatio-Temporal Autoencoder
