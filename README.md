# An Unsupervised Framework for Anomaly Detection in a Water Treatment System 
In this framework, we used the Statistical Correlation Analysis module to characterize the system status at different time steps.  The generated correlation matrices are fed into the core module of the framework - the one with the attention-based Spatio-Temporal Autoencoder  for anomaly detection (STAE-AD). In greater detail, the convolutional encoder is employed to learn the spatial information that is hidden in the correlation matrices, whereas ConvLSTM captures the temporal dependencies. In addition, the soft-Attention technique  is applied in ConvLSTM. Specifically, we introduce an input attention mechanism to select the most relevant input features adaptively. Finally, the convolutional decoder reconstructs the correlation matrices and uses a mean square loss to perform end-to-end learning.


# Matrix Correlation
We use Bayesian Pearson Correlation Analysis in order to generate the statistical correlation matrices

* Bayesian Pearson Correlation Analysis in [AnomalyDetection/Matrix Correlation/BayesCorrelation.py](https://github.com/AlexandraM1011/AnomalyDetection/blob/master/Matrix%20Correlation/BayesCorrelation.py)

* Genereting the statistical correlation matrices in [AnomalyDetection/Matrix Correlation/MatrixProyOneWinProduction.py](https://github.com/AlexandraM1011/AnomalyDetection/blob/master/Matrix%20Correlation/MatrixProyOneWinProduction.py)

# Attention-based Spatio-Temporal Autoencoder (STAE-AD)
* The STAE-AD model and its variants (CNN-ED, CNNED + ConvLSTM-ED) are trained in [AnomalyDetection/STAE-AD/S_NormalTrain_ProyTS.py](https://github.com/AlexandraM1011/AnomalyDetection/blob/master/STAE-AD/S_NormalTrain_ProyTS.py)
* The models are evaluated and the anomaly detection is performed in [AnomalyDetection/Anomaly Detection/S_Evaluation_ProyTS.py](https://github.com/AlexandraM1011/AnomalyDetection/blob/master/Anomaly%20Detection/S_Evaluation_ProyTS.py)



## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
