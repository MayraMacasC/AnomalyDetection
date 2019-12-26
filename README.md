# An Unsupervised Framework for Anomaly Detection in a Water Treatment System 
Current Cyber-Physical Systems (CPSs) are sophisticated, complex, and equipped with networked sensors and actuators. As such, they have become further exposed to cyberattacks. Recent catastrophic events have demonstrated that standard, human-based management of anomaly detection in complex systems is not efficient enough and have underlined the significance of automated detection, intelligent and rapid response. Nevertheless, existing anomaly detection frameworks usually are not capable of dealing with the dynamic and complicated nature of the CPSs. In this study, we introduce an unsupervised framework for anomaly detection based on an Attention-based Spatio-Temporal Autoencoder. In particular, we first construct statistical correlation matrices to characterize the system status across different time steps. Next, a 2D convolutional encoder is employed to encode the patterns of the correlation matrices, whereas an Attention-based Convolutional LSTM Encoder-Decoder (ConvLSTM-ED) is used to capture the temporal dependencies. More precisely, we introduce an input attention mechanism to adaptively select the most significant input features at each time step. Finally, the 2D convolutional decoder reconstructs the correlation matrices. The differences between the reconstructed correlation matrices and the original ones are used as indicators of anomalies. Extensive experimental analysis on data collected from all six stages of Secure Water Treatment (SWaT) testbed, a scaled-down version of a real-world industrial water treatment plant, demonstrates that the proposed model outperforms the state-of-the-art baseline techniques.

---
## Framework  
<p align="center">
<img src="https://github.com/MayraMacasC/AnomalyDetection/blob/master/Framework.png" width="500" height="250">
</p>
Figure 1 shows a high-level overview of the framework. We begin with a set of historical time series data, correctly curated, as input. Next, we use the Statistical Correlation Analysis module to characterize the system status at different time steps. The generated correlation matrices are fed into the core module of the framework - the one with the attention-based Spatio-Temporal Autoencoder for anomaly detection (STAEAD). In greater detail, the convolutional encoder is employed to learn the spatial information that is hidden in the correlation matrices, whereas ConvLSTM captures the temporal dependencies. In addition, the soft-Attention technique is applied in ConvLSTM. Specifically, we introduce an input attention mechanism to select the most relevant input features adaptively. Finally, the convolutional decoder reconstructs the correlation matrices and uses the mean square loss function to perform end-to-end learning.
---

## Requirements
- Python 3.5.6
- TensorFlow framework version 1.11.0 
- Package PyMC3 (That allowed fitting the Bayesian model using Markov Chain Monte Carlo (MCMC))
---

## Table Of Contents
-  [In Details](#in-details)
-  [Contributing](#contributing)
-  [Acknowledgments](#acknowledgments)
- [License](#license)

---

## In Details
```
├── Statistical Correlation Analysis - statistical Correlation Analysis module to characterize the system status at different time steps
│    └── BayesCorrelation.py - this file contains the Bayes Correlation process 
│    └── MatrixProyOneWinProduction.py - this file allows to construct a m × m correlation matrix based on the Bayes Pearson correlation analysis, where m is the number of time series.
│ 
│
├──  STAE-AD - deep learning module 
│    └── S_ModelsProyTS_Deploy.py - main model and its variants
│    └── S_NormalTrain_ProyTS.py - this file containts the process to train the main model and its variants 
│    └── S_Parameters_ProyTS.py - parameter configuration
│    └── S_ProcessingErrorMatrix.py - processing anomaly detection error
│    └── VarConvLSTM.py Atention model process 
│
├──  AnomalyDetection - anomaly detection module
│    └── S_Define_Th_Distance_ProyTS.py - preliminar process to calculate the distance between matrices 
│    └── S_Distance_Matrix.py - diferents methods to calculate the distante between two matrices 
|    └── S_Evaluation_ProyTS.py - metrics evaluation (to threshold)
│
│
├── Helper      
│    └── [S_DataPrepartion_ProyTS.py] - data preparation to split data
│    └── [S_DataProcessingProyTS.py] - split data (training/validation/evaluation)
|    └── [S_DataProcessingEvaluation_ProyTS.py]  - metrics evaluation (anamaly detection process)
```
---

## Contributing
Any kind of enhancement or contribution is welcomed.

---

## Acknowledgments
The authors would like to thank iTrust, Center for Research in Cyber Security, Singapore University of Technology and Design for providing the SWaT dataset.

---

## License
[![License](http://img.shields.io/:license-mit-blue.svg?style=flat-square)](http://badges.mit-license.org)

- **[MIT license](https://github.com/MayraMacasC/AnomalyDetection/blob/master/LICENSE.md)**


