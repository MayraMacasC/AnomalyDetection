This is the reference implementation of the STAE-AD model presented in the paper "An Unsupervised Framework for Anomaly Detection in a Water Treatment System", 2019 18th IEEE International Conference On Machine Learning And Applications (ICMLA) held in Boca Raton, FL, USA, USA.

# An Unsupervised Framework for Anomaly Detection in a Water Treatment System 
Current Cyber-Physical Systems (CPSs) are sophisticated, complex, and equipped with networked sensors and actuators. As such, they have become further exposed to cyberattacks. Recent catastrophic events have demonstrated that standard, human-based management of anomaly detection in complex systems is not efficient enough and have underlined the significance of automated detection, intelligent and rapid response. Nevertheless, existing anomaly detection frameworks usually are not capable of dealing with the dynamic and complicated nature of the CPSs. In this study, we introduce an unsupervised framework for anomaly detection based on an Attention-based Spatio-Temporal Autoencoder. In particular, we first construct statistical correlation matrices to characterize the system status across different time steps. Next, a 2D convolutional encoder is employed to encode the patterns of the correlation matrices, whereas an Attention-based Convolutional LSTM Encoder-Decoder (ConvLSTM-ED) is used to capture the temporal dependencies. More precisely, we introduce an input attention mechanism to adaptively select the most significant input features at each time step. Finally, the 2D convolutional decoder reconstructs the correlation matrices. The differences between the reconstructed correlation matrices and the original ones are used as indicators of anomalies. Extensive experimental analysis on data collected from all six stages of Secure Water Treatment (SWaT) testbed, a scaled-down version of a real-world industrial water treatment plant, demonstrates that the proposed model outperforms the state-of-the-art baseline techniques.

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
├──  STAE-AD - deep learning module 
│    └── S_ModelsProyTS_Deploy.py - main model and its variants
│    └── VarConvLSTM.py Atention model process 
│
├──  AnomalyDetection - anomaly detection module
│    └── S_Distance_Matrix.py - diferents methods to calculate the distante between two matrices 
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


