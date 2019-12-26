# An Unsupervised Framework for Anomaly Detection in a Water Treatment System 
Current Cyber-Physical Systems (CPSs) are sophisticated, complex, and equipped with networked sensors and actuators. As such, they have become further exposed to cyberattacks. Recent catastrophic events have demonstrated that standard, human-based management of anomaly detection in complex systems is not efficient enough and have underlined the significance of automated detection, intelligent and rapid response. Nevertheless, existing anomaly detection frameworks usually are not capable of dealing with the dynamic and complicated nature of the CPSs. In this study, we introduce an unsupervised framework for anomaly detection based on an Attention-based Spatio-Temporal Autoencoder. In particular, we first construct statistical correlation matrices to characterize the system status across different time steps. Next, a 2D convolutional encoder is employed to encode the patterns of the correlation matrices, whereas an Attention-based Convolutional LSTM Encoder-Decoder (ConvLSTM-ED) is used to capture the temporal dependencies. More precisely, we introduce an input attention mechanism to adaptively select the most significant input features at each time step. Finally, the 2D convolutional decoder reconstructs the correlation matrices. The differences between the reconstructed correlation matrices and the original ones are used as indicators of anomalies. Extensive experimental analysis on data collected from all six stages of Secure Water Treatment (SWaT) testbed, a scaled-down version of a real-world industrial water treatment plant, demonstrates that the proposed model outperforms the state-of-the-art baseline techniques.
## Framework  
<p align="center">
<img src="https://github.com/MayraMacasC/AnomalyDetection/blob/master/Framework.png" width="500" height="250">
</p>
Figure 1 shows a high-level overview of the framework. We begin with a set of historical time series data, correctly curated, as input. Next, we use the Statistical Correlation Analysis module to characterize the system status at different time steps. The generated correlation matrices are fed into the core module of the framework - the one with the attention-based Spatio-Temporal Autoencoder for anomaly detection (STAEAD). In greater detail, the convolutional encoder is employed to learn the spatial information that is hidden in the correlation matrices, whereas ConvLSTM captures the temporal dependencies. In addition, the soft-Attention technique is applied in ConvLSTM. Specifically, we introduce an input attention mechanism to select the most relevant input features adaptively. Finally, the convolutional decoder reconstructs the correlation matrices and uses the mean square loss function to perform end-to-end learning.

## Requirements
- Python 3.5.6
- TensorFlow framework version 1.11.0 
- Package PyMC3 (That allowed fitting the Bayesian model using Markov Chain Monte Carlo (MCMC))

## Table Of Contents
-  [In a Nutshell](#in-a-nutshell)
-  [In Details](#in-details)
-  [Contributing](#contributing)
-  [Acknowledgments](#acknowledgments)
-  [License] (#license)

## In a Nutshell   
In a nutshell here's how to use this template, so **for example** assume you want to implement ResNet-18 to train mnist, so you should do the following:
- In `modeling`  folder create a python file named whatever you like, here we named it `example_model.py` . In `modeling/__init__.py` file, you can build a function named `build_model` to call your model

```python
from .example_model import ResNet18

def build_model(cfg):
    model = ResNet18(cfg.MODEL.NUM_CLASSES)
    return model
``` 

   
- In `engine`  folder create a model trainer function and inference function. In trainer function, you need to write the logic of the training process, you can use some third-party library to decrease the repeated stuff.

```python
# trainer
def do_train(cfg, model, train_loader, val_loader, optimizer, scheduler, loss_fn):
 """
 implement the logic of epoch:
 -loop on the number of iterations in the config and call the train step
 -add any summaries you want using the summary
 """
pass

# inference
def inference(cfg, model, val_loader):
"""
implement the logic of the train step
- run the tensorflow session
- return any metrics you need to summarize
 """
pass
```

- In `tools`  folder, you create the `train.py` .  In this file, you need to get the instances of the following objects "Model",  "DataLoader”, “Optimizer”, and config
```python
# create instance of the model you want
model = build_model(cfg)

# create your data generator
train_loader = make_data_loader(cfg, is_train=True)
val_loader = make_data_loader(cfg, is_train=False)

# create your model optimizer
optimizer = make_optimizer(cfg, model)
```

- Pass the all these objects to the function `do_train` , and start your training
```python
# here you train your model
do_train(cfg, model, train_loader, val_loader, optimizer, None, F.cross_entropy)
```

**You will find a template file and a simple example in the model and trainer folder that shows you how to try your first model simply.**


## In Details
```
├── Statistical Correlation Analysis - statistical Correlation Analysis module to characterize the system status at different time steps
│    └── <a href="https://github.com/MayraMacasC/AnomalyDetection/blob/master/Statistical%20Correlation%20Analysis/BayesCorrelation.py" title="BayesCorrelation.py">BayesCorrelation.py</a> - this file contains the bayes correction process 
│    └── MatrixProyOneWinProduction.py  - this file allows to create new construct a m × m correlation matrix based on the Bayes Pearson correlation analysis, where m is the number of time series
│ 
│
├──  STAE-AD 
│    └── S_ModelsProyTS_Deploy.py  - here's the datasets folder that is responsible for all data handling.
│    └── S_NormalTrain_ProyTS.py - here's the data preprocess folder that is responsible for all data augmentation.
│    └── S_Parameters_ProyTS.py 		   - here's the file to make dataloader.
│    └── S_ProcessingErrorMatrix.py  - here's the file that is responsible for merges a list of samples to form a mini-batch.
│    └── VarConvLSTM.py
│
├──  AnomalyDetection
│   ├── S_Define_Th_Distance_ProyTS.py    - this file contains the train loops.
│   └── S_Distance_Matrix.py   - this file contains the inference process.
|   └── S_Evaluation_ProyTS.py   - this file contains the inference process.
│
│
├── Helper              - this folder contains any customed layers of your project.
│   └── S_DataPrepartion_ProyTS.py
│   └── S_DataProcessingEvaluation_ProyTS.py
|   └── S_DataProcessingProyTS.py
```

## Contributing
Any kind of enhancement or contribution is welcomed.


## Acknowledgments
The authors would like to thank iTrust, Center for Research in Cyber Security, Singapore University of Technology and Design for providing the SWaT dataset.
## License

