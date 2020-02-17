# Human Activity Recognition
Following on from [this tutorial](https://machinelearningmastery.com/cnn-models-for-human-activity-recognition-time-series-classification/),
this contains a 1-dimensional convolutional neural network implemented in PyTorch.
It is a first pass with limited optimiztion of hyperparameters or architecture exploration. 

The package is organized with a split between data processing and model training. 
If more architectures are used, each architecture can be structured in a module, separate from the training methods. 
This allows for complex architectures that subclass pytorch modules (e.g. multi-headed CNNs).   

Two notebooks are given:
 1. `example_run`: basic run functionality and training visualization 
 2. `coarse_optimization`: Grid search optimization done manually to explore the hyperparameters in the fixed architecture.
 
The first notebook is mirrored in a script.
   
For more information regrading the dataset see [UCI](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones).


### Potential future considerations
* Bayesian optimization could be used to optimize the architecture and hyperarameters against the test accuracy. 
* Using the object oriented model development, and ModuleLists, CNN architectures with an increasing number of filters 
can be easily described by a set of metaparameters that are optimized to yield model hyperparameters 
(e.g. optimizing the initial filters with a decay rate, can describe a large set of models independent of the number of layers.)
* For an example of this style of optimization, see [BOML.](https://github.com/CooperComputationalCaucus/BOML)
* A multi-headed model or ensemble model can easily be developed as a subclass of the current implementation. 
* For brevity (and running on a laptop cpu), experiments were run for 3 iterations. 
This number could be increased if large variance across runs was noticed. 
* Lastly, a greater variety of architectures should be explored (1D analogues to VGG, Inception, etc..)
 