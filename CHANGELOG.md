# Imbrium

## Released

### 0.1.0

- base library

### 0.1.2

- encoder-decoder predictor support for multistep multivariate time series

### 0.1.3

- added model loss variable type
- architecture layers can now be adjusted by defining number of neurons and
activation function type
- improved docstrings

### 0.1.4

- added keras native call-back feature to fit_model method
- minor docstring changes
- minor default value issues in fit_model method solved

### 0.1.5

- added getter methods for optimizer and model id
- added more unit-tests
- save_model method allows now to specify a customized path

### 0.1.6

- solved python version conflict by only allowing python 3.7.*

### 0.1.7

- improved source code format
- improved scaling selector mechanism

### 0.1.8

- major refactoring of code base
- minor name change from Imbrium to imbrium

### 1.0.0

- added tests for new utils module
- name changes of predictor classes:
  - univarstandard => univarpure
  - BasicMultStepUniVar => PureUni
  - univarhybrid => univarhybrid (unchanged)
  - HybridMultStepUniVar => HybridUni
  - multivarstandard => multivarpure
  - BasicMultSTepMultVar => PureMulti
  - multivarhybrid => multivarhybrid (unchanged)
  - HybridMultStepMultVar => HybridMulti
- tox added
- outsourced Binder demo notebook to https://github.com/maxmekiska/ImbriumTesting-Demo
- new README.md

### 1.0.1

- imbrium supports now:
  - python 3.7, 3.8, 3.9, 3.10

### 1.1.0

- removed batch_size parameter from fit_model method
- hyperparameter optimization added via the Optuna library


### 1.2.0

- added Tensorboard support
- changed show_performance plot to show loss and metric values
- added optional dropout and regularization layers to architectures

### 1.3.0

- added depth parameter to architectures
- added optimizer configuration support
- added optimizer configuration to seeker

### 2.0.0

- adapted `keras`
- removed internal hyperparameter tuning
- removed encoder-decoder architectures
- improved layer configuration via dictionary input
- split data argument into target and feature numpy arrays
 
### 2.0.1

- fix: removed dead pandas imports 
- chore: added tensorflow as base requirement
 
### 2.1.0

- feat!: removed data preparation out of predictor class, sub_seq, steps_past, steps_future need now to be defined in each model method
  - allows for advanced hyper parameter tuning
- fix: removed tensor board activation logic bug

### 3.0.0

- chore!: changed from temp library keras_core to keras > 3.0.0
- chore!: removed python 3.8 support to accomodate tensorflow and keras dependiencies
- chore: increased major to 3.0.0 to align with keras major
- feat: added evaluate_model method to test model performance on test data
- refactor!: removed validation split from `fit_model`. Control validation and test split via evaluation_split and validation_split paramters in class variables 