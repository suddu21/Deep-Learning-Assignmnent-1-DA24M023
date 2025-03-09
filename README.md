# DA6401 Intro to Deep Learning - Assignment 1
* [Click here for wandb report](https://wandb.ai/da24m023-indian-institute-of-technology-madras/fashion-mnist-nn-sweep/reports/Sudhanva-Satish-DA24M023-DA6401-Assignment-1--VmlldzoxMTY5ODY1OA?accessToken=uje0e7k9sa6p71hgv2i4q0empuhil9yzcb8wwai5e4i0fs2v3j5wlllazhzi796c)
* Github repo link - https://github.com/suddu21/Deep-Learning-Assignmnent-1-DA24M023
* 
## Instructions to train and run the model
* The driver file is train.py which can be run with the command
  
  ```
  python train.py
  ```
* It accepts all command line arguments as mentioned in the code specifications. If no arguments are passed, it defaults to the best parameters found which is detailed in the wandb report.
* For example, to run the model with tanh activation and 5 epochs

  ```
  python train.py --activation tanh --epochs 5
  ```
* When train.py is run, it does the following:-
  1. Loads the specified dataset (fashion-mnist by default)
  2. Trains the model with parsed arguments, if any. Defaults to pre-defined hyperparams if none.
  3. Logs the run to wandb along with training and validation metrics.
  4. Evaluates the model on the test dataset and logs testing accuracy.
  5. Creates a confusion matrix for test predictions, saves it locally and logs it to wandb.
* To run a hyperparameter sweep, use the CL argument --mode which takes 'single' and 'sweep' as values. Pass the value 'sweep' to run a wandb sweep.
  ```
  python train.py --mode sweep
  ```
1
### Supported parameters
