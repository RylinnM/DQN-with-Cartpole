# DQN-with-Cartpole
A Deep Q-learning Network that works in the Cartpole environment.

To run the code make sure the gym environment is installed by using:

'pip install gym' and 'pip install gym[all]'

After everything is set up correctly the code can be executed with the following command:

python Cartpole.py --[implementation] [hyperparameter1] .... [hyperparameterN] 

Here 'implementation' can be either two things:

1. 'ablation' to run our ablation study. No extra hyperparameters can be given.
   
An example would be:

'python Cartpole.py --ablation' 

2. 'tune' To for tuning of hyperparameters. Each hyperparameter you want to tune can be given separately.


The hyperparameters that can be tuned consist of 'layer', 'unit', 'optimizer', 'batch_size', 'epochs', 'policy', and 'gamma'.
An example would be:

'python Cartpole.py --tune layer optimizer batch_size'

or 

'python Cartpole.py --tune gamma'

After the code is run, the figures will be created in the current folder.
