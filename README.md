# Dynamic Routing Between Capsules - PyTorch implementation

PyTorch implementation of NIPS 2017 paper [Dynamic Routing Between Capsules](https://arxiv.org/abs/1710.09829) from Sara Sabour, Nicholas Frosst and Geoffrey E. Hinton.

The hyperparameters and data augmentation strategy strictly follow the paper.


# Modification regarding the original fork:
* Modified to work with PyTorch 2.1.1 and Python 3.11
* The network is now organinzed into one torch.nn.Sequential
* python run.py
* I did not copy the jupyter-notebook. 
* The command line arguments are removed and replaced by parameters in run.py

|||
|---|---|
|batch_size|          input batch size for training (default: 128)|
|test-batch-size|      input batch size for testing (default: 1000)|
|epochs |              number of epochs to train (default: 250)|
|lr|                  learning rate (default: 0.001)|
|no_cuda|               disables CUDA training|
|seed                  |random seed (default: 1)|
|log_interval|         how many batches to wait before logging training status (default: 10)|
|routing_iterations|    number of iterations for routing algorithm (default: 3)|
|with_reconstruction|   should reconstruction layers be used|
  
MNIST dataset will be downloaded automatically.



