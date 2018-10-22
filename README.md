# DQN

The DQN algorithm used for solving Gym's cartpole environment.    
I changed the reward function:    
```
reward = np.cos(2*next_state[3]) 
```
The huber loss is implemented with gradient clipping and learning rate decay. Also layer normalization is applied form the paper: 
- *Layer Normalization* Lei Ba et al. [pdf](https://arxiv.org/abs/1607.06450.)

## Requirements

- Tensorflow  
- Numpy   
- Gym 

## Run 

There is a constant: 
```
DEVICE = '/gpu:0'
```
Set it to '/cpu:0' if you don't have one. 

And then run as: 

```
$ python main.py

```

You can see how the loss and the learning rate evolve over time with tensorboard accesing: 
```
tensorboard --logdir=train

```
