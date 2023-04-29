#!/usr/bin/env python3
from __future__ import print_function


import gym
import keras
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
import itertools
import sys

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, LeakyReLU
from keras import backend as K
from collections import deque
from keras.optimizers import RMSprop, Adam


import time
from scipy.signal import savgol_filter
import pandas as pd

#GPU check
import tensorflow as tf
matplotlib.use('Agg')

if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")



#Give model
def Model(inDim, outDim, hypertune, layer_count):
    # default model
    if not hypertune:
        model = keras.models.Sequential()
        model.add(Dense(units=64, input_dim=inDim, activation='linear'))
        model.add(LeakyReLU(alpha=0.05))
        model.add(Dense(units=128, activation='linear'))
        model.add(LeakyReLU(alpha=0.05))
        model.add(Dense(units=outDim, activation='linear'))

        model.compile(loss='mse',
                      optimizer=Adam(learning_rate=0.001),
                      metrics=['accuracy'])
    else:
        model = HyperTuning.model_layers(layer_count)

    return model


class Tool:
    # Computes softmax
    def softmax(x, temp):
        # scale by temperature
        x = x / temp
        # subtract max to prevent overflow of softmax
        z = x - np.max(x)
        # compute softmax
        return np.exp(z) / np.sum(np.exp(z))

    def smooth(y, window, poly=2):
        '''
        y: vector to be smoothed
        window: size of the smoothing window '''
        return savgol_filter(y, window, poly)



#Deep Q Learning Agent
class DQLAgent:
    def __init__(self, env, layer, unit, optimizer):

        #amount of in/output states
        self.n_actions = env.action_space.n
        self.n_states = env.observation_space.shape[0]

        # Create Network
        # modified here so that the DQL agent can take in the hyperparameters and pass them to the model

        self.inDim = len(env.observation_space.low)
        self.outDim = self.n_actions  # self.env.action_space.n
        if unit == None and optimizer == None:
            self.model = HyperTuning(env).model_layers(layer)
            self.model_t = HyperTuning(env).model_layers(layer)
        elif optimizer == None:
            self.model = HyperTuning(env).model_first_layer_units(unit)
            self.model_t = HyperTuning(env).model_first_layer_units(unit)
        else:
            self.model = HyperTuning(env).model_optimizer(optimizer)
            self.model_t = HyperTuning(env).model_optimizer(optimizer)

        #self.model = Model(self.inDim, self.outDim)
        #self.model_t = Model(self.inDim, self.outDim)
        self.model_t.set_weights(self.model.get_weights())

        self.model.save_weights('model.h5')

    def ResetWeights(self):
        self.model.load_weights('model.h5')
        self.model_t.load_weights('model.h5')
        

    def Tupdate(self):
        self.model_t.set_weights(self.model.get_weights())

    def Tpredict(self, observation):
        return self.model_t.predict(observation, verbose=0)

    def predict(self, observation):
        return self.model.predict(observation, verbose=0)

    def update(self, observation, target):
        observation = tf.convert_to_tensor(observation, dtype=tf.float32)
        self.model.fit(observation, target, verbose=0)

    def replay(self, memory, size, targetNetwork ,gamma=0.9, ):
        if len(memory) < size:
            return

        batch = random.sample(memory, size)
        
        #Create NN input
        observation = np.zeros((size, self.n_states))
        nextObservation = np.zeros((size, self.n_states))


        action, reward, terminated = [], [], []
        for i in range(size):
            observation[i] = batch[i][0]
            action.append(batch[i][1])
            nextObservation[i] = batch[i][2]
            reward.append(batch[i][3])
            terminated.append(batch[i][4])

        #Predict output
        q_values = self.predict(observation)

        if targetNetwork:
            next_q_values = self.Tpredict(nextObservation)

        else:
            next_q_values = self.predict(nextObservation)

        #Calculate targets
        for i in range(size):
            if terminated[i]:
                q_values[i][action[i]] = reward[i]
            else:
                #Bootstrap Q-value
                q_values[i][action[i]] = reward[i] + gamma * np.amax(next_q_values[i]).item()

        #train network
        self.update(observation, q_values)

        if targetNetwork:
            self.Tupdate()


class HyperTuning:
    # HT class, used for hyperparameter tuning
    def __init__(self,env):
        self.HT = True
        self.layers = [1, 2, 3] # number of hidden layers, NOT the total number of layers
        self.units = [16, 32, 64] # number of units in the input layer
        self.optimizer = ['Adam(learning_rate=0.001)', 'RMSprop(learning_rate=0.001)', 'Adam(learning_rate=0.0002)',
                          'RMSprop(learning_rate=0.0002)', 'Adam(learning_rate=0.005)', 'RMSprop(learning_rate=0.005)']
        self.batch_size = [32, 64, 128]
        self.epochs = [10, 20, 30]
        self.policy = ['egreedy', 'softmax']
        self.gamma = [0.9, 0.95, 0.99]
        self.epsilon = [0.1, 0.2, 0.3]
        self.temp = [0.1, 0.5, 1]
        self.input_dim = len(env.observation_space.low)
        self.output_dim = env.action_space.n

    def model_layers(self, layer_count):
        # create a model with the given number of layers, the default number of units in the input layer is 32
        units = self.units[1]  # default value

        model = keras.models.Sequential()
        model.add(Dense(units=units, input_dim=self.input_dim, activation='linear')) # input layer


        if layer_count == 1:
            model.add(Dense(units=units * 2, activation='linear')) # 1 hidden layer
            model.add(LeakyReLU(alpha=0.05))

        elif layer_count == 2:
            model.add(Dense(units=units * 2, activation='linear'))
            model.add(LeakyReLU(alpha=0.05))
            model.add(Dense(units=units * 4, activation='linear')) # 2 hidden layers
            model.add(LeakyReLU(alpha=0.05))

        elif layer_count == 3:
            model.add(Dense(units=units * 2, activation='linear'))
            model.add(LeakyReLU(alpha=0.05))
            model.add(Dense(units=units * 4, activation='linear'))
            model.add(LeakyReLU(alpha=0.05))
            model.add(Dense(units=units * 2, activation='linear')) # 3 hidden layers
            model.add(LeakyReLU(alpha=0.05))

        model.add(Dense(units=self.output_dim, activation='linear'))  # output layer

        model.compile(loss='mse',
                      optimizer=Adam(learning_rate=0.001),
                      metrics=['accuracy'])

        return model

    def model_first_layer_units(self,unit_count):
        # create a model with the given number of units in the input layer, the default number of layers is 3
        model = keras.models.Sequential()

        model.add(Dense(units=unit_count, input_dim=self.input_dim, activation='linear'))
        model.add(LeakyReLU(alpha=0.05))
        model.add(Dense(units=unit_count * 2, activation='linear'))
        model.add(LeakyReLU(alpha=0.05))
        model.add(Dense(units=unit_count * 4, activation='linear'))
        model.add(LeakyReLU(alpha=0.05))
        model.add(Dense(units=unit_count * 2, activation='linear'))
        model.add(LeakyReLU(alpha=0.05))
        model.add(Dense(units=self.output_dim, activation='linear'))

        model.compile(loss='mse',
                      optimizer=Adam(learning_rate=0.001),
                      metrics=['accuracy'])

        return model

    def model_optimizer(self,optimizer):
        # create a model with the given optimizer, the default number of layers is 3 and the default number of units in the input layer is 32
            model = keras.models.Sequential()

            model.add(Dense(units=32, input_dim=self.input_dim, activation='linear'))
            model.add(LeakyReLU(alpha=0.05))
            model.add(Dense(units=64, activation='linear'))
            model.add(LeakyReLU(alpha=0.05))
            model.add(Dense(units=128, activation='linear'))
            model.add(LeakyReLU(alpha=0.05))
            model.add(Dense(units=64, activation='linear'))
            model.add(LeakyReLU(alpha=0.05))
            model.add(Dense(units=self.output_dim, activation='linear'))

            model.compile(loss='mse',
                        optimizer=optimizer,
                        metrics=['accuracy'])

            return model

def Ablation(env):
    # hyperparameter tuning experiments
    fig, ax = plt.subplots()
    

    episodes = 100

    env.close()
    env = gym.make("CartPole-v1")
    DQL = DQLAgent(env, 3, 16, Adam(learning_rate=0.005))
    TotalRewards = Q_learn(env, DQL, episodes, gamma=.99, replay=False, targetNetwork=False, policy='egreedy',
            epsilon=0.1, ablation=True)
    TotalRewards = Tool.smooth(TotalRewards, 101)
    DQL.ResetWeights()
    ax.plot(TotalRewards, label=f"DQN")
    env.reset()
    del DQL

    env.close()
    env = gym.make("CartPole-v1")
    DQL = DQLAgent(env, 3, 16, Adam(learning_rate=0.005))
    TotalRewards = Q_learn(env, DQL, episodes, gamma=.99, replay=True,  replaySize = 128, targetNetwork=False,  n_update = 20,  policy='egreedy',
            epsilon=0.1, ablation=True)
    TotalRewards = Tool.smooth(TotalRewards, 101)
    DQL.ResetWeights()
    ax.plot(TotalRewards, label=f"ER=True_TN=False")
    env.reset()
    del DQL

    env.close()
    env = gym.make("CartPole-v1")
    DQL = DQLAgent(env, 3, 16, Adam(learning_rate=0.005))

    TotalRewards = Q_learn(env, DQL, episodes, gamma=.99, replay=False,  replaySize = 128, targetNetwork=True,  n_update = 20,  policy='egreedy',
            epsilon=0.1, ablation=True)
    DQL.ResetWeights()
    TotalRewards = Tool.smooth(TotalRewards, 101)
    ax.plot(TotalRewards, label=f"ER=False_TN=True")
    env.reset()
    del DQL

    env.close()
    env = gym.make("CartPole-v1")
    DQL = DQLAgent(env, 3, 16, Adam(learning_rate=0.005))
    TotalRewards = Q_learn(env, DQL, episodes, gamma=.99, replay=True,  replaySize = 128, targetNetwork=True,  n_update = 20,  policy='egreedy',
            epsilon=0.1, ablation=True)
    DQL.ResetWeights()
    TotalRewards = Tool.smooth(TotalRewards, 101)
    ax.plot(TotalRewards, label=f"ER=True_TN=True")
    env.reset()
    del DQL


    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title("Comparison of Base Model and Expanded Model")
    ax.legend()
    plt.savefig('./ER-TN.png')


def experiment(env, episodes, command):
    # hyperparameter tuning experiments
    fig, ax = plt.subplots()

    hyperparams = HyperTuning(env)
    if command == 'layer':
        # This will test three configurations of hidden layers, with 1, 2, and 3 hidden layers respectively
        for layer_count in hyperparams.layers:
            env.close()
            env = gym.make("CartPole-v1")
            print(f"Layer Count: {layer_count}")

            DQL = DQLAgent(env, layer_count, None, None)
            TotalRewards = Q_learn(env, DQL, episodes, gamma=.9, replay=True, targetNetwork=True, policy='egreedy',
                                   epsilon=0.3, temp=1)
            TotalRewards = Tool.smooth(TotalRewards, 5)

            ax.plot(TotalRewards, label=f"Layer Count: {layer_count}")
            DQL.ResetWeights()
            env.reset()
            del DQL

        ax.set_xlabel("Episode")
        ax.set_ylabel("Total Reward")
        ax.set_title("Comparison of Models with Different Hidden Layer Counts")
        ax.legend()
        plt.savefig('./layercount.png')




    elif command == 'unit':
        # This will test three configurations of units in the input layer, with 16, 32, and 64 units respectively
        for unit_count in hyperparams.units:
            env.close()
            env = gym.make("CartPole-v1")
            print(f"First Layer Unit Count: {unit_count}")

            DQL = DQLAgent(env, 3, unit_count, None)
            TotalRewards = Q_learn(env, DQL, episodes, gamma=.9, replay=True, targetNetwork=True, policy='egreedy',
                               epsilon=0.3, temp=1)
            TotalRewards = Tool.smooth(TotalRewards, 5)
            avg_reward = np.mean(TotalRewards)
            DQL.ResetWeights()
            ax.plot(TotalRewards, label=f"First layer unit count: {unit_count}")
            env.reset()

            del DQL
            #color = np.random.rand(3, )
            #ax.axhline(y=avg_reward, linestyle='-', label=f"Average Reward of model with {unit_count} units",
                       #color=color)


        ax.set_xlabel("Episode")
        ax.set_ylabel("Total Reward")
        ax.set_title("Comparison of Models with Different First Hidden Layer Unit Counts")
        ax.legend()
        plt.savefig('./unitcount.png')



    elif command == 'optimizer':
        # This will test three configurations of optimizers, with Adam, RMSprop containing different LR respectively
        for optimizer in hyperparams.optimizer:
            env.close()
            env = gym.make("CartPole-v1")
            optimizer_t = eval(optimizer)
            print(f"Optimizer: {str(optimizer)}")

            DQL = DQLAgent(env, 3, 64, optimizer_t)
            TotalRewards = Q_learn(env, DQL, episodes, gamma=.9, replay=True, targetNetwork=True, policy='egreedy',
                               epsilon=0.3, temp=1)
            TotalRewards = Tool.smooth(TotalRewards, 5)
            DQL.ResetWeights()
            ax.plot(TotalRewards, label=f"Optimizer: {optimizer}")
            env.reset()
            del DQL


        ax.set_xlabel("Episode")
        ax.set_ylabel("Total Reward")
        ax.set_title("Comparison of Models with Different Optimizers")
        ax.legend()
        plt.savefig('./optimizer.png')


    elif command == 'batch_size':
        # This will test three configurations of batch_sizes with [32, 64, 128] different sizes respectively
        for batch_size in hyperparams.batch_size:
            env.close()
            env = gym.make("CartPole-v1")
            print(f"batch_size: {batch_size}")

            DQL = DQLAgent(env, 3, 64, Adam(learning_rate=0.005))
            TotalRewards = Q_learn(env, DQL, episodes, gamma=.9, replay=True, replaySize = batch_size, targetNetwork=True, policy='egreedy',
                               epsilon=0.3, temp=1)
            TotalRewards = Tool.smooth(TotalRewards, 5)
            DQL.ResetWeights()
            ax.plot(TotalRewards, label=f"batch_size: {batch_size}")
            env.reset()
            del DQL


        ax.set_xlabel("Episode")
        ax.set_ylabel("Total Reward")
        ax.set_title("Comparison of Models with batch_sizes")
        ax.legend()
        plt.savefig('./batch_size.png')


    elif command == 'epochs':
        # This will test three configurations of epochs with [10, 20, 30] different sizes respectively
        for epochs in hyperparams.epochs:
            env.close()
            env = gym.make("CartPole-v1")
            print(f"epochs: {epochs}")

            DQL = DQLAgent(env, 3, 64, Adam(learning_rate=0.005))
            TotalRewards = Q_learn(env, DQL, episodes, gamma=.9, replay=True,replaySize = 128 , targetNetwork=True, n_update = epochs, policy='egreedy',
                               epsilon=0.3, temp=1)
            TotalRewards = Tool.smooth(TotalRewards, 5)
            DQL.ResetWeights()
            ax.plot(TotalRewards, label=f"epochs: {epochs}")
            env.reset()
            del DQL


        ax.set_xlabel("Episode")
        ax.set_ylabel("Total Reward")
        ax.set_title("Comparison of Models with different epochs before updating TN")
        ax.legend()
        plt.savefig('./epochs.png')




    elif command == 'gamma':
        # This will test three configurations of gamma with [0.9, 0.95, 0.99] different sizes respectively
        for gamma in hyperparams.gamma:
            env.close()
            env = gym.make("CartPole-v1")
            print(f"gamma: {gamma}")

            DQL = DQLAgent(env, 3, 64, Adam(learning_rate=0.005))
            TotalRewards = Q_learn(env, DQL, episodes, gamma=gamma, replay=True, replaySize = 128 , targetNetwork=True, n_update = 20, policy='egreedy',
                               epsilon=0.3, temp=1)
            TotalRewards = Tool.smooth(TotalRewards, 5)
            DQL.ResetWeights()
            ax.plot(TotalRewards, label=f"gamma: {gamma}")
            env.reset()
            del DQL


        ax.set_xlabel("Episode")
        ax.set_ylabel("Total Reward")
        ax.set_title("Comparison of Models with different gamma")
        ax.legend()
        plt.savefig('./gamma.png')

    elif command == 'policy':
        # This will test three configurations of batch_sizes with [32, 64, 128] different sizes respectively
        for policy in hyperparams.policy:

            if policy == 'egreedy':
                # This will test three configurations of epsilon with [0.1, 0.2, 0.3] different sizes respectively
                for epsilon in hyperparams.epsilon:
                    env.close()
                    env = gym.make("CartPole-v1")
                    print(f"epsilon: {epsilon}")

                    DQL = DQLAgent(env, 3, 64, Adam(learning_rate=0.005))
                    TotalRewards = Q_learn(env, DQL, episodes, gamma=.9, replay=True,  replaySize = 128 , targetNetwork=True, n_update = 20, policy='egreedy',
                                epsilon=epsilon, temp=1)
                    TotalRewards = Tool.smooth(TotalRewards, 5)
                    DQL.ResetWeights()
                    ax.plot(TotalRewards, label=f"epsilon: {epsilon}")
                    env.reset()
                    del DQL



            if policy == 'softmax':
            # This will test three configurations of epsilon with [0.1, 0.2, 0.3] different sizes respectively
                for temp in hyperparams.temp:
                    env.close()
                    env = gym.make("CartPole-v1")
                    print(f"temp: {temp}")

                    DQL = DQLAgent(env, 3, 64, Adam(learning_rate=0.005))
                    TotalRewards = Q_learn(env, DQL, episodes, gamma=.9, replay=True,  replaySize = 128 , targetNetwork=True, n_update = 20, policy='softmax',
                               epsilon=0.3, temp=temp)
                    TotalRewards = Tool.smooth(TotalRewards, 5)
                    DQL.ResetWeights()
                    ax.plot(TotalRewards, label=f"temp: {temp}")
                    env.reset()
                    del DQL


        ax.set_xlabel("Episode")
        ax.set_ylabel("Total Reward")
        ax.set_title("Comparison of Models with different policies")
        ax.legend()
        plt.savefig('./policy.png')
        


def plot(reward_data):
    # Plot the reward and loss and save it in the current directory
    plt.plot(reward_data)
    plt.title('Reward')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.savefig('./reward.png')
    plt.show()

    #Exploration strategy
def select_action(env, DQLAgent, observation, policy = 'egreedy', epsilon = 0.3, temp = 1):

    if policy == 'egreedy':
        if epsilon is None:
            raise KeyError("Provide an epsilon")
        if np.random.uniform(0, 1, 1) < epsilon:
            action = random.randrange(DQLAgent.n_actions)
        else:
            action = np.argmax(DQLAgent.predict(observation))


    elif policy == 'softmax':
        if temp is None:
            raise KeyError("Provide a temperature")
        pi = Tool.softmax(DQLAgent.predict(observation), temp)[0]  
        action = np.random.choice(DQLAgent.n_actions, p=pi)
    return action

def Q_learn(env, DQLAgent, episodes, gamma, epsilonDecay = 0.99,
            replay = False, replaySize = 20, targetNetwork = False,
            n_update = 10, policy = 'egreedy', epsilon = 0.3, temp = 1, ablation = False):
        
    totalRewards = []
    memory = deque(maxlen=8000)


    for episode in range(episodes):
        print(episode, "-----------------------------------------------")
        if targetNetwork:
            if episode % n_update == 0:
                DQLAgent.Tupdate()


        observation, info = env.reset(seed=42)
        observation = np.reshape(observation, [1, DQLAgent.n_states])

        done = False
        totalReward = 0

        #self.observation = np.reshape(self.observation, [1, self.n_states])

        while not done:
            action = select_action(env, DQLAgent, observation, policy, epsilon, temp) 
            nextObservation, reward, done, truncated, info = env.step(action)
            nextObservation = np.reshape(nextObservation, [1, DQLAgent.n_states])

            totalReward += reward


            memory.append((observation, action, nextObservation, reward, done))
            q_values = DQLAgent.predict(observation)
            

            if done:
                if not replay:
                    q_values[0][action] = reward
                    DQLAgent.update(observation, q_values)
                    break

            if replay:
                DQLAgent.replay(memory, replaySize,targetNetwork ,gamma)
            
            else:
                if targetNetwork:
                    q_value_next = DQLAgent.Tpredict(nextObservation)
                else:
                    q_value_next = DQLAgent.predict(nextObservation)

                q_values[0][action] = reward + gamma * np.amax(q_value_next).item()
                DQLAgent.update(observation, q_values)
                
            if totalReward >= 500:
                break

            observation = nextObservation

        epsilon = max(epsilon * epsilonDecay, 0.01)
        totalRewards.append(totalReward)
        print("reward is =", totalReward)

    return(totalRewards)


def main():

    episodes = 50
    env = gym.make("CartPole-v1")

    args = sys.argv[1:]

    if args[0] == '--ablation':
        Ablation(env)

    elif args[0] == '--tune':
        try:
            for arg in args[1:]:
                experiment(env, episodes, arg)

        except:
            print(f"Invalid hyperparameter {arg}. Try: 'layer', 'unit', 'optimizer', 'batch_size', 'epochs', 'policy', 'gamma'")

    else: 
        print("Invalid input")

    env.close()

if __name__ == '__main__':

    try:
        main()
        print("Done!")

    except KeyboardInterrupt: 
        print("Pressed Ctrl-C to kill program.")

