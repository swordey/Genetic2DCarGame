from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import numpy as np
import random


class Brain:
    def __init__(self, input_size, action2movetable, number_of_brains, learning, path, learning_rate=0.002):
        self.mutation_rate = 0.01
        self.learning_rate = learning_rate
        self.number_of_brains = number_of_brains
        self.path = path

        self.input_size = input_size
        self.action2movetable = action2movetable
        self.model = None

        self.create_model(learning)

    def create_model(self, learning):
        if learning:
            self.create_scaled_model()
        else:
            self.load_model()

    def create_scaled_model(self):
        self.model = Sequential()
        self.model.add(Dense(len(self.action2movetable)*self.number_of_brains,
                             input_dim=(self.input_size*self.number_of_brains),
                             activation='softmax'))
        self.model.compile(loss='mse',
                           optimizer=Adam(lr=self.learning_rate))
        weights, bias = self.get_weights_separated()
        self.set_weights_separated(weights, bias)

    def create_taught_model(self):
        self.model = Sequential()
        self.model.add(Dense(len(self.action2movetable),
                             input_dim=(self.input_size),
                             activation='softmax'))
        self.model.compile(loss='mse',
                           optimizer=Adam(lr=self.learning_rate))

    def set_weights_separated(self,weights, biases):
        weigth, bias = self.model.get_weights()
        new_weights = np.zeros_like(weigth)
        for idx in range(self.number_of_brains):
            new_weights[idx*self.input_size:(idx+1)*self.input_size,
                        idx*len(self.action2movetable):(idx+1)*len(self.action2movetable)] = \
                        weights[idx]
        self.model.set_weights((new_weights,biases))

    def get_weights_separated(self):
        weigth, bias = self.model.get_weights()
        weights = []
        for idx in range(self.number_of_brains):
            weights.append(weigth[idx * self.input_size:(idx + 1) * self.input_size,
                           idx * len(self.action2movetable):(idx + 1) * len(self.action2movetable)])
        return np.array(weights), bias

    def generate_new_weights(self,fathers, mothers):
        weights, bias = self.get_weights_separated()
        length, width = np.array(weights[0]).shape

        new_weights = []

        for mother_id, father_id in zip(mothers,fathers):
            cross_over_point = random.randint(1, length - 1)
            new_weights.append(
                np.array(list(weights[mother_id][0:cross_over_point, :]) +
                         list(weights[father_id][cross_over_point:, :]))
            )

        new_weight = np.zeros([weights[0].shape[0]*self.number_of_brains,weights[0].shape[1]*self.number_of_brains])
        for idx in range(len(new_weights)):
            new_weight[idx*self.input_size:(idx+1)*self.input_size,
                        idx*len(self.action2movetable):(idx+1)*len(self.action2movetable)] = new_weights[idx]

        self.model.set_weights((new_weight,bias))

    def mutate(self):
        weights, bias = self.get_weights_separated()
        new_weights = []
        for weight in weights:
            for idx in range(weight.shape[0]):
                for idy in range(weight.shape[1]):
                    rand = random.uniform(0, 1)
                    if rand < self.mutation_rate:
                        change = (random.uniform(0, 4) - 2)
                        weight[idx, idy] += change
            new_weights.append(weight)

        self.set_weights_separated(new_weights,bias)

    def action2mov(self,action):
        return self.action2movetable[action]

    def act(self,state):
        actions = self.model.predict(state)[0]
        movements = []
        for idx in range(self.number_of_brains):
            num = np.argmax(actions[idx*3:(idx+1)*3])
            movements.append(self.action2mov(num))
        return movements

    def save_model_by_id(self,id):
        weigths, bias = self.get_weights_separated()
        np.save(self.path+"_weights.npy",weigths[id])

    def load_model(self):
        self.create_taught_model()
        weight = np.squeeze(np.load(self.path+"_weights.npy"))
        bias = np.zeros([weight.shape[1]])
        self.model.set_weights((weight,bias))