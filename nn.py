import numpy as np


class NeuralNetwork():

    def __init__(self, layer_sizes):

        # TODO
        # layer_sizes example: [4, 10, 2]
        self.input_layer_sizes = layer_sizes[0]
        self.hidden_layer_sizes = layer_sizes[1]
        self.output_layer_sizes = layer_sizes[2]
        self.first_layer_weights_matrix = np.random.normal(0, 1, size=(layer_sizes[1],layer_sizes[0] ))
        self.second_layer_weights_matrix =  np.random.normal(0, 1, size=(layer_sizes[2],layer_sizes[1]))
        self.b1 =  np.random.normal(0,1, size = (layer_sizes[1],1))
        self.b2 =  np.random.normal(0,1, size = (layer_sizes[2],1))
    
    def activation(self, x):    
        # TODO
        x = 1/(1+np.exp(-x))
        
        return x

    def forward(self, x):
        a1 = self.activation((self.first_layer_weights_matrix @ x) +self.b1)
        a2 = self.activation((self.second_layer_weights_matrix @ a1 ) +self.b2)
        return a2
