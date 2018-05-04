import sys
import numpy as np
from keras import models, layers
from matplotlib import pyplot as plt
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from sys import exit
nn=models.Sequential()
class LyrParams:
    def __init__(self):
        self.layer_type=''
        self.num_channels_or_neurons=0
        self.dropout_rate=0.0
        self.regularization=None
        self.activation_function=''
        self.non_train_layer=''
    def __repr__(self):
        num_channels_or_neurons_string=' channels ' if self.layer_type=='c' else ' neurons '
        layer_type='conv ' if self.layer_type=='c' else 'dense '
        return layer_type+str(self.num_channels_or_neurons)+num_channels_or_neurons_string+str(self.dropout_rate)+" reg "+self.regularization
def str_to_lyr(lyr_params_split):
    layer=LyrParams()
    layer.layer_type=lyr_params_split[0]
    layer.num_channels_or_neurons=int(lyr_params_split[1])
    layer.activation_function=lyr_params_split[2]
    layer.dropout_rate=float(lyr_params_split[3])
    layer.regularization=lyr_params_split[4]
    layer.non_train_layer=lyr_params_split[5]
    return layer
def lyr_list_to_nn(lyr_params_list):
    for layer in lyr_params_list:
        if(layer.layer_type=='c'):
            nn.add(layers.Conv2D(layer.num_channels_or_neurons,(3,3),activation=layer.activation_function))
def make_a_nn(nn_description):
    nn_list=[str_to_lyr(layer_description.split()) for layer_description in nn_description]
    #terrible assumption that the first layer is a conv2d with 28,28,1 input shape
    nn = models.Sequential()
    print(*nn_list,sep='\n')
    first_layer=nn_description.pop(0)
    print(first_layer)
    nn.add(layers.Conv2D(first_layer.num_channels_or_neurons, (3, 3), activation=first_layer.activation_function, input_shape=(28, 28, 1)))
def main():
    #read lines of stdin into list
    stdin_list=sys.stdin.readlines()
    while len(stdin_list)!=0:
        num_layers=int(stdin_list.pop(0))
        make_a_nn(stdin_list[:num_layers])
        stdin_list=stdin_list[num_layers:]
if __name__ == '__main__':
    main()
