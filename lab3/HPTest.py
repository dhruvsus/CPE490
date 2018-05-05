import sys
import numpy as np
from keras import models, layers
from matplotlib import pyplot as plt
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers
from sys import exit
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Reorganize training and test data

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train/=255
X_test/=255

number_of_classes = 10

Y_train = to_categorical(y_train, number_of_classes)
Y_test = to_categorical(y_test, number_of_classes)
gen = ImageDataGenerator(rotation_range=8, width_shift_range=0.08, shear_range=0.3,
                         height_shift_range=0.08, zoom_range=0.08)

test_gen = ImageDataGenerator()
train_generator = gen.flow(X_train, Y_train, batch_size=64)
test_generator = test_gen.flow(X_test, Y_test, batch_size=64)
class LyrParams:
    def __init__(self):
        self.layer_type = ''
        self.num_channels_or_neurons = 0
        self.dropout_rate = 0.0
        self.regularization = None
        self.activation_function = ''
        self.non_train_layer = ''

    def __repr__(self):
        num_channels_or_neurons_string = ' channels ' if self.layer_type == 'c' else ' neurons '
        regularization_string = 'None' if self.regularization == None else 'True'
        #layer_type='conv ' if self.layer_type=='c' else 'dense '
        return self.layer_type + ' ' + str(
            self.num_channels_or_neurons
        ) + num_channels_or_neurons_string + str(
            self.dropout_rate) + " reg " + regularization_string


def str_to_lyr(lyr_params_split):
    layer = LyrParams()
    layer.layer_type = lyr_params_split[0]
    layer.num_channels_or_neurons = int(lyr_params_split[1])
    layer.activation_function = lyr_params_split[2]
    layer.dropout_rate = float(lyr_params_split[3])
    layer.regularization = lyr_params_split[4]
    layer.non_train_layer = lyr_params_split[5]
    return layer


def fix_nn_description(nn_list):
    for layer in nn_list:
        layer.layer_type = 'Conv2D' if layer.layer_type == 'c' else 'Dense'
        regularization_str=layer.regularization
        if 'l1' in regularization_str:
            layer.regularization = regularizers.l1()
        if 'l2' in regularization_str:
            layer.regularization = regularizers.l2()
        if 'n' in regularization_str:
            layer.regularization = None
    return nn_list


def make_a_nn(nn_description):
    nn = models.Sequential()
    nn_list = [
        str_to_lyr(layer_description.split())
        for layer_description in nn_description
    ]
    nn_list = fix_nn_description(nn_list)
    print(*nn_list,sep='\n')
    #terrible assumption that the first layer is a conv2d with 28,28,1 input shape
    nn = models.Sequential()
    first_layer = nn_list.pop(0)
    nn.add(
        layers.Conv2D(
            first_layer.num_channels_or_neurons, (3, 3),
            activation=first_layer.activation_function,
            input_shape=(28, 28, 1),
            kernel_regularizer=first_layer.regularization))
    if 'd' in first_layer.non_train_layer:
        nn.add(layers.Dropout(rate=first_layer.dropout_rate))
    if 'p' in first_layer.non_train_layer:
        nn.add(layers.MaxPooling2D((2, 2)))
    if 'b' in first_layer.non_train_layer:
        nn.add(layers.BatchNormalization())
    if 'f' in first_layer.non_train_layer:
        nn.add(layers.Flatten)
    #Now for the rest of the layers
    for layer in nn_list:
        if (layer.layer_type == 'Conv2D'):
            nn.add(
                layers.Conv2D(
                    layer.num_channels_or_neurons, (3, 3),
                    activation=layer.activation_function,
                    kernel_regularizer=layer.regularization))
        if (layer.layer_type == 'Dense'):
            nn.add(layers.Dense(layer.num_channels_or_neurons,activation=layer.activation_function,
            kernel_regularizer=layer.regularization))
        if 'd' in layer.non_train_layer:
            nn.add(layers.Dropout(rate=layer.dropout_rate))
        if 'p' in layer.non_train_layer:
            nn.add(layers.MaxPooling2D())
        if 'b' in layer.non_train_layer:
            nn.add(layers.BatchNormalization())
        if 'f' in layer.non_train_layer:
            nn.add(layers.Flatten())
    #Time to compile and return
    nn.compile(
        optimizer="rmsprop",  # Improved backprop algorithm
        loss='categorical_crossentropy',  # "Misprediction" measure
        metrics=['accuracy']  # Report CCE value as we train
    )
    return nn
def test_nn(nn,epochs):
    hst=nn.fit_generator(train_generator, steps_per_epoch=60000//64, epochs=epochs,
                    validation_data=test_generator, validation_steps=10000//64,verbose=0)
    score = nn.evaluate(X_test, Y_test,verbose=0)
    print(score)
    hst=hst.history
    for i in range(epochs):
        print(str(hst['val_acc'][i]) + ' / ' + str(hst['val_loss'][i]) + '  '+str(hst['acc'][i])+' / '+str(hst['loss'][i]))
    print()
def main():
    #read lines of stdin into list
    stdin_list = sys.stdin.readlines()
    while len(stdin_list) != 0:
        num_layers = int(stdin_list.pop(0))
        nn=make_a_nn(stdin_list[:num_layers])
        #nn.summary()
        test_nn(nn,2)
        stdin_list = stdin_list[num_layers:]


if __name__ == '__main__':
    main()
