# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 21:42:51 2016

@author: picturio
"""

    
import os
import numpy as np
import matplotlib.pyplot as plt

#from cntk.device import gpu, set_default_device
from cntk import cross_entropy_with_softmax, classification_error, relu, input_variable, softmax, element_times
from cntk.layers import Convolution, MaxPooling, Dropout, Dense, For, Sequential, default_options
from cntk.io import MinibatchSource, ImageDeserializer, StreamDef, StreamDefs, transforms
from cntk.initializer import glorot_uniform
from cntk import Trainer
from cntk import momentum_sgd, learning_rate_schedule, UnitType, momentum_as_time_constant_schedule
from cntk.logging import log_number_of_parameters, ProgressPrinter

user='picturio'
output_base_dir=os.path.join(r'C:\Users',user,'OneDrive\WBC\DATA')
train_dir=os.path.join(output_base_dir,'Training')

model_file=os.path.join(train_dir,'cnn_model.dnn')
model_temp_file=os.path.join(train_dir,'cnn_model_temp.dnn')

#train_filename = os.path.join(train_dir,'Train_cntk_text.txt')
#test_filename = os.path.join(train_dir,'Test_cntk_text.txt')
#train_regr_labels=os.path.join(train_dir,'train_regrLabels.txt')

train_map=os.path.join(train_dir,'train_map_e600_3000.txt')
test_map=os.path.join(train_dir,'test_map_e200_1000.txt')
# GET train and test map from prepare4train

data_mean_file=os.path.join(train_dir,'data_mean.xml')

# model dimensions
image_height = 32
image_width  = 32
num_channels = 3
num_classes  = 6


def create_basic_model(input, out_dims):
    
    convolutional_layer_1  = Convolution((5,5), 16, init=glorot_uniform(), activation=relu, pad=True, strides=(1,1))(input)
    pooling_layer_1  = MaxPooling((2,2), strides=(1,1))(convolutional_layer_1 )

    convolutional_layer_2 = Convolution((9,9), 16, init=glorot_uniform(), activation=relu, pad=True, strides=(1,1))(pooling_layer_1)
    pooling_layer_2 = MaxPooling((5,5), strides=(2,2))(convolutional_layer_2)

#    convolutional_layer_3 = Convolution((7,7), 32, init=glorot_uniform(), activation=relu, pad=True, strides=(1,1))(pooling_layer_2)
#    pooling_layer_3 = MaxPooling((3,3), strides=(2,2))(convolutional_layer_3)
#    
    fully_connected_layer  = Dense(128, init=glorot_uniform())(pooling_layer_2)
    dropout_layer = Dropout(0.5)(fully_connected_layer)

    output_layer = Dense(out_dims, init=glorot_uniform(), activation=None)(dropout_layer)
    
    return output_layer

def create_advanced_model(input, out_dims):
    
    with default_options(activation=relu):
        model = Sequential([
            For(range(2), lambda i: [  # lambda with one parameter
                Convolution((3,3), [16,32][i], pad=True),  # depth depends on i
                #Convolution((5,5), [16,16][i], pad=True),
                Convolution((9,9), [16,32][i], pad=True),            
                MaxPooling((3,3), strides=(2,2))
            ]),
            For(range(2), lambda : [   # lambda without parameter
                Dense(128),
                Dropout(0.5)
            ]),
            Dense(out_dims, activation=None)
        ])
    output_layer=model(input)
    
    return output_layer

#
# Define the reader for both training and evaluation action.
#
def create_reader(map_file, mean_file, train):
  
    # transformation pipeline for the features has jitter/crop only when training
    trs = []
#    if train:
#        transforms += [
#            ImageDeserializer.crop(crop_type='Random', ratio=0.8, jitter_type='uniRatio') # train uses jitter
#        ]
    trs += [
        transforms.scale(width=image_width, height=image_height, channels=num_channels, interpolations='linear'),
        transforms.mean(mean_file)
    ]
    # deserializer
    return MinibatchSource(ImageDeserializer(map_file, StreamDefs(
        features = StreamDef(field='image', transforms=trs), # first column in map file is referred to as 'image'
        labels   = StreamDef(field='label', shape=num_classes)      # and second as 'label'
    )))
    
#
# Train and evaluate the network.
#
def train_and_evaluate(reader_train, reader_test, max_epochs, model_func):
    # Input variables denoting the features and label data
    input_var = input_variable((num_channels, image_height, image_width))
    label_var = input_variable((num_classes))

    # Normalize the input
    feature_scale = 1.0 / 256.0
    input_var_norm = element_times(feature_scale, input_var)
    
    # apply model to input
    z = model_func(input_var_norm, out_dims=num_classes)

    #
    # Training action
    #

    # loss and metric
    ce = cross_entropy_with_softmax(z, label_var)
    pe = classification_error(z, label_var)

    # training config
    epoch_size     = 9000
    minibatch_size = 64

    # Set training parameters
    lr_per_minibatch       = learning_rate_schedule([0.01]*10 + [0.003]*10 + [0.001],  UnitType.minibatch, epoch_size)
    momentum_time_constant = momentum_as_time_constant_schedule(-minibatch_size/np.log(0.9))
    l2_reg_weight          = 0.001
    
    # trainer object
    progress_printer = ProgressPrinter(0)

    learner     = momentum_sgd(z.parameters, 
                               lr = lr_per_minibatch, momentum = momentum_time_constant, 
                               l2_regularization_weight=l2_reg_weight)
    trainer     = Trainer(z, (ce, pe), [learner], [progress_printer])

    # define mapping from reader streams to network inputs
    input_map = {
        input_var: reader_train.streams.features,
        label_var: reader_train.streams.labels
    }

    log_number_of_parameters(z) ; print()
    #progress_printer = ProgressPrinter(tag='Training')

    # perform model training
    batch_index = 0
    plot_data = {'batchindex':[], 'loss':[], 'error':[]}
    for epoch in range(max_epochs):       # loop over epochs
        sample_count = 0
        while sample_count < epoch_size:  # loop over minibatches in the epoch
            data = reader_train.next_minibatch(min(minibatch_size, epoch_size - sample_count), input_map=input_map) # fetch minibatch.
            trainer.train_minibatch(data)                                   # update model with it

            sample_count += data[label_var].num_samples                     # count samples processed so far
            
            # For visualization...            
            plot_data['batchindex'].append(batch_index)
            plot_data['loss'].append(trainer.previous_minibatch_loss_average)
            plot_data['error'].append(trainer.previous_minibatch_evaluation_average)
            
            progress_printer.update_with_trainer(trainer, with_metric=True) # log progress
            batch_index += 1
        progress_printer.epoch_summary(with_metric=True)
        #trainer.save_checkpoint(model_temp_file)
        
    #
    # Evaluation action
    #
    epoch_size     = 3000
    minibatch_size = 32

    # process minibatches and evaluate the model
    metric_numer    = 0
    metric_denom    = 0
    sample_count    = 0
    minibatch_index = 0

    input_map = {
        input_var: reader_test.streams.features,
        label_var: reader_test.streams.labels
    }

    while sample_count < epoch_size:
        current_minibatch = min(minibatch_size, epoch_size - sample_count)

        # Fetch next test min batch.
        data = reader_test.next_minibatch(current_minibatch, input_map=input_map)

        # minibatch data to be trained with
        metric_numer += trainer.test_minibatch(data) * current_minibatch
        metric_denom += current_minibatch

        # Keep track of the number of samples processed so far.
        sample_count += data[label_var].num_samples
        minibatch_index += 1

    print("")
    print("Final Results: Minibatch[1-{}]: errs = {:0.1f}% * {}".format(minibatch_index+1, (metric_numer*100.0)/metric_denom, metric_denom))
    print("")
    
    # Visualize training result:
    window_width            = 32
    loss_cumsum             = np.cumsum(np.insert(plot_data['loss'], 0, 0)) 
    error_cumsum            = np.cumsum(np.insert(plot_data['error'], 0, 0)) 

    # Moving average.
    plot_data['batchindex'] = np.insert(plot_data['batchindex'], 0, 0)[window_width:]
    plot_data['avg_loss']   = (loss_cumsum[window_width:] - loss_cumsum[:-window_width]) / window_width
    plot_data['avg_error']  = (error_cumsum[window_width:] - error_cumsum[:-window_width]) / window_width
    
    plt.figure(1)
    plt.subplot(211)
    plt.plot(plot_data["batchindex"], plot_data["avg_loss"], 'b--')
    plt.xlabel('Minibatch number')
    plt.ylabel('Loss')
    plt.title('Minibatch run vs. Training loss ')
    
    plt.show()

    plt.subplot(212)
    plt.plot(plot_data["batchindex"], plot_data["avg_error"], 'r--')
    plt.xlabel('Minibatch number')
    plt.ylabel('Label Prediction Error')
    plt.title('Minibatch run vs. Label Prediction Error ')
    plt.show()
    
    return softmax(z)
    
reader_train = create_reader(train_map, data_mean_file, True)
reader_test  = create_reader(test_map, data_mean_file, False)

pred = train_and_evaluate(reader_train, reader_test, max_epochs=500,
                          model_func=create_advanced_model)
#pred_batch= train_and_evaluate(reader_train, reader_test, max_epochs=10, model_func=create_basic_model_with_batch_normalization)

pred.save_model(model_file)