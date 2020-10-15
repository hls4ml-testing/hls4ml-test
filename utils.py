import os

import numpy as np
import hls4ml

from qkeras import *

__all__=['_accuracy','_output_shape','_layer_number','_alpha','cleanup']

def _accuracy(prediction1,prediction2,percent):
    prediction1=prediction1[0]
    c=[]
    for i in range(len(prediction2)):
        if prediction1[i]==prediction2[i]:
            c.append(0)
        else:
            c.append(np.abs(prediction2[i] - prediction1[i]) * 100 /prediction1[i])

    return np.average(c)<percent

def _output_shape(prediction1,prediction2):
    return len(prediction1[0])==len(prediction2)

def _layer_number(model,hls_model):
    return len(model.layers) + 1 == len(hls_model.get_layers())

def _alpha(model,hls_model):
    return (list(hls_model.get_layers())[1].attributes['class_name'] == model.layers[1].__class__.__name__) & (list(hls_model.get_layers())[2].attributes['class_name'] == 'Alpha')
