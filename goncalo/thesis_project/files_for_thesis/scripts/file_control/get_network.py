import os
import time
import math
import numpy as np
import nibabel as nib
import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, load_model
from keras.utils.vis_utils import plot_model
from tensorflow.keras.applications.vgg16 import VGG16

def create_model_from_vgg16_bai_weights():
    saver = tf.train.import_meta_graph('./FCN_sa/FCN_sa.meta')
    sess = tf.Session()
    # Top = False means we're not loading the classification layers (decoder).
    model = VGG16(include_top=False, weights=saver.restore(sess, './FCN_sa/FCN_sa'))
    # Handcarving vgg16 layers to match bai's...
    tmp = model.get_config()
    tmp['layers'][0]['config']['batch_input_shape'] = (None, 192, 192, 16)
    tmp['layers'][1]['config']['batch_input_shape'] = (None, 192, 192, 16)
    tmp['layers'][2]['config']['batch_input_shape'] = (None, 96, 96, 32)
    tmp['layers'][3]['config']['batch_input_shape'] = (None, 96, 96, 32)
    tmp['layers'][4]['config']['batch_input_shape'] = (None, 48, 48, 64)
    tmp['layers'][5]['config']['batch_input_shape'] = (None, 48, 48, 64)
    tmp['layers'][6]['config']['batch_input_shape'] = (None, 48, 48, 64)
    tmp['layers'][7]['config']['batch_input_shape'] = (None, 24, 24, 128)
    tmp['layers'][8]['config']['batch_input_shape'] = (None, 24, 24, 128)
    tmp['layers'][9]['config']['batch_input_shape'] = (None, 24, 24, 128)
    tmp['layers'][10]['config']['batch_input_shape'] = (None, 12, 12, 256)
    tmp['layers'][11]['config']['batch_input_shape'] = (None, 12, 12, 256)
    tmp['layers'][12]['config']['batch_input_shape'] = (None, 12, 12, 256)
    
    model = model.from_config(tmp)

    return model

def write_model_sum(model, file_path):
    with open(file_path,'w') as fh:
        # Pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda x: fh.write(x + '\n'))

def save_model(model, name):
    model.save(name)
    print('Saved model!')

def load_model(model_path):
  model = tf.keras.models.load_model(model_path)
  return model

def write_output(output, file):
    ''' Writes output to files.
    '''
    with open(file ,"a+") as f:
        for line in output:
            f.write(str(line) + "\n")

def main():

    ''' load models '''
    # vgg16_bai = load_model("./vgg16/vgg16_bai_weights.h5")

    model = create_model_from_vgg16_bai_weights()
    # save_model(model, './vgg16/bai_loaded.h5')
    # save_model(model, './vgg16/bai_loaded.pb')

    ''' plot model archs '''
    # plot_model(model, to_file='./model_archs/bai_loaded.png')

    '''Make models and save to .h5'''
    # model = VGG16(include_top = True)
    # save_model(model, "./vgg16/vgg16_vanilla_top_on.h5")


    # model = create_model_from_vgg16_bai_weights()
    write_model_sum(model, "./model_archs/bai_loaded.txt")

    # write_output(model.summary(), "./vgg16_arch_table.txt")
    # keras.utils.plot_model(model, show_dtype=True, 
    #                     show_layer_names=True, show_shapes=True,  
    #                     to_file='vgg16_arch.png')

    # try:
    # print("Attempting to load Bai s model")
    # Import the computation graph and restore the variable values
    # saver = tf.train.import_meta_graph("./FCN_sa.meta")
    # sess = tf.Session()
    # saver.restore(sess, "./FCN_sa")

    # print("Loaded saver is of type: {}".format(type(saver)))
    # except:
    #     print("Something wrong with loading the existing model from Bai")
    #     pass
    # print()
    # try:
    # print('OUTPUT model layers')
    # print(saver.outputLayers)
    # except:
    #     print("OutputLayers do not exist")
    #     pass
    # print()
    # try:
    #     print('INPUT model layers')
    #     print(saver.inputLayers)
    # except:
    #     print("Input layers do not exist")
    #     pass
    # print()
    # try:
    #     print("Trying to access graph")
    #     # Access the graph
    #     graph = tf.get_default_graph()
    #     print("Graph is of type: {}".format(type(graph)))
    # except:
    #     print("Something wrong with accessing the graph")
    #     pass
    # print()
    # '''
    # Where is the model...?
    # '''
    # try:
    #     print("Attempting to serialize loaded model")
    #     # Attempt to serialize the model - object here depends on above outputs: is it graph or is it saver or something else where the model is saved?
    #     graph.save('./serializedmodel')
    #     # To load model from serializable
    #     # model = await tf.loadLayersModel(./serializedmodel);
    # except:
    #     print("Something wrong with serializing Bai s model")
    #     pass
    # print()
    # try:
    #     print("Trying to save model architecture to png: using saver")
    #     # try with saver and then with graph - which of these is model
    #     tf.keras.utils.plot_model(
    #         saver, to_file='model.png', show_shapes=False, show_dtype=False,
    #         show_layer_names=True, rankdir='TB', expand_nested=False, dpi=96
    #     )
    # except:
    #     print("Something wrong with saver - cant save model.png")
    #     pass
    # print()
    # try:
    #     print("Trying to save model architecture to png: using graph")
    #     # try with graph
    #     tf.keras.utils.plot_model(
    #         graph, to_file='model.png', show_shapes=False, show_dtype=False,
    #         show_layer_names=True, rankdir='TB', expand_nested=False, dpi=96
    #     )
    # except:
    #     print("Something wrong with graph - cant save model.png")
    #     pass
    print("Done")

if __name__ == '__main__':
    main()