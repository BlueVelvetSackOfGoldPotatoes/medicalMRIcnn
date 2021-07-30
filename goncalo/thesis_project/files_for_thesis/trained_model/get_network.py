import os
import time
import math
import numpy as np
import nibabel as nib
import tensorflow as tf

def main():
    # Import the computation graph and restore the variable values
    saver = tf.train.import_meta_graph('{0}.meta'.format('./trained_model'))
    saver.restore(sess, '{0}'.format('./trained_model'))
    
    # Access the graph
    graph = tf.get_default_graph()
    ## Prepare the feed_dict for feeding data for fine-tuning 

    # try with saver and then with graph - which of these is model
    tf.keras.utils.plot_model(
    saver, to_file='model.png', show_shapes=False, show_dtype=False,
    show_layer_names=True, rankdir='TB', expand_nested=False, dpi=96
)


if __name__ == '__main__':
    main()