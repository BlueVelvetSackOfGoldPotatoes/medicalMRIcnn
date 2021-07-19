import tensorflow.compat.v1 as tf
from tensorflow.keras.applications import VGG16
from tensorflow import keras
from keras.utils.vis_utils import plot_model
from image_prep import *

def create_model_from_vgg16():
  saver = tf.train.import_meta_graph('./checkpoint/FCN_sa.meta')
  sess = tf.Session()
  # saver.restore(sess, './checkpoint/FCN_sa')
  model = VGG16(include_top=False, weights=saver.restore(sess, './checkpoint/FCN_sa'))

  return model

def save_model(model, name):
  model.save(name)
  print('Normal Keras save done!')

def load_model():
  model = tf.keras.models.load_model('model_h5/vgg16')
  return model

def model_summary(model):
  model.summary()
  plot_model(model, to_file='vgg.png')

def main():
  # LOAD MODEL FROM .ckpt
  # saver = tf.train.import_meta_graph('./checkpoint/FCN_sa.meta')
  # sess = tf.Session()
  # saver.restore(sess, './checkpoint/FCN_sa')
  # saver.restore(sess, tf.train.latest_checkpoint('./checkpoint'))

  model = load_model()

  new_model = connect_model(model)

  
if __name__ == '__main__':
  main()