# Copyright 2017, Wenjia Bai. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import os
import time
import random
import numpy as np
import nibabel as nib
import tensorflow.compat.v1 as tf
import tensorflow_core.tools.graph_transforms as graph_transforms
from common.network import build_FCN
from common.image_utils import tf_categorical_accuracy, tf_categorical_dice
from common.image_utils import crop_image, rescale_intensity, data_augmenter
from random import seed
from random import randint

# So that tf.placeholder() can be used
tf.disable_eager_execution()

""" Parameters """
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_enum('seq_name', 'sa',
                         ['sa'],
                         'Sequence name.')
tf.app.flags.DEFINE_integer('image_size', 192,
                            'Image size after cropping.')
tf.app.flags.DEFINE_integer('train_batch_size', 3,
                            'Number of images for each training batch.')
tf.app.flags.DEFINE_integer('validation_batch_size', 3,
                            'Number of images for each validation batch.')
# 5000 before edit
tf.app.flags.DEFINE_integer('train_iteration',  10,
                            'Number of training iterations.')
tf.app.flags.DEFINE_integer('num_filter', 16,
                            'Number of filters for the first convolution layer.')
tf.app.flags.DEFINE_integer('num_level', 5,
                            'Number of network levels.')
# Was 1e-3 before; using 1e-5 for finetuning  
tf.app.flags.DEFINE_float('learning_rate', 1e-3,
                          'Learning rate.')
tf.app.flags.DEFINE_string('dataset_dir',
                           '/home/goncalo/Documents/RUG/4th Year/2B/thesis/medicalMRIcnn/goncalo/thesis_project/files_for_thesis/data/data_for_training',
                           'Path to the dataset directory, which is split into '
                           'training, validation and test subdirectories.')
tf.app.flags.DEFINE_string('model_path',
                           '/home/goncalo/Documents/RUG/4th Year/2B/thesis/medicalMRIcnn/trained_model/FCN_sa',
                           'Path to the saved trained model.')

''' flag conflict with absl library which tf depends on 
https://github.com/tensorflow/tensorflow/issues/30472 requires log_dir flag to be changed
'''
tf.app.flags.DEFINE_string('log_dir_thesis',
                           '/home/goncalo/Documents/RUG/4th Year/2B/thesis/medicalMRIcnn/goncalo/thesis_project/files_for_thesis/log',
                           'Directory for saving the log file.')

tf.app.flags.DEFINE_string('checkpoint_dir',
                           '/home/goncalo/Documents/RUG/4th Year/2B/thesis/medicalMRIcnn/goncalo/thesis_project/files_for_thesis/checkpoint/',
                           'Directory for saving the trained model.')

def get_random_batch(filename_list, batch_size, image_size=192, data_augmentation=False,
                     shift=0.0, rotate=0.0, scale=0.0, intensity=0.0, flip=False):
    # Randomly select batch_size images from filename_list
    n_file = len(filename_list)
    n_selected = 0
    images = []
    labels = []
    seed(1)

    while n_selected < batch_size:
        rand_index = randint(0, n_file-1)

        image_name, label_name = filename_list[rand_index]
        if os.path.exists(image_name) and os.path.exists(label_name):
            print('  Select {0} {1}'.format(image_name, label_name))

            # Read image and label
            image = nib.load(image_name).get_data()
            label = nib.load(label_name).get_data()

            # Handle exceptions
            if image.shape != label.shape:
                print('Error: mismatched size, image.shape = {0}, '
                      'label.shape = {1}'.format(image.shape, label.shape))
                print('Skip {0}, {1}'.format(image_name, label_name))
                continue

            if image.max() < 1e-6:
                print('Error: blank image, image.max = {0}'.format(image.max()))
                print('Skip {0} {1}'.format(image_name, label_name))
                continue

            # Normalise the image size
            X, Y, Z = image.shape
            cx, cy = int(X / 2), int(Y / 2)
            image = crop_image(image, cx, cy, image_size)
            label = crop_image(label, cx, cy, image_size)

            # Intensity rescaling
            image = rescale_intensity(image, (1.0, 99.0))

            # Append the image slices to the batch
            # Use list for appending, which is much faster than numpy array
            for z in range(Z):
                images += [image[:, :, z]]
                labels += [label[:, :, z]]

            # Increase the counter
            n_selected += 1

    # Convert to a numpy array
    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)

    # Add the channel dimension
    # tensorflow by default assumes NHWC format
    images = np.expand_dims(images, axis=3)

    # Perform data augmentation
    if data_augmentation:
        images, labels = data_augmenter(images, labels,
                                        shift=shift, rotate=rotate,
                                        scale=scale,
                                        intensity=intensity, flip=flip)
    return images, labels

def main(argv=None):
    """ Main function """
    # Go through each subset (train, validation, test) under the data directory
    # and list the file names of the subjects
    init_op = tf.global_variables_initializer()
    data_list = {}
    for k in ['train', 'validation', 'test']:
        subset_dir = os.path.join(FLAGS.dataset_dir, k)
        data_list[k] = []

        for data in sorted(os.listdir(subset_dir)):
            data_dir = os.path.join(subset_dir, data)
            # Check the existence of the image and label map at ED and ES time frames
            # and add their file names to the list
            for fr in ['ED', 'ES']:
                image_name = '{0}/{1}_{2}.nii.gz'.format(data_dir, FLAGS.seq_name, fr)
                label_name = '{0}/label_{1}_{2}.nii.gz'.format(data_dir, FLAGS.seq_name, fr)
                if os.path.exists(image_name) and os.path.exists(label_name):
                    data_list[k] += [[image_name, label_name]]

    # Initialise variables
    init_op = tf.global_variables_initializer()
    # Start the tensorflow session
    with tf.Session() as sess:

        sess.run(init_op)
        saver = tf.train.import_meta_graph('{0}.meta'.format(FLAGS.model_path))

        # Import the computation graph and restore the variable values
        saver.restore(sess, '{0}'.format(FLAGS.model_path))

        print('Start training...')
        start_time = time.time()
        # print('{0}.meta'.format(FLAGS.model_path))

        graph = tf.get_default_graph()
 
        output_node_names = [n.name for n in graph.as_graph_def().node]

        # Attempting to freeze bn layers - might be done through bazel
        print("------------------ FREEZING BATCH NORMALIZATION NODES FOR FINE TUNING ------------------")
        graph_def = graph_transforms.TransformGraph(
            graph.as_graph_def(),
            [
                'strip_unused_nodes',
                'fold_constants(ignore_errors=true)',
                'fold_batch_norms',
            ],
            output_node_names,
            ["sort_by_execution_order"]
        )

        # print("------------------ FREEZING BATCH NORMALIZATION NODES FOR FINE TUNING ------------------")
        # bn_node_names = "batch_normalization/gamma,batch_normalization_1/beta,batch_normalization_1/gamma,batch_normalization_2/beta,batch_normalization_2/gamma,batch_normalization_3/beta,batch_normalization_3/gamma,batch_normalization_4/beta,batch_normalization_4/gamma,batch_normalization_5/beta,batch_normalization_5/gamma,batch_normalization_6/beta,batch_normalization_6/gamma,batch_normalization_7/beta,batch_normalization_7/gamma,batch_normalization_8/beta,batch_normalization_8/gamma,batch_normalization_9/beta,batch_normalization_9/gamma,batch_normalization_10/beta,batch_normalization_10/gamma,batch_normalization_11/beta,batch_normalization_11/gamma,batch_normalization_12/beta,batch_normalization_12/gamma,batch_normalization_13/beta,batch_normalization_13/gamma,batch_normalization_14/beta,batch_normalization_14/gamma,batch_normalization_15/beta,batch_normalization_15/gamma,batch_normalization_16/beta,batch_normalization_16/gamma,batch_normalization_17/beta,batch_normalization_17/gamma,batch_normalization_18/beta,batch_normalization_18/gamma,batch_normalization_19/beta,batch_normalization_19/gamma"
        # bn_node_names = bn_node_names.split(",")
        # print(bn_node_names)


        # DEPRECATED - We use a built-in TF helper to export variables to constants 
        # tensor_freeze = tf.graph_util.convert_variables_to_constants(
        #     sess, # The session is used to retrieve the weights
        #     graph_def, # The graph_def is used to retrieve the nodes 
        #     output_node_names # The output node names are used to select the usefull nodes
        # )

        # Prepare tensors for the image and label map pairs
        # Use int32 for label_pl as tf.one_hot uses int32
        # image_pl = tf.placeholder(tf.float32, shape=[None, None, None, 1], name='image')
        # print('Tensor freeze: ',tensor_freeze)
        # for l in tf.trainable_variables():
        #     print(l)
        
        image_pl = tf.placeholder(tf.float32, shape=[None, None, None, 1], name='image')
        image_pl = tf.get_default_graph().get_tensor_by_name('image:0')
        label_pl = tf.placeholder(tf.int32, shape=[None, None, None], name='label')
        label_pl = tf.get_default_graph().get_tensor_by_name('label:0')

        # Print out the placeholders' names, which will be useful when deploying the network
        print('Placeholder image_pl.name = ' + image_pl.name)
        print('Placeholder label_pl.name = ' + label_pl.name)

        training_pl = tf.placeholder(tf.bool, shape=[], name='training')
        training_pl = tf.get_default_graph().get_tensor_by_name('training:0')
        print('Placeholder training_pl.name = ' + training_pl.name)

        n_class = 4

        # The number of resolution levels
        n_level = FLAGS.num_level

        # The number of filters at each resolution level
        # Follow the VGG philosophy, increasing the dimension
        # by a factor of 2 for each level
        n_filter = []
        for i in range(n_level):
            n_filter += [FLAGS.num_filter * pow(2, i)]
        print('Number of filters at each level =', n_filter)
        print('Note: The connection between neurons is proportional to '
            'n_filter * n_filter. Increasing n_filter by a factor of 2 '
            'will increase the number of parameters by a factor of 4. '
            'So it is better to start experiments with a small n_filter '
            'and increase it later.')

        # Build the neural network, which outputs the logits,
        # i.e. the unscaled values just before the softmax layer,
        # which will then normalise the logits into the probabilities.
        n_block = [2, 2, 3, 3, 3]
        logits = build_FCN(image_pl, n_class, n_level=n_level,
                        n_filter=n_filter, n_block=n_block,
                        training=training_pl, same_dim=32, fc=64)
        print()
        print('PRINTING LOGITS ---------------------------------------------------')
        print(logits)
        print()

        # The softmax probability and the predicted segmentation
        prob = tf.nn.softmax(logits, name='prob')
        pred = tf.cast(tf.argmax(prob, axis=-1), dtype=tf.int32, name='pred')
        print('prob.name = ' + prob.name)
        print('pred.name = ' + pred.name)

        # Loss
        label_1hot = tf.one_hot(indices=label_pl, depth=n_class)
        label_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=label_1hot, logits=logits)
        loss = tf.reduce_mean(label_loss)

        # Evaluation metrics
        accuracy = tf_categorical_accuracy(pred, label_pl)
        dice_lv = tf_categorical_dice(pred, label_pl, 1)
        dice_myo = tf_categorical_dice(pred, label_pl, 2)
        dice_rv = tf_categorical_dice(pred, label_pl, 3)
        dice_la = tf_categorical_dice(pred, label_pl, 1)
        dice_ra = tf_categorical_dice(pred, label_pl, 2)

        # # Optimiser
        lr = FLAGS.learning_rate

        # We need to add the operators associated with batch_normalization
        # to the optimiser, according to
        # https://www.tensorflow.org/api_docs/python/tf/layers/batch_normalization
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            print('Using Adam optimizer.')
            train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

        model_name = 'FCN_sa'
        model_dir = os.path.join(FLAGS.checkpoint_dir, model_name)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
 
        # Summary writer
        summary_dir = FLAGS.log_dir_thesis
        if os.path.exists(summary_dir):
            os.system('rm -rf {0}'.format(summary_dir))
        train_writer = tf.summary.FileWriter(os.path.join(summary_dir, 'train'), graph=sess.graph)
        validation_writer = tf.summary.FileWriter(os.path.join(summary_dir, 'validation'), graph=sess.graph)

        # Iterate
        for iteration in range(1, 1 + FLAGS.train_iteration):
            # For each iteration, we randomly choose a batch of subjects
            print('Iteration {0}: training...'.format(iteration))
            start_time_iter = time.time()

            images, labels = get_random_batch(data_list['train'],
                                              FLAGS.train_batch_size,
                                              image_size=FLAGS.image_size,
                                              data_augmentation=True,
                                              shift=0, rotate=10, scale=0.2,
                                              intensity=0, flip=False)

            # Stochastic optimisation using this batch

            _, train_loss, train_acc = sess.run([train_op, loss, accuracy],
                                                {image_pl: images, label_pl: labels, training_pl: True})

            summary = tf.Summary()
            summary.value.add(tag='loss', simple_value=train_loss)
            summary.value.add(tag='accuracy', simple_value=train_acc)
            train_writer.add_summary(summary, iteration)

            # After every ten iterations, we perform validation
            if iteration % 10 == 0:
                print('Iteration {0}: validation...'.format(iteration))
                images, labels = get_random_batch(data_list['validation'],
                                                  FLAGS.validation_batch_size,
                                                  image_size=FLAGS.image_size,
                                                  data_augmentation=False)

                if FLAGS.seq_name == 'sa':
                    validation_loss, validation_acc, validation_dice_lv, validation_dice_myo, validation_dice_rv = \
                        sess.run([loss, accuracy, dice_lv, dice_myo, dice_rv],
                                 {image_pl: images, label_pl: labels, training_pl: False})

                summary = tf.Summary()
                summary.value.add(tag='loss', simple_value=validation_loss)
                summary.value.add(tag='accuracy', simple_value=validation_acc)
                if FLAGS.seq_name == 'sa':
                    summary.value.add(tag='dice_lv', simple_value=validation_dice_lv)
                    summary.value.add(tag='dice_myo', simple_value=validation_dice_myo)
                    summary.value.add(tag='dice_rv', simple_value=validation_dice_rv)
                validation_writer.add_summary(summary, iteration)

                # Print the results for this iteration
                print('Iteration {} of {} took {:.3f}s'.format(iteration, FLAGS.train_iteration,
                                                               time.time() - start_time_iter))
                print('  training loss:\t\t{:.6f}'.format(train_loss))
                print('  training accuracy:\t\t{:.2f}%'.format(train_acc * 100))
                print('  validation loss: \t\t{:.6f}'.format(validation_loss))
                print('  validation accuracy:\t\t{:.2f}%'.format(validation_acc * 100))
                if FLAGS.seq_name == 'sa':
                    print('  validation Dice LV:\t\t{:.6f}'.format(validation_dice_lv))
                    print('  validation Dice Myo:\t\t{:.6f}'.format(validation_dice_myo))
                    print('  validation Dice RV:\t\t{:.6f}\n'.format(validation_dice_rv))

            else:
                # Print the results for this iteration
                print('Iteration {} of {} took {:.3f}s'.format(iteration, FLAGS.train_iteration,
                                                               time.time() - start_time_iter))
                print('  training loss:\t\t{:.6f}'.format(train_loss))
                print('  training accuracy:\t\t{:.2f}%'.format(train_acc * 100))

            # Save models after every 1000 iterations (1 epoch)
            # One epoch needs to go through
            #   1000 subjects * 2 time frames = 2000 images = 1000 training iterations
            # if one iteration processes 2 images.
            if iteration % 10 == 0:
                saver.save(sess, save_path=os.path.join(model_dir, '{0}'.format(model_name)),
                           global_step=iteration)

        # Close the summary writers
        train_writer.close()
        validation_writer.close()
        print('Training took {:.3f}s in total.\n'.format(time.time() - start_time))


if __name__ == '__main__':
    tf.app.run()
