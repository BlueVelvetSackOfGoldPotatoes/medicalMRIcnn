import os
import time
import math
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import tensorflow.compat.v1 as tf
from image_utils import rescale_intensity
# import nilearn.plotting as nip
from niwidgets import NiftiWidget

""" Deployment parameters """
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_enum('seq_name', 'sa',
                         ['sa'],
                         'Sequence name.')
# Bai default data_dir: '/vol/bitbucket/wbai/own_work/ukbb_cardiac_demo'
tf.app.flags.DEFINE_string('data_dir', '../demo_image',
                           'Path to the data set directory, under which images '
                           'are organised in subdirectories for each subject.')
tf.app.flags.DEFINE_string('model_path',
                           '../trained_model/FCN_sa',
                           'Path to the saved trained model.')
tf.app.flags.DEFINE_boolean('process_seq', True,
                            'Process a time sequence of images.')
tf.app.flags.DEFINE_boolean('save_seg', True,
                            'Save segmentation.')
tf.app.flags.DEFINE_boolean('plot_stuff', False,
                            'Plot segmentations and other images.')

if __name__ == '__main__':
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Import the computation graph and restore the variable values
        saver = tf.train.import_meta_graph('{0}.meta'.format(FLAGS.model_path))
        saver.restore(sess, '{0}'.format(FLAGS.model_path))

        print('Start deployment on the data set ...')
        start_time = time.time()

        # Process each subject subdirectory
        data_list = sorted(os.listdir(FLAGS.data_dir))
        processed_list = []
        table_time = []
        for data in data_list:
            print(data)
            data_dir = os.path.join(FLAGS.data_dir, data)
            
            seg_name = '{0}/seg_{1}.nii.gz'.format(data_dir, FLAGS.seq_name)

            if FLAGS.process_seq:
                # Process the temporal sequence
                image_name = '{0}/{1}.nii.gz'.format(data_dir, FLAGS.seq_name)

                if not os.path.exists(image_name):
                    print('  Directory {0} does not contain an image with file '
                          'name {1}. Skip.'.format(data_dir, os.path.basename(image_name)))
                    continue

                # Read the image
                print('  Reading {} ...'.format(image_name))
                nim = nib.load(image_name)
                image = nim.get_data()
                X, Y, Z, T = image.shape
                orig_image = image

                print('  Segmenting full sequence ...')
                start_seg_time = time.time()

                # Intensity rescaling
                image = rescale_intensity(image, (1, 99))

                # Prediction (segmentation) zero array of image.shape size
                pred = np.zeros(image.shape)

                # Pad the image size to be a factor of 16 so that the
                # downsample and upsample procedures in the network will
                # result in the same image size at each resolution level.
                X2, Y2 = int(math.ceil(X / 16.0)) * 16, int(math.ceil(Y / 16.0)) * 16
                x_pre, y_pre = int((X2 - X) / 2), int((Y2 - Y) / 2)
                x_post, y_post = (X2 - X) - x_pre, (Y2 - Y) - y_pre
                image = np.pad(image, ((x_pre, x_post), (y_pre, y_post), (0, 0), (0, 0)), 'constant')

                # Process each time frame
                for t in range(T):
                    # Transpose the shape to NXYC
                    image_fr = image[:, :, :, t]
                    image_fr = np.transpose(image_fr, axes=(2, 0, 1)).astype(np.float32)
                    image_fr = np.expand_dims(image_fr, axis=-1)

                    # Evaluate the network
                    prob_fr, pred_fr = sess.run(['prob:0', 'pred:0'],
                                                feed_dict={'image:0': image_fr, 'training:0': False})

                    # Transpose and crop segmentation to recover the original size
                    pred_fr = np.transpose(pred_fr, axes=(1, 2, 0))
                    pred_fr = pred_fr[x_pre:x_pre + X, y_pre:y_pre + Y]
                    pred[:, :, :, t] = pred_fr

                    # Finding bottleneck ---------------------------------------------------- GC THESIS
                    if FLAGS.plot_stuff:
                        print(" Plotting image...")
                        # (12, 224, 208, 1)
                        image_fr_img = np.squeeze(image_fr, axis=3)
                        # (224, 208, 12)
                        image_fr_img = np.transpose(image_fr_img, (1,2,0))
                        labels_prob = ["LV-CAVITY, RV-MYOCARDIUM, and RV-CAVITY", "LV-CAVITY", "RV-MYOCARDIUM", "RV-CAVITY"]
                        for i in range(12):
                            plt.imshow(image_fr_img[:, :, i], cmap='gray')
                            plt.show()

                            print(" Plotting probs...")
                            # (12, 224, 208, 4)
                            prob_fr_img_prob = []
                            prob_fr_img_prob.append(prob_fr[:,:,:,0])
                            prob_fr_img_prob.append(prob_fr[:,:,:,1])
                            prob_fr_img_prob.append(prob_fr[:,:,:,2])
                            prob_fr_img_prob.append(prob_fr[:,:,:,3])
                            prob_fr_img_prob = np.array(prob_fr_img_prob)
                            label_index = 0
                            for img_prob in prob_fr_img_prob:
                                print(labels_prob[label_index])
                                # (224, 208, 12)
                                prob_fr_img = np.transpose(img_prob, (1,2,0))
                                plt.imshow(prob_fr_img[:, :, i], cmap='gray')
                                plt.show()
                                label_index += 1

                seg_time = time.time() - start_seg_time
                # processed_list += [data]

                """
                Saves segmentation result ----------------------------------------------------------------------
                """
                # k = {}
                # k['ED'] = 0
                # k['ES'] = np.argmin(np.sum(pred == 1, axis=(0, 1, 2)))


                # print('  Saving segmentation matrix ...')

                # nim2 = nib.Nifti1Image(pred, nim.affine)
                # nim2.header['pixdim'] = nim.header['pixdim']

                # for fr in ['ED', 'ES']:
                #     nib.save(nib.Nifti1Image(orig_image[:, :, :, k[fr]], nim.affine),
                #                 '{0}/{1}_{2}.nii.gz'.format(data_dir, FLAGS.seq_name, fr))

            # Works only in jupyter notebooks
            # nim2_widget = NiftiWidget(nim2)
            # nim2_widget.nifti_plotter(plotting_func=nip.plot_img, display_mode=['ortho', 'x', 'y', 'z'], colormap='hot')
