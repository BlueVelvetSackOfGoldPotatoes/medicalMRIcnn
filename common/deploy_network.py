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
# ============================================================================
from operator import truediv
import os
from os import path
import glob
from pickle import FALSE
import shutil
import time
import math
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import PIL
import tensorflow.compat.v1 as tf
from PIL import Image
from PIL import ImageOps
from image_utils import rescale_intensity
from tensorflow.keras.applications.vgg16 import VGG16

""" Deployment parameters """
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_enum('seq_name', 'sa',
                         ['sa', 'la_2ch', 'la_4ch'],
                         'Sequence name.')
tf.app.flags.DEFINE_string('model_path',
                           '/home/goncalo/Documents/RUG/4th Year/2B/thesis/medicalMRIcnn/trained_model/FCN_sa',
                           'Path to the saved trained model.')
tf.app.flags.DEFINE_boolean('process_seq', True,
                            'Process a time sequence of images.')
tf.app.flags.DEFINE_boolean('save_seg', False,
                            'Save segmentation.')
tf.app.flags.DEFINE_boolean('seg4', False,
                            'Segment all the 4 chambers in long-axis 4 chamber view. '
                            'This seg4 network is trained using 200 subjects from Application 18545.'
                            'By default, for all the other tasks (ventricular segmentation'
                            'on short-axis images and atrial segmentation on long-axis images,'
                            'the networks are trained using 3,975 subjects from Application 2964.')
# THESIS FLAGS
tf.app.flags.DEFINE_boolean('make_avg_matrix', False,
                            'Code for building the avg matrix')
tf.app.flags.DEFINE_boolean('make_seg_matrices', True,
                            'Code for building segmentation matrices')
tf.app.flags.DEFINE_boolean('deploy_updated_model', False,
                            'Process data using loaded updated model instead of pipeline')
tf.app.flags.DEFINE_boolean('UMCG_data', False,
                            'If true use umcg data, if false use ukbb data')

# Set dataset:
if FLAGS.UMCG_data:
    tf.app.flags.DEFINE_string('data_dir', '/home/goncalo/Documents/RUG/4th Year/2B/thesis/medicalMRIcnn/goncalo/thesis_project/files_for_thesis/data/data_umcg',
                            'Path to the data set directory, under which images '
                            'are organised in subdirectories for each subject.')
else:
    tf.app.flags.DEFINE_string('data_dir', '/home/goncalo/Documents/RUG/4th Year/2B/thesis/medicalMRIcnn/demo_image',
                            'Path to the data set directory, under which images '
                            'are organised in subdirectories for each subject.')

def create_model_from_vgg16_bai_weights():
    saver = tf.train.import_meta_graph('{0}/FCN_sa.meta'.format(FLAGS.model_path))
    sess = tf.Session()
    # Top = False means we're not loading the classification layers (decoder).
    model = VGG16(include_top=True, weights=saver.restore(sess, '{0}/FCN_sa'.format(FLAGS.model_path)))

    # # Handcarving vgg16 layers to match bai's...
    tmp = model.get_config()

    tmp['layers'][0]['config']['batch_input_shape'] = (None, 192, 192, 16)
    tmp['layers'][1]['config']['batch_input_shape'] = (None, 192, 192, 16)
    tmp['layers'][3]['config']['batch_input_shape'] = (None, 96, 96, 32)
    tmp['layers'][4]['config']['batch_input_shape'] = (None, 96, 96, 32)
    tmp['layers'][6]['config']['batch_input_shape'] = (None, 48, 48, 64)
    tmp['layers'][7]['config']['batch_input_shape'] = (None, 48, 48, 64)
    tmp['layers'][8]['config']['batch_input_shape'] = (None, 48, 48, 64)
    tmp['layers'][10]['config']['batch_input_shape'] = (None, 24, 24, 128)
    tmp['layers'][11]['config']['batch_input_shape'] = (None, 24, 24, 128)
    tmp['layers'][12]['config']['batch_input_shape'] = (None, 24, 24, 128)
    tmp['layers'][14]['config']['batch_input_shape'] = (None, 12, 12, 256)
    tmp['layers'][15]['config']['batch_input_shape'] = (None, 12, 12, 256)
    tmp['layers'][16]['config']['batch_input_shape'] = (None, 12, 12, 256)
    
    model = model.from_config(tmp)
    # model.save("./vgg16/bai_arch_top_off")
    return model

if __name__ == '__main__':
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Import the computation graph and restore the variable values
        saver = tf.train.import_meta_graph('{0}.meta'.format(FLAGS.model_path))
        saver.restore(sess, '{0}'.format(FLAGS.model_path))
        # saver.save(sess, 'FCN_sa')

        ################################## save model variables - THESIS v
        # variables_names = [v.name for v in tf.trainable_variables()]
        # values = sess.run(variables_names)
        
        # with open("/home/goncalo/Documents/RUG/4th Year/2B/thesis/medicalMRIcnn/goncalo/thesis_project/files_for_thesis/trained_model/model_archs/bai_loaded_saver.txt" ,"a+") as f:
        #     for k, v in zip(variables_names, values):
        #         string = "Variable: " + str(k)
        #         string = string + "/nShape: " + str(v.shape)
        #         string = string + "/n" + str(v)
        #         string = string + "/n"
        #         f.write(string)
        ################################## save model variables - THESIS ^

        print('Start deployment on the data set ...')
        start_time = time.time()

        # Process each subject subdirectory
        data_list = sorted(os.listdir(FLAGS.data_dir))
        processed_list = []
        table_time = []
        for data in data_list:
            print(data)
            data_dir = os.path.join(FLAGS.data_dir, data)

            # MAKE VANILLA MATRIX FOLDER THESIS -------------------------------------------
            if FLAGS.make_avg_matrix:
                list_matrice_paths = []
                if not path.exists("{0}/vanilla_matrice_imgs".format(data_dir)):
                    os.mkdir("{0}/vanilla_matrice_imgs".format(data_dir))
                path_to_matrices = "{0}/vanilla_matrice_imgs".format(data_dir)
                list_matrice_paths.append(path_to_matrices)
            # MAKE VANILLA MATRIX FOLDER THESIS -------------------------------------------

            # MAKE SEG MATRIX FOLDER THESIS -----------------------------------------------
            if FLAGS.make_seg_matrices:
                list_seg_matrice_paths = []
                if not path.exists("{0}/seg_matrice_imgs".format(data_dir)):
                    os.mkdir("{0}/seg_matrice_imgs".format(data_dir))
                path_to_seg_matrices = "{0}/seg_matrice_imgs".format(data_dir)
                list_seg_matrice_paths.append(path_to_seg_matrices)
                print('list of seg matrices:',list_seg_matrice_paths)
            # MAKE SEG MATRIX FOLDER THESIS -----------------------------------------------

            if FLAGS.seq_name == 'la_4ch' and FLAGS.seg4:
                seg_name = '{0}/seg4_{1}.nii.gz'.format(data_dir, FLAGS.seq_name)
            else:
                seg_name = '{0}/seg_{1}.nii.gz'.format(data_dir, FLAGS.seq_name)
            # if os.path.exists(seg_name):
            #     continue

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
                print(image.shape)
                X, Y, Z, T = image.shape
                orig_image = image

                ''' These matrices will be used to calculate avg vector '''
                # Writing matrix to file THESIS ----------------------------------------------------
                if FLAGS.make_avg_matrix:
                    for t in range(T):
                        images_tosave = np.array(image)
                        images_tosave = image[:, :, :, t]
                        image_n = images_tosave.shape[2]
                        # because this is a nifti of image_n images
                        for i in range(image_n):
                            image_tosave = images_tosave[:, :, i]
                            # These are vanilla images (without any changes) as can be observed 
                            # plt.imshow(image_tosave, cmap='gray')
                            # plt.show()
                            name_tosave = path_to_matrices + '/vanilla_matrice_img' + '_' + str(i)
                            np.save(str(name_tosave), image_tosave)
                        # Copy matrices to  
                        for file in glob.glob(os.path.join(path_to_matrices,"*.pyn")):
                            shutil.copy2(file,'/home/goncalo/Documents/RUG/4th Year/2B/thesis/medicalMRIcnn/common/img_matrixes')

                # Writing matrix to file THESIS ----------------------------------------------------
                print('  Segmenting full sequence ...')
                start_seg_time = time.time()

                # Intensity rescaling
                image = rescale_intensity(image, (1, 99))

                # Prediction (segmentation)
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
                    if FLAGS.deploy_updated_model:
                        # I miss the probabilities here but they are not used by Bai
                        model = create_model_from_vgg16_bai_weights()
                        pred_fr = model.predict(image_fr)
                    else:
                        prob_fr, pred_fr = sess.run(['prob:0', 'pred:0'],
                            feed_dict={'image:0': image_fr, 'training:0': False})

                    # Transpose and crop segmentation to recover the original size
                    pred_fr = np.transpose(pred_fr, axes=(1, 2, 0))
                    pred_fr = pred_fr[x_pre:x_pre + X, y_pre:y_pre + Y]
                    pred[:, :, :, t] = pred_fr

                    # See colored predictions
                    # for z in range(Z):
                    #     print('Observing {}:{}'.format(z, t))
                    #     plt.imshow(pred[:, :, z, t])
                    #     plt.show()

                    # Finding bottleneck mask THESIS ----------------------------------------------------
                    if FLAGS.make_seg_matrices:
                        # print("image_fr: ", image_fr)
                        # print("prob_fr: ", prob_fr)

                        # print(" Plotting image...")
                        # # (12, 224, 208, 1)
                        image_fr_img = np.squeeze(image_fr, axis=3)
                        # # (224, 208, 12)

                        image_fr_img = np.transpose(image_fr_img, (1,2,0))
                        image_fr_z = image_fr_img.shape[2]
                        labels_prob = ["LV-CAVITY, RV-MYOCARDIUM, and RV-CAVITY", "LV-CAVITY", "RV-MYOCARDIUM", "RV-CAVITY"]
                        for i in range(image_fr_z):
                            # plt.imshow(image_fr_img[:, :, i], cmap='gray')
                            # plt.show()

                            # print(" Plotting probs...")
                            # (12, 224, 208, 4)
                            prob_fr_img_prob = []
                            prob_fr_img_prob.append(prob_fr[:,:,:,0])
                            prob_fr_img_prob.append(prob_fr[:,:,:,1])
                            prob_fr_img_prob.append(prob_fr[:,:,:,2])
                            prob_fr_img_prob.append(prob_fr[:,:,:,3])

                            # Saving LV seg from bottleneck
                            # prob_fr_img = prob_fr[i,:,:,1]
                            # prob_fr_img = np.array(prob_fr_img)
                            # np.save(str(name_tosave), prob_fr_img)

                        # Copy matrices to  
                        # for file in glob.glob(os.path.join(path_to_seg_matrices,"*.pyn")):
                        #     shutil.copy2(file,'/home/goncalo/Documents/RUG/4th Year/2B/thesis/medicalMRIcnn/common/seg_img_matrixes')

                            label_index = 0
                            for img_prob in prob_fr_img_prob:
                                # print(labels_prob[label_index])
                                if labels_prob[label_index] == 'LV-CAVITY':
                                # (224, 208, 12)
                                    prob_fr_img = np.transpose(img_prob, (1,2,0))
                                    plt.imshow(prob_fr_img[:, :, i], cmap='gray')
                                    # plt.show()
                                    name_tosave = path_to_seg_matrices + '/seg_matrice_img' + '_' + str(i) + '_' + str(t)
                                    plt.savefig(name_tosave + '.png')
                                label_index += 1

                    # Finding bottleneck THESIS ----------------------------------------------------
                exit()
                seg_time = time.time() - start_seg_time
                print('  Segmentation time = {:3f}s'.format(seg_time))
                table_time += [seg_time]
                processed_list += [data]

                # ED frame defaults to be the first time frame.
                # Determine ES frame according to the minimum LV volume.
                k = {}
                k['ED'] = 0
                if FLAGS.seq_name == 'sa' or (FLAGS.seq_name == 'la_4ch' and FLAGS.seg4):
                    k['ES'] = np.argmin(np.sum(pred == 1, axis=(0, 1, 2)))
                else:
                    k['ES'] = np.argmax(np.sum(pred == 1, axis=(0, 1, 2)))
                print('  ED frame = {:d}, ES frame = {:d}'.format(k['ED'], k['ES']))

                # Save the segmentation
                if FLAGS.save_seg:
                    print('  Saving segmentation ...')
                    nim2 = nib.Nifti1Image(pred, nim.affine)
                    nim2.header['pixdim'] = nim.header['pixdim']
                    if FLAGS.seq_name == 'la_4ch' and FLAGS.seg4:
                        seg_name = '{0}/seg4_{1}.nii.gz'.format(data_dir, FLAGS.seq_name)
                    else:
                        seg_name = '{0}/seg_{1}.nii.gz'.format(data_dir, FLAGS.seq_name)
                    nib.save(nim2, seg_name)

                    for fr in ['ED', 'ES']:
                        nib.save(nib.Nifti1Image(orig_image[:, :, :, k[fr]], nim.affine),
                                 '{0}/{1}_{2}.nii.gz'.format(data_dir, FLAGS.seq_name, fr))
                        if FLAGS.seq_name == 'la_4ch' and FLAGS.seg4:
                            seg_name = '{0}/seg4_{1}_{2}.nii.gz'.format(data_dir, FLAGS.seq_name, fr)
                        else:
                            seg_name = '{0}/seg_{1}_{2}.nii.gz'.format(data_dir, FLAGS.seq_name, fr)
                        nib.save(nib.Nifti1Image(pred[:, :, :, k[fr]], nim.affine), seg_name)
            else:
                # Process ED and ES time frames
                image_ED_name = '{0}/{1}_{2}.nii.gz'.format(data_dir, FLAGS.seq_name, 'ED')
                image_ES_name = '{0}/{1}_{2}.nii.gz'.format(data_dir, FLAGS.seq_name, 'ES')
                if not os.path.exists(image_ED_name) or not os.path.exists(image_ES_name):
                    print('  Directory {0} does not contain an image with '
                          'file name {1} or {2}. Skip.'.format(data_dir,
                                                               os.path.basename(image_ED_name),
                                                               os.path.basename(image_ES_name)))
                    continue

                measure = {}
                for fr in ['ED', 'ES']:
                    image_name = '{0}/{1}_{2}.nii.gz'.format(data_dir, FLAGS.seq_name, fr)

                    # Read the image
                    print('  Reading {} ...'.format(image_name))
                    nim = nib.load(image_name)
                    image = nim.get_data()
                    X, Y = image.shape[:2]
                    if image.ndim == 2:
                        image = np.expand_dims(image, axis=2)

                    print('  Segmenting {} frame ...'.format(fr))
                    start_seg_time = time.time()

                    # Intensity rescaling
                    image = rescale_intensity(image, (1, 99))

                    # Pad the image size to be a factor of 16 so that
                    # the downsample and upsample procedures in the network
                    # will result in the same image size at each resolution
                    # level.
                    X2, Y2 = int(math.ceil(X / 16.0)) * 16, int(math.ceil(Y / 16.0)) * 16
                    x_pre, y_pre = int((X2 - X) / 2), int((Y2 - Y) / 2)
                    x_post, y_post = (X2 - X) - x_pre, (Y2 - Y) - y_pre
                    image = np.pad(image, ((x_pre, x_post), (y_pre, y_post), (0, 0)), 'constant')

                    # Transpose the shape to NXYCd

                    # Evaluate the network
                    prob, pred = sess.run(['prob:0', 'pred:0'],
                                          feed_dict={'image:0': image, 'training:0': False})

                    # Transpose and crop the segmentation to recover the original size
                    pred = np.transpose(pred, axes=(1, 2, 0))
                    pred = pred[x_pre:x_pre + X, y_pre:y_pre + Y]

                    seg_time = time.time() - start_seg_time
                    print('  Segmentation time = {:3f}s'.format(seg_time))
                    table_time += [seg_time]
                    processed_list += [data]

                    # Save the segmentation
                    if FLAGS.save_seg:
                        print('  Saving segmentation ...')
                        nim2 = nib.Nifti1Image(pred, nim.affine)
                        nim2.header['pixdim'] = nim.header['pixdim']
                        if FLAGS.seq_name == 'la_4ch' and FLAGS.seg4:
                            seg_name = '{0}/seg4_{1}_{2}.nii.gz'.format(data_dir, FLAGS.seq_name, fr)
                        else:
                            seg_name = '{0}/seg_{1}_{2}.nii.gz'.format(data_dir, FLAGS.seq_name, fr)
                        nib.save(nim2, seg_name)

        if FLAGS.process_seq:
            print('Average segmentation time = {:.3f}s per sequence'.format(np.mean(table_time)))
        else:
            print('Average segmentation time = {:.3f}s per frame'.format(np.mean(table_time)))
        process_time = time.time() - start_time
        
        # Averaging matrix to file THESIS ----------------------------------------------------
        # If folder is not empty then load matrices and calculate avg matrix
        if FLAGS.make_avg_matrix:
            for matrice_path in list_matrice_paths:
                if len(os.listdir(matrice_path)):
                    list_matrices = []
                    # i is each matrix file
                    for i in os.listdir(matrice_path):
                        image_i_path = os.path.join(matrice_path, i)
                        list_matrices.append(np.load(image_i_path))
                    avg_vector = np.mean(list_matrices, axis=0)
                    name_tosave = '/home/goncalo/Documents/RUG/4th Year/2B/thesis/medicalMRIcnn/common/img_matrixes/vanilla_avg_vector'
                    np.save(str(name_tosave), avg_vector)
        # Averaging matrix to file THESIS ----------------------------------------------------

        # Saving seg matrix to general folder THESIS ----------------------------------------------------
        # If folder is not empty then load matrices, normalize them and calculate avg matrix
        # if FLAGS.make_seg_matrices:
        #     subject_count = 1
        #     for matrice_path in list_seg_matrice_paths:
        #         if len(os.listdir(matrice_path)):
        #             # i is each matrix file
        #             for i in os.listdir(matrice_path):
        #                 image_i_path = os.path.join(matrice_path, i)
        #                 new_name = image_i_path[:-4] + '_' + str(subject_count) + image_i_path[-4:]
        #                 os.rename(image_i_path, new_name)
        #                 shutil.copy2(new_name,'/home/goncalo/Documents/RUG/4th Year/2B/thesis/medicalMRIcnn/common/seg_img_matrixes')
        #         subject_count = subject_count + 1

        # Saving seg matrix to general folder THESIS ----------------------------------------------------

        # print('Including image I/O, CUDA resource allocation, '
        #       'it took {:.3f}s for processing {:d} subjects ({:.3f}s per subjects).'.format(
        #     process_time, len(processed_list), process_time / len(processed_list)))
