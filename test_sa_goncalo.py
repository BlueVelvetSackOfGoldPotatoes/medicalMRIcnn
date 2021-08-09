import os
import urllib.request
import shutil
from goncalo.thesis_project.files_for_thesis.scripts.file_control import convert_dataset as conv_dat

def retrain_short_axis():
    # Analyse short-axis images
    print('*******************************')
    print(' Short-axis network retraining ')
    print('*******************************')

    # Deploy the transfer learning script
    print('Deploying the training environment...')
    os.system('python3 common/retrain_network.py')

def train_short_axis():
    # Analyse short-axis images
    print('*******************************')
    print('  Short-axis network training  ')
    print('*******************************')

    # Deploy the transfer learning script
    print('Deploying the training environment...')
    os.system('python3 common/train_network.py')

def short_axis_retrained():

   # The GPU device id
    CUDA_VISIBLE_DEVICES = 0

    # The URL for downloading demo data
    URL = 'https://www.doc.ic.ac.uk/~wbai/data/ukbb_cardiac/'

    # Analyse show-axis images
    print('***************************************')
    print('  Short-axis image analysis retrained  ')
    print('***************************************')

    '''
    Deploying retrained model
    '''
    # Deploy the segmentation network
    print('Deploying the segmentation network ...')
    os.system('CUDA_VISIBLE_DEVICES={0} python3 common/deploy_network.py --seq_name sa --data_dir /home/goncalo/Documents/RUG/4th\ Year/2B/thesis/medicalMRIcnn/goncalo/thesis_project/files_for_thesis/data/demo_image '
              '--model_path /home/goncalo/Documents/RUG/4th\ Year/2B/thesis/medicalMRIcnn/goncalo/thesis_project/files_for_thesis/checkpoint/FCN_sa/FCN_sa'.format(CUDA_VISIBLE_DEVICES))

    # Evaluate ventricular volumes
    print('Evaluating ventricular volumes ...')
    os.system('python3 short_axis/eval_ventricular_volume.py --data_dir /home/goncalo/Documents/RUG/4th\ Year/2B/thesis/medicalMRIcnn/goncalo/thesis_project/files_for_thesis/data/demo_image '
            '--output_csv goncalo/thesis_project/files_for_thesis/results/table_ventricular_volume.csv')

    # Evaluate wall thickness
    print('Evaluating myocardial wall thickness ...')
    os.system('python3 short_axis/eval_wall_thickness.py --data_dir /home/goncalo/Documents/RUG/4th\ Year/2B/thesis/medicalMRIcnn/goncalo/thesis_project/files_for_thesis/data/demo_image '
            '--output_csv goncalo/thesis_project/files_for_thesis/results/table_wall_thickness.csv')
    
    print('Done.')

def short_axis():

   # The GPU device id
    CUDA_VISIBLE_DEVICES = 0

    # The URL for downloading demo data
    URL = 'https://www.doc.ic.ac.uk/~wbai/data/ukbb_cardiac/'

    # Analyse show-axis images
    print('*****************************')
    print('  Short-axis image analysis  ')
    print('*****************************')

    '''
    Deploying Bai trained model
    '''
    # Deploy the segmentation network
    print('Deploying the segmentation network ...')
    os.system('CUDA_VISIBLE_DEVICES={0} python3 common/deploy_network.py --seq_name sa --data_dir demo_image '
              '--model_path trained_model/FCN_sa'.format(CUDA_VISIBLE_DEVICES))

    # Evaluate ventricular volumes
    print('Evaluating ventricular volumes ...')
    os.system('python3 short_axis/eval_ventricular_volume.py --data_dir demo_image '
            '--output_csv demo_csv/table_ventricular_volume.csv')

    # Evaluate wall thickness
    print('Evaluating myocardial wall thickness ...')
    os.system('python3 short_axis/eval_wall_thickness.py --data_dir demo_image '
              '--output_csv demo_csv/table_wall_thickness.csv')
    
    print('Done.')

def main():
    # short_axis_retrained()
    # short_axis()
    # train_short_axis()
    # retrain_short_axis()
    conv_dat.nifti_to_jpg_dataset("/home/goncalo/Documents/RUG/4th Year/2B/thesis/medicalMRIcnn/goncalo/thesis_project/files_for_thesis/data/demo_image/2/", "/home/goncalo/Documents/RUG/4th Year/2B/thesis/medicalMRIcnn/goncalo/thesis_project/files_for_thesis/data/jpg_retrained_seg")

if __name__ == '__main__':
    main()