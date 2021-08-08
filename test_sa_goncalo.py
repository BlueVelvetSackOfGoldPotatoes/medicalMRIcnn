import os
import urllib.request
import shutil

def train_short_axis():
    # Analyse short-axis images
    print('*******************************')
    print(' Short-axis network retraining ')
    print('*******************************')

    # Deploy the transfer learning script
    print('Deploying the training environment...')
    os.system('python3 common/train_network.py')

def short_axis():

   # The GPU device id
    CUDA_VISIBLE_DEVICES = 0

    # The URL for downloading demo data
    URL = 'https://www.doc.ic.ac.uk/~wbai/data/ukbb_cardiac/'

    # Analyse show-axis images
    print('******************************')
    print('  Short-axis image analysis')
    print('******************************')

    # Deploy the segmentation network
    print('Deploying the segmentation network ...')
    os.system('CUDA_VISIBLE_DEVICES={0} python3 common/deploy_network.py --seq_name sa --data_dir goncalo/thesis_project/files_for_thesis/data/demo_image '
              '--model_path trained_model/FCN_sa'.format(CUDA_VISIBLE_DEVICES))

    # Evaluate ventricular volumes
    print('Evaluating ventricular volumes ...')
    os.system('python3 short_axis/eval_ventricular_volume.py --data_dir demo_image '
              '--output_csv goncalo/thesis_project/files_for_thesis/results/table_ventricular_volume.csv')

    # Evaluate wall thickness
    print('Evaluating myocardial wall thickness ...')
    os.system('python3 short_axis/eval_wall_thickness.py --data_dir demo_image '
              '--output_csv goncalo/thesis_project/files_for_thesis/results/table_wall_thickness.csv')
    
    print('Done.')

def main():
    # short_axis()
    train_short_axis()

if __name__ == '__main__':
    main()