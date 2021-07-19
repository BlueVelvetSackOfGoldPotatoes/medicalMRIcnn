import os
import urllib.request
import shutil
from pca_control.fit_pca_avg_img import main as fit_pca
from bai_control.train_network import main as train    

def short_axis_segmentation():

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
    os.system('CUDA_VISIBLE_DEVICES={0} python3 deploy_network.py --seq_name sa --data_dir demo_image '
              '--model_path trained_model/FCN_sa'.format(CUDA_VISIBLE_DEVICES))

    # Evaluate ventricular volumes
    print('Evaluating ventricular volumes ...')
    os.system('python3 eval/eval_ventricular_volume.py --data_dir demo_image '
              '--output_csv table_ventricular_volume.csv')

    # Evaluate wall thickness
    print('Evaluating myocardial wall thickness ...')
    os.system('python3 eval/eval_wall_thickness.py --data_dir demo_image '
              '--output_csv table_wall_thickness.csv')
    
    print('Done.')

def main():
    # short_axis_segmentation()
    fit_pca()

if __name__ == '__main__':
    main()