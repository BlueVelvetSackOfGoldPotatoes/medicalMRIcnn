import os
import urllib.request
import shutil

def short_axis():

   # The GPU device id
    CUDA_VISIBLE_DEVICES = 0

    # The URL for downloading demo data
    URL = 'https://www.doc.ic.ac.uk/~wbai/data/ukbb_cardiac/'

    # Download demo images
    # print('Downloading demo images ...')
    # for i in [1, 2]:
    #     if not os.path.exists('demo_image/{0}'.format(i)):
    #         os.makedirs('demo_image/{0}'.format(i))
    #     for seq_name in ['sa']:
    #         f = 'demo_image/{0}/{1}.nii.gz'.format(i, seq_name)
    #         urllib.request.urlretrieve(URL + f, f)

    # Download information spreadsheet
    # print('Downloading information spreadsheet ...')
    # if not os.path.exists('demo_csv'):
    #     os.makedirs('demo_csv')
    # for f in ['demo_csv/blood_pressure_info.csv']:
    #     urllib.request.urlretrieve(URL + f, f)

    # Download trained models
    # print('Downloading trained models ...')
    # if not os.path.exists('trained_model'):
    #     os.makedirs('trained_model')
    # for model_name in ['FCN_sa']:
    #     for f in ['trained_model/{0}.meta'.format(model_name),
    #               'trained_model/{0}.index'.format(model_name),
    #               'trained_model/{0}.data-00000-of-00001'.format(model_name)]:
    #         urllib.request.urlretrieve(URL + f, f)

    # Analyse show-axis images
    print('******************************')
    print('  Short-axis image analysis')
    print('******************************')

    # Deploy the segmentation network - THIS WORKS!
    print('Deploying the segmentation network ...')
    os.system('CUDA_VISIBLE_DEVICES={0} python3 common/deploy_network.py --seq_name sa --data_dir goncalo/thesis_project/files_for_thesis/data/demo_image '
              '--model_path trained_model/FCN_sa'.format(CUDA_VISIBLE_DEVICES))

    # Evaluate ventricular volumes - THIS WORKS!
    print('Evaluating ventricular volumes ...')
    os.system('python3 short_axis/eval_ventricular_volume.py --data_dir demo_image '
              '--output_csv goncalo/thesis_project/files_for_thesis/results/table_ventricular_volume.csv')

    # Evaluate wall thickness - THIS WORKS!
    print('Evaluating myocardial wall thickness ...')
    os.system('python3 short_axis/eval_wall_thickness.py --data_dir demo_image '
              '--output_csv goncalo/thesis_project/files_for_thesis/results/table_wall_thickness.csv')
    
    print('Done.')

def main():
    short_axis()

if __name__ == '__main__':
    main()