import pydicom
import dicom2nifti
import dicom2nifti.settings as settings
import matplotlib.pyplot as plt
from pydicom import dcmread
from pydicom.pixel_data_handlers.util import apply_voi_lut

# Enable single slice conversion dicom2nifti
settings.disable_validate_slicecount()

def plot_dicom_windowed(dicom_path):
    ds = dicom2image(dicom_path)
    if 'WindowWidth' in ds:
        windowed = apply_voi_lut(ds.pixel_array, ds)
        print("Showing windowed")
        show_img(windowed)
    else:
        print("No need to show window")
        show_img(ds)

# Plot dicom image
def show_dicom_using_plt(ds):
    plt.imshow(ds.pixel_array, cmap=plt.cm.bone) 

# Read dicom file
def readDicom(dicom_path):
    ds = pydicom.dcmread(dicom_path)
    return ds

# Translate whole dicom dir to nifti
def dicom2nifti(in_dir, out_dir):
    dicom2nifti.convert_directory(in_dir, out_dir, compression=True, reorient=True)