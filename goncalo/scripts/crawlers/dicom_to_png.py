import cv2
import os
import sys, struct
import pydicom
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO

'''
Script that converts dicom to png
'''

def convert():
    inputdir = os.getcwd() + '/'
    outdir = '/home/goncalo/Documents/UMCG/PROJECTS/Goncalo/goncalo/data/umcg_data/PNG_RV/'
    #os.mkdir(outdir)

    dicom_list = [ f for f in  os.listdir(inputdir)]

    for f in dicom_list:   # add "[:10]" to convert 10 images
        file = open(os.path.abspath(f), 'rb')
        ds = pydicom.read_file(file) # read dicom image - it was using f before instead of file

        plt.imshow(ds.pixel_array, cmap=plt.cm.gist_gray)

        base = os.path.basename(f)
        png_file = outdir + os.path.splitext(base)[0] + '.png'

        plt.axis('off')
        plt.savefig(png_file, bbox_inches='tight', pad_inches=0)
        # plt.show()

def main():
    convert()

if __name__ == '__main__':
    main()


############################### AVAILABLE TAG NAMES #######################################
    '''
    print(ds.dir())
        ['AccessionNumber', 'AcquisitionDate', 'AcquisitionDuration', 'AcquisitionMatrix', 'AcquisitionNumber', 'AcquisitionTime', 'AdmittingDiagnosesDescription', 'BitsAllocated', 'BitsStored', 'BodyPartExamined', 'CodeValue', 'CodingSchemeDesignator', 'Columns', 'ContentDate', 'ContentTime', 'DiffusionBValue', 'DiffusionGradientOrientation', 'EchoNumbers', 'EchoTime', 'EchoTrainLength', 'FilmConsumptionSequence', 'FlipAngle', 'FrameOfReferenceUID', 'HeartRate', 'HighBit', 'HighRRValue', 'ImageOrientationPatient', 'ImagePositionPatient', 'ImageType', 'ImagedNucleus', 'ImagingFrequency', 'InPlanePhaseEncodingDirection', 'InstanceCreationDate', 'InstanceCreationTime', 'InstanceCreatorUID', 'InstanceNumber', 'IntervalsAcquired', 'IntervalsRejected', 'LossyImageCompression', 'LowRRValue', 'MRAcquisitionType', 'MagneticFieldStrength', 'Manufacturer', 'ManufacturerModelName', 'Modality', 'NumberOfAverages', 'NumberOfPhaseEncodingSteps', 'NumberOfTemporalPositions', 'OperatorsName', 'OtherPatientIDs', 'PatientBirthDate', 'PatientID', 'PatientName', 'PatientPosition', 'PatientSex', 'PatientWeight', 'PercentPhaseFieldOfView', 'PercentSampling', 'PerformedProcedureStepEndDate', 'PerformedProcedureStepEndTime', 'PerformedProcedureStepID', 'PerformedProcedureStepStartDate', 'PerformedProcedureStepStartTime', 'PerformedProtocolCodeSequence', 'PerformedStationAETitle', 'PhotometricInterpretation', 'PixelBandwidth', 'PixelData', 'PixelRepresentation', 'PixelSpacing', 'PositionReferenceIndicator', 'PresentationLUTShape', 'ProtocolName', 'RealWorldValueMappingSequence', 'ReceiveCoilName', 'ReconstructionDiameter', 'ReferencedImageSequence', 'ReferencedPerformedProcedureStepSequence', 'ReferringPhysicianName', 'RepetitionTime', 'Rows', 'SAR', 'SOPClassUID', 'SOPInstanceUID', 'SamplesPerPixel', 'ScanOptions', 'ScanningSequence', 'ScheduledPerformingPhysicianName', 'SequenceName', 'SequenceVariant', 'SeriesDate', 'SeriesInstanceUID', 'SeriesNumber', 'SeriesTime', 'SliceLocation', 'SliceThickness', 'SoftwareVersions', 'SpacingBetweenSlices', 'SpecificCharacterSet', 'StudyDate', 'StudyID', 'StudyInstanceUID', 'StudyTime', 'TemporalPositionIdentifier', 'TransmitCoilName', 'TriggerTime', 'WindowCenter', 'WindowWidth', 'dBdt']
    '''

    ################## Notes #####################

    '''
def convert_string_to_bytes(string):
    bytes = b''
    for i in string:
        bytes += struct.pack("B", ord(i))
    return bytes

def convert():
    inputdir = os.getcwd() + '/'
    outdir = '/home/goncalo/Documents/UMCG/PROJECTS/Goncalo/goncalo/data/umcg_data/PNG_RV/'
    #os.mkdir(outdir)

    dicom_list = [ f for f in  os.listdir(inputdir)]

    for f in dicom_list[:1]:   # add "[:10]" to convert 10 images
        file = open(os.path.abspath(f), 'rb')
        ds = pydicom.read_file(file) # read dicom image - it was using f before instead of file

        binary = str(ds.PixelData)

        # binary = binary.encode()

        # stream = BytesIO(binary)

        base = os.path.basename(f)
        png_file = outdir + os.path.splitext(base)[0] + '.png'

        # image = Image.open(stream).convert("RGBA")
        # stream.close()
        # image.show()

        stream = BytesIO(convert_string_to_bytes(binary))
        image = Image.open(stream).convert("RGBA")
        stream.close()

        image.save(png_file)

        # print(ds.PixelData.decode('utf-16'))
        # ds.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
        # img = ds.PixelData # get image array
        # cv2.imwrite(outdir + f.replace('.dcm','.png'),img) # write png image

    '''