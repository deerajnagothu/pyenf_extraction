# Author: Deeraj Nagothu
# ENF estimation from video recordings using Rolling Shutter Mechanism

# Import required packages
import numpy as np
import cv2
import pickle
import pyenf
#from scipy import signal, io
import scipy.io.wavfile
import math
from scipy.fftpack import fftshift
import matplotlib.pyplot as plt
import librosa
from skimage.util import img_as_float
from skimage.segmentation import slic
from scipy.stats.stats import pearsonr

# Constants
open_video_to_extract_Row_signal = False  # set it to True to extract, else False to use the dump file
video_folder = "Recordings/2022/HomeRec/Deepfake1/"
video_rec_name = "deepfake1.mp4"
#video_folder = "/home/deeraj/Documents/Projects/pyENF_extraction_rolling_shutter/Recordings/Deepfake/set1/original/"
#video_rec_name = "01__talking_against_wall.mp4"
power_rec_name = "power_deepfake1.wav"
numSegments = 500  # number of superpixel segments per frame
# video_rec_name = "resized_MVI_0288.avi"
# power_rec_name = "80D_power_recording_3_20min.wav"
video_filepath = video_folder + video_rec_name
power_filepath = video_folder + power_rec_name
do_ssm = 0  # decides if SSM should be applied or not to a code.
motion_detection_threshold = 40  # threshold decides after how many pixel changes, it should apply Superpixel mask
window_size = 30
shift_size = 5

def SSM(frame, frame_segments, new_frame_with_mask, motion_detection_threshold, ones_Superpixel_mask):
    motion_threshold = np.count_nonzero(new_frame_with_mask)  # count how many pixels were effected
    # print(motion_threshold)
    if motion_threshold >= motion_detection_threshold:  # no. of pixels effected more than threshold then apply mask
        new_frame_with_mask[new_frame_with_mask == 255] = 1  # 255 represents white pixels which are motion detections
        new_frame_with_mask[new_frame_with_mask == 127] = 1  # 127 represents gray pixels which are shadow of object
        superpixel_motion_mask = np.multiply(frame_segments,
                                             new_frame_with_mask)  # multiplying to see which superpixels were effected
        effected_superpixels = np.unique(superpixel_motion_mask)
        for each_superpixel in effected_superpixels:  # all the effected superpixels are set to zero
            frame_segments[frame_segments == each_superpixel] = 0
        ones_Superpixel_mask[frame_segments == 0] = 0
        if frame.shape[2] == 3:  # its RGB frame, so the mask is applied to each layer individually
            frame[:, :, 0] = np.multiply(ones_Superpixel_mask, frame[:, :, 0])
            frame[:, :, 1] = np.multiply(ones_Superpixel_mask, frame[:, :, 1])
            frame[:, :, 2] = np.multiply(ones_Superpixel_mask, frame[:, :, 2])
        else:  # for grayscale
            frame = np.multiply(ones_Superpixel_mask, frame)
    return ones_Superpixel_mask,frame

def correlation_vector(ENF_signal1, ENF_signal2, window_size, shift_size):
    correlation_ENF = []
    length_of_signal = min(len(ENF_signal1), len(ENF_signal2))
    total_windows = math.ceil(( length_of_signal - window_size + 1) / shift_size)
    rho = np.zeros((1,total_windows))
    for i in range(0,total_windows):
        enf_sig1 = ENF_signal1[i * shift_size: i * shift_size + window_size]
        enf_sig2 = ENF_signal2[i * shift_size: i * shift_size + window_size]
        enf_sig1 = np.reshape(enf_sig1, (len(enf_sig1),))
        enf_sig2 = np.reshape(enf_sig2,(len(enf_sig2), ))
        r,p = pearsonr(enf_sig1, enf_sig2)
        rho[0][i] = r
    return rho,total_windows

def extract_row_pixel_with_SSM(frame):
    # check for grayscale or RGB
    frame_shape = frame.shape
    if frame_shape[2] == 3:  # its an RGB frame
        average_frame_across_rgb = np.mean(frame, axis=2)
        dup_average_frame_across_rgb = average_frame_across_rgb.astype(np.float32)
        dup_average_frame_across_rgb[dup_average_frame_across_rgb == 0] = np.nan
        dup_average_frame_across_column = np.nanmean(dup_average_frame_across_rgb, axis=1)
        dup_average_frame_across_column[np.isnan(dup_average_frame_across_column)] = 0
        average_frame_across_column = dup_average_frame_across_column.astype(np.int32)
    else:
        dup_frame = frame.astype(np.float32)
        dup_frame[dup_frame == 0] = np.nan
        dup_average_frame_across_column = np.nanmean(dup_frame, axis=1)
        average_frame_across_column = dup_average_frame_across_column.astype(np.int32)
    average_frame_across_column = np.reshape(average_frame_across_column, (frame_shape[0],))
    return average_frame_across_column

def second_half_extract_row_pixel(frame):
    # check for grayscale or RGB
    frame_shape = frame.shape
    if frame_shape[2] == 3:  # its an RGB frame
        average_frame_across_rgb = np.mean(frame, axis=2)
        average_frame_across_column = np.mean(average_frame_across_rgb[:,:1000], axis=1)
    else:
        average_frame_across_column = np.mean(frame, axis=1)
    average_frame_across_column = np.reshape(average_frame_across_column, (frame_shape[0],))
    return average_frame_across_column

def extract_row_pixel(frame):
    # check for grayscale or RGB
    frame_shape = frame.shape
    if frame_shape[2] == 3:  # its an RGB frame
        average_frame_across_rgb = np.mean(frame, axis=2)
        average_frame_across_column = np.mean(average_frame_across_rgb, axis=1)
    else:
        average_frame_across_column = np.mean(frame, axis=1)
    average_frame_across_column = np.reshape(average_frame_across_column, (frame_shape[0],))
    return average_frame_across_column


# Input the video stream
video = cv2.VideoCapture(video_filepath)

# Validating the read of input video
if not video.isOpened():
    print("Error Opening the video stream or file")

# Video specifics extraction
total_number_of_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
height_of_frame = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))  # total number of rows
width_of_frame = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))  # total number of columns
frame_rate = float(video.get(cv2.CAP_PROP_FPS))
size_of_row_signal = int(np.multiply(total_number_of_frames, height_of_frame))
# print(size_of_row_signal)
#print(width_of_frame)

# row_signal = np.zeros((size_of_row_signal, 1), dtype=float)
row_signal = np.zeros((total_number_of_frames, height_of_frame, 1), dtype=float)
# Collect the row signal from the buffered frames

# Generating superpixel segment from first frame of the video
ret, frame = video.read()
frame = img_as_float(frame)
segments = slic(frame, n_segments=numSegments, sigma=5, start_label=1)  # Initializing superpixels segments using SLIC algorithm
motion_mask = cv2.createBackgroundSubtractorMOG2(history=50, varThreshold=150, detectShadows=True)
# ones superpixel is created for final AND operation with superpixel mask. The superpixels effected is set to zero,
# so same pixels are also set to zero in ones superpixel
master_ones_Superpixel_mask = np.ones([height_of_frame,width_of_frame],dtype=int)
if open_video_to_extract_Row_signal is True:
    frame_counter = 0
    while video.isOpened():
        ret, frame = video.read()
        if ret is True:
            frame_shape = frame.shape
            start_index = frame_counter * height_of_frame
            # row_signal[start_index:start_index + height_of_frame] = extract_row_pixel(frame)
            if do_ssm == 1:
                ones_Superpixel_mask = master_ones_Superpixel_mask.copy()
                frame_segments = segments.copy()  # creating a copy of SLIC segments for each frame
                new_frame_with_mask = motion_mask.apply(frame)  # applying the background subtractor to frame
                oSSM_mask,frame = SSM(frame, frame_segments, new_frame_with_mask, motion_detection_threshold, ones_Superpixel_mask)
                row_signal[frame_counter, :, 0] = second_half_extract_row_pixel(frame)
            else:
                row_signal[frame_counter, :, 0] = second_half_extract_row_pixel(frame)

            #oSSM_mask = oSSM_mask * 255
            #oSSM_mask = oSSM_mask.astype(np.uint8)
            cv2.imshow('Frame', frame)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break
        frame_counter += 1
        # print(frame_counter)
    video.release()
    cv2.destroyAllWindows()
    # store the variables for faster future use
    variable_location = video_folder + "row_signal.pkl"
    store_variable_file = open(variable_location, 'wb')
    pickle.dump(row_signal, store_variable_file)
    store_variable_file.close()
    print("Extracted Row Signal and stored in dump.\n")
else:
    variable_location = video_folder + "row_signal.pkl"
    load_variable_file = open(variable_location, 'rb')
    row_signal = pickle.load(load_variable_file)
    load_variable_file.close()
    print("Loaded the Row Signal. \n")

time = np.arange(0.0, size_of_row_signal)

# For a static video, clean the row signal with its video signal
# that should leave only the ENF signal
# Refer to { Exploiting Rolling Shutter For ENF Signal Extraction From Video }
# row_signal = video_signal + enf_signal
# average_of_each_row_element(row_signal) = average_of_each_row_element(video_signal) [since average of enf is 0]
# enf_signal = row_signal - average_of_each_row_element(row_signal)

# Estimate the ENF signal using the row signal collected
average_of_each_row_element = np.mean(row_signal, axis=0)
enf_video_signal = row_signal - average_of_each_row_element
flattened_enf_signal = enf_video_signal.flatten()  # the matrix shape ENF data is flattened to one dim data

fs = 500  # downsampling frequency
nfft = 8192
frame_size = 6  # change it to 6 for videos with large length recording
overlap = 0
filename = "mediator.wav"
# Writing the ENF data to the wav file for data type conversion
scipy.io.wavfile.write(filename, rate=int(frame_rate * height_of_frame), data=flattened_enf_signal)
filename_dup = "mediator_dup.wav"
signal0, fs = librosa.load(filename, sr=fs)  # loading the video ENF data
power_signal_filename = power_filepath
power_signal0, fs = librosa.load(power_signal_filename, sr=fs)  # loading the power ENF data


# ENF extraction from video recording
video_signal_object = pyenf.pyENF(signal0=signal0, fs=fs, nominal=120, harmonic_multiples=1, duration=1,
                                  strip_index=0, frame_size_secs=frame_size, nfft=nfft, overlap_amount_secs=overlap, width_signal=0.02, width_band=0.5)
spectro_strip, frequency_support = video_signal_object.compute_spectrogam_strips()
weights = video_signal_object.compute_combining_weights_from_harmonics()
OurStripCell, initial_frequency = video_signal_object.compute_combined_spectrum(spectro_strip, weights,
                                                                                frequency_support)
ENF = video_signal_object.compute_ENF_from_combined_strip(OurStripCell, initial_frequency)
#print(ENF)
# uncomment when comparing only 2 graphs.
"""
fig, (video, power) = plt.subplots(2, 1, sharex=True)
video.plot(ENF[:-12],'b')
video.set_title("ENF Signal without SSM", fontsize=12)
video.ticklabel_format(useOffset=False)
video.set_ylabel("Frequency (Hz)", fontsize=12)
#video.hlines(y=0.7, xmin=0, xmax=len(t), colors='r', linestyles='--', lw=2)
"""
fig, (video,videom, power) = plt.subplots(3, 1, sharex=True)
video.plot(ENF[:-12],'r')
video.set_title("ENF Signal without SSM", fontsize=12)
video.ticklabel_format(useOffset=False)
video.set_ylabel("Freq (Hz)", fontsize=12)

"""
# use this to compare 3 enf's from with ssm, without ssm, and power enf
variable_location = video_folder + "mask_enf.pkl"
load_variable_file = open(variable_location, 'rb')
ENF_mask = pickle.load(load_variable_file)
load_variable_file.close()

videom.plot(ENF_mask[:-12],'b')
videom.set_title("ENF Signal with SSM", fontsize=12)
videom.ticklabel_format(useOffset=False)
videom.set_ylabel("Freq (Hz)", fontsize=12)
"""

# ENF extraction from power recording
power_signal_object = pyenf.pyENF(signal0=power_signal0, fs=fs, nominal=60, harmonic_multiples=1, duration=0.1,
                                  strip_index=0, frame_size_secs=frame_size, nfft=nfft, overlap_amount_secs=overlap)
power_spectro_strip, power_frequency_support = power_signal_object.compute_spectrogam_strips()
power_weights = power_signal_object.compute_combining_weights_from_harmonics()
power_OurStripCell, power_initial_frequency = power_signal_object.compute_combined_spectrum(power_spectro_strip,
                                                                                            power_weights,
                                                                                            power_frequency_support)
power_ENF = power_signal_object.compute_ENF_from_combined_strip(power_OurStripCell, power_initial_frequency)

power.plot(power_ENF[:-8],'g')
power.set_title("Power ENF Signal", fontsize=12)
power.set_ylabel("Freq (Hz)", fontsize=12)
power.set_xlabel("Time", fontsize=12)
# plt.show()
power.ticklabel_format(useOffset=False)
print("Correlating the signal")
#enf_corr = signal.correlate(ENF, power_ENF, mode='same')
#corr.plot(enf_corr)
#corr.axhline(0.5, ls=':')
fig.tight_layout()
plt.show()

rho,total_windows = correlation_vector(ENF[:-7], power_ENF[:-7],window_size,shift_size)

"""
# temp load of rho
variable_location = video_folder + "rho_without_SSM.pkl"
load_variable_file = open(variable_location, 'rb')
rho_without_SSM = pickle.load(load_variable_file)
load_variable_file.close()

#


"""
t = np.arange(0,total_windows-1,1)
plt.plot(t,rho[0][1:],'g--', label="Plain Wall")
#plt.plot(t,rho_without_SSM[0][1:],'b', label="Without SSM")
plt.hlines(y=0.8, xmin=0, xmax=len(t), colors='r', linestyles='--', lw=2)
plt.ylabel('Correlation Coefficient', fontsize=12)
plt.xlabel('Number of Windows compared', fontsize=12)
plt.title('ENF fluctuations compared', fontsize=12)
#plt.set_legend('With SSM','Without SSM')
plt.legend(loc="lower right")
plt.show()
