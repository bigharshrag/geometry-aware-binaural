import os.path
import time
import librosa
import h5py
import random
import math
import numpy as np
import glob
import torch
import pickle
from PIL import Image, ImageEnhance
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset

def normalize(samples, desired_rms = 0.1, eps = 1e-4):
  rms = np.maximum(eps, np.sqrt(np.mean(samples**2)))
  samples = samples * (desired_rms / rms)
  return samples

def safe_log10(x, eps=1e-10):
    result = np.where(x > eps, x, -10)
    np.log10(result, out=result, where=result > 0)
    return result

def generate_spectrogram(audio):
    spectro = librosa.core.stft(audio, n_fft=512, hop_length=160, win_length=400, center=True)
    real = np.expand_dims(np.real(spectro), axis=0)
    imag = np.expand_dims(np.imag(spectro), axis=0)
    spectro_two_channel = np.concatenate((real, imag), axis=0)
    return spectro_two_channel

def process_image(image, augment, resize = True):
    if resize:
        image = image.resize((480,240))
    w,h = image.size
    w_offset = w - 448
    h_offset = h - 224
    left = random.randrange(0, w_offset + 1)
    upper = random.randrange(0, h_offset + 1)
    image = image.crop((left, upper, left+448, upper+224))

    if augment:
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(random.random()*0.6 + 0.7)
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(random.random()*0.6 + 0.7)
    return image

class FairPlayDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.audios = []

        h5f_path = os.path.join(opt.hdf5FolderPath, opt.mode+".h5")
        h5f = h5py.File(h5f_path, 'r')
        b_audios = h5f['audio'][:]
        for n in b_audios:
            a = n.decode('UTF-8')
            self.audios.append(os.path.join(opt.base_path, a.split('/')[-1]))

        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        vision_transform_list = [transforms.ToTensor(), normalize]
        self.vision_transform = transforms.Compose(vision_transform_list)

    def __getitem__(self, index):
        # load audio
        audio_full, audio_rate = librosa.load(self.audios[index], sr=self.opt.audio_sampling_rate, mono=False)

        # randomly get a start time for the audio segment from the 10s clip
        audio_start_time = random.uniform(0, 9.9 - self.opt.audio_length)
        audio_end_time = audio_start_time + self.opt.audio_length
        audio_start = int(audio_start_time * self.opt.audio_sampling_rate)
        audio_end = audio_start + int(self.opt.audio_length * self.opt.audio_sampling_rate)
        audio = audio_full[:, audio_start:audio_end]
        audio = normalize(audio)
        audio_channel1 = audio[0,:]
        audio_channel2 = audio[1,:]

        # get the frame dir path based on audio path
        path_parts = self.audios[index].strip().split('/')
        path_parts[-1] = path_parts[-1][:-4] + '.mp4'
        path_parts[-2] = 'frames'
        frame_path = '/'.join(path_parts)

        # get the closest frame to the audio segment
        frame_index = int(round(((audio_start_time + audio_end_time) / 2.0 + 0.05) * 10))  #10 frames extracted per second
        frame = process_image(Image.open(os.path.join(frame_path, str(frame_index).zfill(6) + '.png')).convert('RGB'), self.opt.enable_data_augmentation)
        frame = self.vision_transform(frame)

        # passing the spectrogram of the difference
        audio_diff_spec = torch.FloatTensor(generate_spectrogram(audio_channel1 - audio_channel2))
        audio_mix_spec = torch.FloatTensor(generate_spectrogram(audio_channel1 + audio_channel2))

        left_spec = torch.FloatTensor(generate_spectrogram(audio_channel1)[:, :256, :])
        right_spec = torch.FloatTensor(generate_spectrogram(audio_channel2)[:, :256, :])
        if np.random.random() < 0.5:
            coherence_spec = torch.cat((left_spec, right_spec), dim=0)
            label = torch.FloatTensor([0])
        else:
            coherence_spec = torch.cat((right_spec, left_spec), dim=0)
            label = torch.FloatTensor([1])

        consistency_frame_idx = np.random.choice(np.arange(max(1, frame_index-10), min(99, frame_index+11)))
        while consistency_frame_idx == frame_index:
            consistency_frame_idx = np.random.choice(np.arange(max(1, frame_index-10), min(99, frame_index+11)))
        consistency_frame = process_image(Image.open(os.path.join(frame_path, str(consistency_frame_idx).zfill(6) + '.png')).convert('RGB'), self.opt.enable_data_augmentation)
        consistency_frame = self.vision_transform(consistency_frame)

        # rir_spec returned is dummy to prevent errors with joint training
        return {'frame': frame, 'audio_diff_spec': audio_diff_spec, 'audio_mix_spec': audio_mix_spec, \
                'consistency_frame': consistency_frame, 'left_spec': left_spec, 'right_spec': right_spec, \
                'rir_spec': left_spec, \
                'cl_spec': coherence_spec, 'label': label}

    def __len__(self):
        return len(self.audios)

    def name(self):
        return 'FairPlayDataset'


class AudioVisualDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.audios = []
        self.audio_cache = {}
        self.frames_cache = {}

        # load hdf5 file here
        with open(os.path.join(opt.hdf5FolderPath, f"{opt.split}_{opt.mode}.txt")) as f:
            for line in f:
                self.audios.append(line.strip())
        
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        vision_transform_list = [transforms.ToTensor(), normalize]
        self.vision_transform = transforms.Compose(vision_transform_list)

    def __getitem__(self, index):
        #load audio
        audio_full, audio_rate = librosa.load(self.audios[index], sr=self.opt.audio_sampling_rate, mono=False)
        vid_name = self.audios[index].strip().split('/')[-1].split('.')[0]

        # randomly get a start time for the audio segment from the clip
        ch_segment = np.random.choice(int(audio_full.shape[1]/(audio_rate * 5))) # Choose a 5 second interval to work with
        audio_start_time = random.uniform(0, 4.3 - self.opt.audio_length) + (ch_segment*5.0)
        audio_end_time = audio_start_time + self.opt.audio_length
        audio_start = int(audio_start_time * self.opt.audio_sampling_rate)
        audio_end = audio_start + int(self.opt.audio_length * self.opt.audio_sampling_rate)
        audio = audio_full[:, audio_start:audio_end]
        audio = normalize(audio)
        audio_channel1 = audio[0,:]
        audio_channel2 = audio[1,:]

        #get the frame dir path based on audio path
        path_parts = self.audios[index].strip().split('/')
        path_parts[-1] = path_parts[-1][:-4]
        path_parts[-2] = 'frames'
        frame_path = '/'.join(path_parts)
        frame_path = os.path.join(frame_path, vid_name)

        # get the closest frame to the audio segment
        frame_index = int(round(((audio_start_time + audio_end_time) / 2.0) * 5))  #5 frames extracted per second
        frame_key = '{}_{}.jpg'.format(frame_path, str(frame_index).zfill(4))
        frame = process_image(Image.open(frame_key).convert('RGB'), self.opt.enable_data_augmentation)
        frame = self.vision_transform(frame)
        
        consistency_frame_idx = int(round((random.uniform(0, 4.9) + (ch_segment*5.0)) * 5))
        while consistency_frame_idx == frame_index:
            consistency_frame_idx = int(round((random.uniform(0, 4.9) + (ch_segment*5.0)) * 5))
        consistency_frame_key = '{}_{}.jpg'.format(frame_path, str(consistency_frame_idx).zfill(4))
        consistency_frame = process_image(Image.open(consistency_frame_key).convert('RGB'), self.opt.enable_data_augmentation)
        consistency_frame = self.vision_transform(consistency_frame)

        # passing the spectrogram of the difference
        audio_diff_spec = torch.FloatTensor(generate_spectrogram(audio_channel1 - audio_channel2))
        audio_mix_spec = torch.FloatTensor(generate_spectrogram(audio_channel1 + audio_channel2))

        left_spec = torch.FloatTensor(generate_spectrogram(audio_channel1)[:, :256, :])
        right_spec = torch.FloatTensor(generate_spectrogram(audio_channel2)[:, :256, :])
        if np.random.random() < 0.5:
            cl_spec = torch.cat((left_spec, right_spec), dim=0)
            label = torch.FloatTensor([0])
        else:
            cl_spec = torch.cat((right_spec, left_spec), dim=0)
            label = torch.FloatTensor([1])

        # Load RIR
        binaural_rir_file = os.path.join(self.opt.rir_base_path, f'{vid_name}_{ch_segment}.wav')
        rir, sr = librosa.load(binaural_rir_file, sr=16000, mono=False)
        rir = np.pad(rir, ((0,0), (0, max(0, 3568 - rir.shape[1]))), 'constant', constant_values=0)
        spec1 = np.abs(librosa.stft(rir[0, :], n_fft=512, hop_length=16, win_length=64, center=True))[:224, :224]
        spec2 = np.abs(librosa.stft(rir[1, :], n_fft=512, hop_length=16, win_length=64, center=True))[:224, :224]
        spectro = torch.FloatTensor(np.stack((spec1, spec2)))

        return {'frame': frame, 'audio_diff_spec': audio_diff_spec, 'audio_mix_spec': audio_mix_spec, \
                'consistency_frame': consistency_frame, 'left_spec': left_spec, 'right_spec': right_spec,\
                'rir_spec': spectro, \
                'cl_spec': cl_spec, 'label': label}

    def __len__(self):
        return len(self.audios)

    def name(self):
        return 'AudioVisualDataset'


class YoutubeBinauralDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.video_ids = []
        self.run_mode = opt.mode

        #load hdf5 file here
        with open(os.path.join(opt.hdf5FolderPath, f"YoutubeBinaural_{opt.mode}.txt")) as f:
            for line in f:
                self.video_ids.append(line.strip())

        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        vision_transform_list = [transforms.ToTensor(), normalize]
        self.vision_transform = transforms.Compose(vision_transform_list)

    def __getitem__(self, index):
        vid_name = self.video_ids[index]

        # load audio
        audio_path = os.path.join(self.opt.base_path, vid_name, f"{vid_name}.wav")
        audio_full, audio_rate = librosa.load(audio_path, sr=self.opt.audio_sampling_rate, mono=False)

        # randomly get a start time for the audio segment from the clip
        audio_start_time = random.uniform(0, 0.85 - self.opt.audio_length)
        audio_start = int(audio_start_time * self.opt.audio_sampling_rate)
        audio_end = audio_start + int(self.opt.audio_length * self.opt.audio_sampling_rate)
        audio = audio_full[:, audio_start:audio_end]
        audio = normalize(audio)
        audio_channel1 = audio[0,:]
        audio_channel2 = audio[1,:]

        frame_dir_path = os.path.join(self.opt.ytb_frames_path, vid_name)
        # get the closest frame to the audio segment
        frame_path = os.path.join(frame_dir_path, f"{vid_name}-04.png")
        frame = process_image(Image.open(frame_path).convert('RGB'), self.opt.enable_data_augmentation, False)
        frame = self.vision_transform(frame)

        # passing the spectrogram of the difference
        audio_diff_spec = torch.FloatTensor(generate_spectrogram(audio_channel1 - audio_channel2))
        audio_mix_spec = torch.FloatTensor(generate_spectrogram(audio_channel1 + audio_channel2))

        left_spec = torch.FloatTensor(generate_spectrogram(audio_channel1)[:, :256, :])
        right_spec = torch.FloatTensor(generate_spectrogram(audio_channel2)[:, :256, :])
        if np.random.random() < 0.5:
            cl_spec = torch.cat((left_spec, right_spec), dim=0)
            label = torch.FloatTensor([0])
        else:
            cl_spec = torch.cat((right_spec, left_spec), dim=0)
            label = torch.FloatTensor([1])

        consistency_frame_idx = np.random.choice(['01', '02', '03', '05', '06', '07', '08'])
        consistency_frame_path = os.path.join(frame_dir_path, f"{vid_name}-{consistency_frame_idx}.png")
        consistency_frame = process_image(Image.open(consistency_frame_path).convert('RGB'), self.opt.enable_data_augmentation, False)
        consistency_frame = self.vision_transform(consistency_frame)


        # rir_spec returned is dummy to prevent errors with joint training
        return {'frame': frame, 'audio_diff_spec': audio_diff_spec, 'audio_mix_spec': audio_mix_spec, \
                'consistency_frame': consistency_frame,\
                'left_spec': left_spec, 'right_spec': right_spec,\
                'rir_spec': left_spec, \
                'cl_spec': cl_spec, 'label': label}

    def __len__(self):
        return len(self.video_ids)

    def name(self):
        return 'YoutubeBinauralDataset'

