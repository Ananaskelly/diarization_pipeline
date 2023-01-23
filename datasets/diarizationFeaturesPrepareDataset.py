import os
import pickle
import shutil
import math
import numpy as np
import torch
import torchaudio
import torchaudio.functional as F

from copy import copy
from tqdm import tqdm
from sklearn import preprocessing
from multiprocessing import Pool
from speechbrain.pretrained import SpeakerRecognition


# def get_speechbrain_emb(wav_path, temp_folder):
#     # waveform_x = Speaker_Recognition.load_audio(path=wav_path, savedir=temp_folder)
#     # batch_x = waveform_x.unsqueeze(0)
#     batch_x, rate = torchaudio.load(wav_path)
#     print(rate, batch_x.shape)
#     emb = Speaker_Recognition.encode_batch(batch_x, normalize=True)[0, :, :].cpu().numpy()
#
#     return emb


class DiarizationFeaturesPrepareDataset:

    def __init__(self, output_dir, wav_dir_lst=None, features_dir=None, feature_extractor_model=None, target_sr=16000,
                 min_spk=1, max_spk=10, min_dur=1.0, max_dur=7.0, min_utt=10, max_utt=15, win_step=0.75, win_size=1.5,
                 speaker_sep='_', device='cpu'):

        ##################
        #
        # initialize variables
        #
        ##################

        assert wav_dir_lst is not None or features_dir is not None, 'One of wav_dir_lst and features_dir must be not ' \
                                                                    'None!'
        wav_lst = list()
        files_lst = list()
        self._wav_based = False

        if wav_dir_lst is not None:
            if isinstance(wav_dir_lst, list):
                for d in wav_dir_lst:
                    curr_wav_p = list(map(lambda x: os.path.join(d, x), os.listdir(d)))
                    wav_lst.extend(curr_wav_p)
            elif isinstance(wav_dir_lst, str):
                wav_lst = list(map(lambda x: os.path.join(wav_dir_lst, x), os.listdir(wav_dir_lst)))
            else:
                raise NotImplemented
            files_lst = wav_lst
            self._wav_based = True
        else:
            files_lst = list(map(lambda x: os.path.join(features_dir, x), os.listdir(features_dir)))

        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.model = feature_extractor_model
        self.target_sr = target_sr
        self.min_spk = min_spk
        self.max_spk = max_spk
        self.min_dur = min_dur
        self.max_dur = max_dur
        self.min_utt = min_utt
        self.max_utt = max_utt
        self.win_size_frames = math.ceil(win_size * target_sr)
        self.win_step_frames = math.ceil(win_step * target_sr)

        self.device = device

        ##################
        #
        # parse wav_lst to unique speakers list and speakers to utterance dictionary
        #
        ##################

        self.unique_speakers = set()
        self.spk2utt = {}

        for file_p in tqdm(files_lst):
            wav_name = os.path.splitext(os.path.basename(file_p))[0]
            speaker_id = wav_name.split(speaker_sep)[0]
            self.unique_speakers.add(speaker_id)
            if speaker_id not in self.spk2utt:
                self.spk2utt[speaker_id] = []
            self.spk2utt[speaker_id].append(file_p)

        self.unique_speakers = list(self.unique_speakers)
        self.label_encoder = preprocessing.LabelEncoder().fit(self.unique_speakers)

    def make_single_wav(self):
        feats = []
        labels = []
        num_speakers = np.random.randint(low=self.min_spk, high=self.max_spk + 1)
        num_utt = np.random.randint(low=self.min_utt, high=self.max_utt + 1)
        target_speakers = np.random.choice(self.unique_speakers, size=num_speakers, replace=False)
        all_slice = torch.zeros((1, 0))
        for i in range(num_utt):
            dur = np.random.uniform(low=self.min_dur, high=self.max_dur)
            spk = np.random.choice(target_speakers)
            spk_utt = np.random.choice(self.spk2utt[spk])
            data, rate = torchaudio.load(spk_utt)
            if rate != self.target_sr:
                data = F.resample(data, rate, self.target_sr)
            actual_dur = data.shape[1] / rate

            if actual_dur < dur:
                orig_data = copy(data)
                orig_dur = actual_dur
                while actual_dur < dur:
                    data = torch.concatenate((data, orig_data), dim=1)
                    actual_dur += orig_dur

            frame_dur = math.ceil(dur * rate)
            frame_actual_dur = data.shape[1]
            bias = frame_actual_dur - frame_dur
            start_frame = np.random.randint(low=0, high=bias)

            full_data_slice = data[:, start_frame: start_frame + frame_dur]
            full_steps_num = full_data_slice.shape[1] // self.win_step_frames

            for j in range(full_steps_num):
                current_slice = full_data_slice[:, j * self.win_step_frames:
                                                j * self.win_step_frames + self.win_size_frames].to(self.device)
                current_embedding = self.model.encode_batch(current_slice,
                                                            normalize=True)[0, :, :].cpu().numpy().squeeze()

                feats.append(current_embedding)
                labels.append(spk)


            last_slice = full_data_slice[:, -self.win_size_frames:]
            last_embedding = self.model.encode_batch(last_slice,
                                                     normalize=True)[0, :, :].cpu().numpy().squeeze()
            all_slice = torch.concatenate((all_slice, full_data_slice.cpu()), dim=1)
            feats.append(last_embedding)
            labels.append(spk)

        labels = self.label_encoder.transform(labels)
        return np.stack(feats), np.array(labels)

    def make_single_feats(self):
        feats = []
        labels = []
        num_speakers = np.random.randint(low=self.min_spk, high=self.max_spk + 1)
        num_utt = np.random.randint(low=self.min_utt, high=self.max_utt + 1)
        target_speakers = np.random.choice(self.unique_speakers, size=num_speakers, replace=False)
        # all_slice = np.zeros((1, 0))

        for i in range(num_utt):
            dur = np.random.uniform(low=self.min_dur, high=self.max_dur)
            num_feats = math.ceil(dur / 0.75)

            spk = np.random.choice(target_speakers)
            spk_utt = np.random.choice(self.spk2utt[spk])
            data = np.load(spk_utt)
            actual_num_feats = data.shape[0]

            if actual_num_feats < num_feats:
                orig_data = copy(data)
                orig_num_feats = actual_num_feats
                while actual_num_feats < num_feats:
                    data = np.concatenate((data, orig_data), axis=0)
                    actual_num_feats += orig_num_feats

            bias = actual_num_feats - num_feats
            if bias > 0:
                start_idx = np.random.randint(low=0, high=bias)
            else:
                start_idx = 0

            feats.append(data[start_idx: start_idx + num_feats])
            labels.extend([spk] * num_feats)

        labels = self.label_encoder.transform(labels)

        feats = np.vstack(feats).astype(float)
        labels = np.asanyarray(labels).astype(str)
        return feats, labels

    def make_dataset(self, num_samples=100000):
        feats_lst = []
        labels_lst = []

        if self._wav_based:
            func = self.make_single_wav
        else:
            func = self.make_single_feats

        with Pool(10) as p:
            results = p.starmap(func, [() for _ in range(num_samples)])

        for p in results:
            feats_lst.append(p[0])
            labels_lst.append(p[1])
        # np.save(file=os.path.join(self.output_dir, 'features.npy'),
        #         arr=feats_lst)
        # np.save(file=os.path.join(self.output_dir, 'labels.npy'),
        #         arr=labels_lst)
        pickle.dump(feats_lst, open(os.path.join(self.output_dir, 'features.npy'), 'wb'))
        pickle.dump(labels_lst, open(os.path.join(self.output_dir, 'labels.npy'), 'wb'))

    @staticmethod
    def extract_all_vectors(wav_dir, save_dir, model, target_sr, win_size_dur, win_step_dur, device):
        win_size_frames = math.ceil(win_size_dur * target_sr)
        win_step_frames = math.ceil(win_step_dur * target_sr)
        for wav in tqdm(os.listdir(wav_dir)):
            if os.path.exists(os.path.join(save_dir, wav.replace('wav', 'npy'))):
                continue
            data, rate = torchaudio.load(os.path.join(wav_dir, wav))
            if rate != target_sr:
                data = F.resample(data, rate, target_sr)

            num_steps = data.shape[1] // target_sr

            batch = []

            for j in range(num_steps):
                current_slice = data[:, j * win_step_frames:
                                        j * win_step_frames + win_size_frames].to(device)
                batch.append(current_slice)

            last_slice = data[:, -win_size_frames:].to(device)
            batch.append(last_slice)
            batch = torch.stack(batch).squeeze()
            try:
                embeddings = model.encode_batch(batch,
                                                normalize=True).cpu().numpy().squeeze()
            except:
                continue
            np.save(file=os.path.join(save_dir, wav.replace('wav', 'npy')),
                    arr=embeddings)


if __name__ == '__main__':
    speaker_recognition_model = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                                                savedir="../pretrained_models/spkrec-ecapa-voxceleb",
                                                                run_opts={"device": "mps"})
    dataset = DiarizationFeaturesPrepareDataset(features_dir='/Users/ananaskelly/work_dir/datasets/'
                                                             'voxceleb_concatenated_listed_ecapa_vectors',
                                                output_dir='/Users/ananaskelly/work_dir/ecapa_features_uisrnn_train_v1')
    dataset.make_dataset(num_samples=100000)

    # DiarizationFeaturesPrepareDataset.extract_all_vectors(wav_dir='/Users/ananaskelly/work_dir/datasets/'
    #                                                               'voxceleb_concatenated_listed',
    #                                                       save_dir='/Users/ananaskelly/work_dir/datasets/'
    #                                                                'voxceleb_concatenated_listed_ecapa_vectors',
    #                                                       model=speaker_recognition_model,
    #                                                       target_sr=16000,
    #                                                       win_size_dur=1.5,
    #                                                       win_step_dur=0.75,
    #                                                       device='mps')

    # data = np.load('/Users/ananaskelly/work_dir/test_features_prepare/features.npy', allow_pickle=True)
    # print(data[0].shape)
    # data = np.load('/Users/ananaskelly/work_dir/test_features_prepare/labels.npy', allow_pickle=True)
    # print(data[0].shape)
