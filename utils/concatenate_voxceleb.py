import os
import shutil
import soundfile as sf

from tqdm import tqdm


def concatenate(in_path, out_path):
    for spk_d in tqdm(os.listdir(in_path)):
        spk_d_path = os.path.join(in_path, spk_d)
        for sess_d in os.listdir(spk_d_path):
            sess_d_path = os.path.join(spk_d_path, sess_d)
            session_wav_path = os.path.join(out_path, spk_d, sess_d + '.wav')
            os.makedirs(os.path.join(out_path, spk_d), exist_ok=True)
            wav_data = []

            for f in os.listdir(sess_d_path):
                wav_path = os.path.join(sess_d_path, f)
                data, rate = sf.read(wav_path)
                wav_data.extend(data.tolist())

            sf.write(session_wav_path, wav_data, rate)


def reorganize(in_path, out_path):

    for d in os.listdir(in_path):
        if d.startswith('.'):
            continue
        for f in os.listdir(os.path.join(in_path, d)):
            if f.startswith('.'):
                continue
            in_f_name = os.path.join(in_path, d, f)
            out_f_name = os.path.join(out_path, '{}_{}'.format(d, f))
            print(out_f_name)
            shutil.move(in_f_name, out_f_name)


if __name__ == '__main__':
    _in_path = '/Users/ananaskelly/work_dir/datasets/voxceleb/wav'
    _out_path = '/Users/ananaskelly/work_dir/datasets/voxceleb_concatenated'
    _out_path2 = '/Users/ananaskelly/work_dir/datasets/voxceleb_concatenated_listed'
    os.makedirs(_out_path2, exist_ok=True)
    reorganize(_out_path, _out_path2)


