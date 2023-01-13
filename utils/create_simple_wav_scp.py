import os

if __name__ == '__main__':
    wav_scp_path = '../meta_files/voxceleb_meta/wav.scp'
    utt2spk_path = '../meta_files/voxceleb_meta/utt2spk'
    in_dir = '/Users/ananaskelly/work_dir/datasets/voxceleb_concatenated'

    utt2spk_dict = {}
    scp_lines = []

    for spk_d in os.listdir(in_dir):
        spk_d_path = os.path.join(in_dir, spk_d)
        if not os.path.isdir(spk_d_path):
            continue
        for f in os.listdir(spk_d_path):
            f_name = os.path.splitext(f)[0]
            f_path = os.path.join(spk_d_path, f)
            utt2spk_dict[f_name] = spk_d
            scp_lines.append('{} sox {} -r 16000 -t wav - |\n'.format(f_name, f_path))

    with open(wav_scp_path, 'w') as out_file:
        for line in scp_lines:
            out_file.write(line)

    with open(utt2spk_path, 'w') as out_file:
        for key, val in utt2spk_dict.items():
            out_file.write('{} {}\n'.format(key, val))
