import sys
import pickle
import numpy as np

sys.path.append('../')

from uis_rnn_module import uisrnn


if __name__ == '__main__':
    model_args, training_args, inference_args = uisrnn.parse_arguments()
    model = uisrnn.UISRNN(model_args)
    print(model.device)
    model.verbose = 3

    features_path = '/Users/ananaskelly/work_dir/ecapa_features_uisrnn_train_v1/features.npy'
    labels_path = '/Users/ananaskelly/work_dir/ecapa_features_uisrnn_train_v1/labels.npy'

    features = pickle.load(open(features_path, 'rb'))
    labels = pickle.load(open(labels_path, 'rb'))

    training_args.enforce_cluster_id_uniqueness = False
    model.fit(train_sequences=features[:10000],
              train_cluster_ids=labels[:10000],
              args=training_args)
