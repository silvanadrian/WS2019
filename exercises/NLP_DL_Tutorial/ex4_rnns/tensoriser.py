from ex4_rnns.map import numpify, tokenize, lower, deep_map, deep_seq_map, map_to_targets

from ex4_rnns.vocab import Vocab
import numpy as np

def prepare_data(data, vocab=None):
    data_tokenized = deep_map(data, tokenize, ['tweets', 'targets'])
    data_lower = deep_seq_map(data_tokenized, lower, ['tweets', 'targets'])
    data = deep_seq_map(data_lower, lambda xs: ["<SOS>"] + xs + ["<EOS>"], ["tweets", "targets"])

    if vocab is None:
        vocab = Vocab()
        for instance in data["tweets"] + data["targets"]:
            for token in instance:
                vocab(token)

    #data = map_to_targets(data, "stance", "labels")  # map stance IDs to one-hot vectors, save in data["targets"]
    data["labels"] = transform_labels(data["labels"])

    data_ids = deep_map(data, vocab, ["tweets", "targets"])
    data_ids = deep_seq_map(data_ids, lambda xs: len(xs), keys=['tweets', 'targets'], fun_name='lengths', expand=True)

    return data_ids, vocab



def transform_labels(labels):
    labels_t = []
    for lab in labels:
        v = [0.0, 0.0, 0.0] #np.zeros(3, np.float32)
        if lab == 'NONE':
            ix = 0
        elif lab == 'AGAINST':
            ix = 1
        elif lab == 'FAVOR':
            ix = 2
        v[ix] = 1.0
        labels_t.append(v)
    return labels_t