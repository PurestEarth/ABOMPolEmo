import json
import os
from utils.annotation import Annotation
from keras.utils import Sequence
import math


def read_json(path):
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
        text_tuples = []
        ann_dict = {}
        chunks = data['chunks']
        annotations = data['annotations']
        for chunk in chunks[0]:
            for sub_chunk in chunk:
                text_tuples.append((sub_chunk['orth'], sub_chunk['annotations']))
        for annotation in annotations:
            ann_dict[annotation['id']] = Annotation(annotation['id'], annotation['type_id'], annotation['type'], annotation['name'])
        labels = []
        tokens = []
        for (text, ann) in text_tuples:
            if len(ann) > 0:
                labels.append(ann_dict[ann[0]].get_type())
            else:
                labels.append('O')
            tokens.append(text)
        assert len(tokens) == len(labels)
        return tokens, labels


def load_from_folder(path):
    x_train, y_train = [], []
    files = os.listdir(path)
    files = list(filter(lambda x: x.split('.')[1] == 'json', files))
    for f in files:
        tokens, labels = read_json('{}/{}'.format(path, f))
        x_train.append(tokens)
        y_train.append(labels)
        # TODO delet
        if len(x_train) > 100:
            break
    return x_train, y_train


class NERSequence(Sequence):

    def __init__(self, x, y, batch_size=1, preprocess=None):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.preprocess = preprocess

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size: (idx + 1) * self.batch_size]

        return self.preprocess(batch_x, batch_y)

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)