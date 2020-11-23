import json
import os
from utils.annotation import Annotation
from keras.utils import Sequence
import torch
import math
from torch.utils.data import TensorDataset


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


def get_examples(path, set_type='train'):
    examples = []
    x_train, y_train = load_from_folder(path)
    for i in range(0, len(x_train)):
        guid = "%s-%s" % (set_type, i)
        text_a = ' '.join(x_train[i])
        text_b = None
        label = ' '.join(y_train[i])
        examples.append(InputExample(
            guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples


def convert_examples_to_features(examples, label_list, max_seq_length, encode_method):
    ignored_label = "IGNORE"
    label_map = {label: i for i, label in enumerate(label_list, 1)}
    label_map[ignored_label] = 0  # 0 label is to be ignored
    pending_token_ids = []
    pending_input_mask = []
    pending_label_ids = []
    pending_valid = []
    pending_label_mask = []
    features = []
    for (ex_index, example) in enumerate(examples):
        textlist = example.text_a.split(' ')
        labellist = example.label
        labels = []
        valid = []
        label_mask = []
        token_ids = []
       
        for i, word in enumerate(textlist):  
            tokens = encode_method(word.strip())  # word token ids   
            token_ids.extend(tokens)  # all sentence token ids
            label_1 = labellist[i]
            for m in range(len(tokens)):
                if m == 0:  # only label the first BPE token of each work
                    labels.append(label_1)
                    valid.append(1)
                    label_mask.append(1)
                else:
                    labels.append(ignored_label)  # unlabeled BPE token
                    label_mask.append(0)
                    valid.append(0)

        if len(token_ids) >= max_seq_length - 1:  # trim extra tokens
            token_ids = token_ids[0:(max_seq_length-2)]
            labels = labels[0:(max_seq_length-2)]
            valid = valid[0:(max_seq_length-2)]
            label_mask = label_mask[0:(max_seq_length-2)]

        # adding <s>
        token_ids.insert(0, 0)
        labels.insert(0, ignored_label)
        label_mask.insert(0, 0)
        valid.insert(0, 0)

        # adding </s>
        token_ids.append(2)
        labels.append(ignored_label)
        label_mask.append(0)
        valid.append(0)

        assert len(token_ids) == len(labels)
        assert len(valid) == len(labels)

        label_ids = []
        for i, _ in enumerate(token_ids):
            label_ids.append(label_map[labels[i]])

        assert len(token_ids) == len(label_ids)
        assert len(valid) == len(label_ids)
        input_mask = [1] * len(token_ids)
        if len(token_ids) + len(pending_token_ids) > max_seq_length:
            features.append(append_pending(ignored_label, pending_token_ids, pending_input_mask, pending_label_ids,
                                           pending_valid, pending_label_mask, max_seq_length, label_map, ex_index,
                                           example))
            pending_token_ids = token_ids
            pending_input_mask = input_mask
            pending_label_ids = label_ids
            pending_valid = valid
            pending_label_mask = label_mask
        else:
            pending_token_ids.extend(token_ids)
            pending_input_mask.extend(input_mask)
            pending_label_ids.extend(label_ids)
            pending_valid.extend(valid)
            pending_label_mask.extend(label_mask)
    features.append(append_pending(ignored_label, pending_token_ids, pending_input_mask, pending_label_ids,
                                   pending_valid, pending_label_mask, max_seq_length, label_map))
    return features


def create_dataset(features):
    
    all_input_ids = torch.tensor(
        [f.input_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor(
        [f.label_id for f in features], dtype=torch.long)
    all_valid_ids = torch.tensor(
        [f.valid_ids for f in features], dtype=torch.long)
    all_lmask_ids = torch.tensor(
        [f.label_mask for f in features], dtype=torch.long)

    return TensorDataset(
        all_input_ids, all_label_ids, all_lmask_ids, all_valid_ids)


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


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, label_id, valid_ids=None, label_mask=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.label_id = label_id
        self.valid_ids = valid_ids
        self.label_mask = label_mask

