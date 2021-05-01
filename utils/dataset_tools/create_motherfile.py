from __future__ import absolute_import, division, print_function
import json
import argparse
import codecs
import os
import hashlib
import numpy as np
import time
from attr import dataclass

class Annotation:

    def __init__(self, id, type_id, ann_type, name):
        self.id = id
        self.type_id = type_id
        self.ann_type = ann_type
        self.ann_name = name

    def get_type(self):
        return self.ann_type


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
    out = {}
    files = os.listdir(path)
    files = list(filter(lambda x: x.split('.')[1] == 'json', files))
    for f in files:
        tokens, labels = read_json('{}/{}'.format(path, f))
        out[f] = {"tokens": tokens, "labels": labels}
    return out


def main(args):
    # read files
    train_examples, test_examples = {}, {}
    read_method = load_from_folder
    mother_file = {}
    if args.input_dir:
        loaded_ds = read_method(args.input_dir)
        if args.split:
            divs = args.split.split('|')
            for i, file_name in enumerate(loaded_ds):
                if i < int(float(divs[0])*len(loaded_ds))+1:
                    train_examples[file_name] = loaded_ds[file_name]
                else:
                    test_examples[file_name] = loaded_ds[file_name]
        mother_file = {
            "train": train_examples,
            "test": test_examples
        }
        print(len(train_examples))
        print(len(test_examples))
        codecs.open(args.output, "w", "utf8").write(json.dumps(mother_file, indent=4))
        


def parse_args():
    parser = argparse.ArgumentParser(
        description='Set of tools')
    parser.add_argument('--input_dir', required=False, metavar='PATH', help='path to train input directory')
    parser.add_argument('--output', required=True, metavar='PATH', help='path to json')
    parser.add_argument('--split', required=False, type=str,
                        help='In absence of preexisting split in train/test/valid, train set can be split given ratio in form 0.6|0.2|0.2 for train|test|valid')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    try:
        main(args)
    except ValueError as er:
        print("[ERROR] %s" % er)