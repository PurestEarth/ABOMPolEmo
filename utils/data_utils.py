import json
from annotation import Annotation

def read_json(path):
    json_out = {}
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
                print(ann_dict[ann[0]].get_type())
                print(text)
            else:
                labels.append('O')
            tokens.append(text)
        assert len(tokens) == len(labels)
        print(tokens)
        print(labels)
    return json_out

read_json('../dataset/inforex_export_349/documents/00140104.json')

# annotacje które mamy to subjecty
# sprawdzic czy w ctagu nie ma info o sentymecie