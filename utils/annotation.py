from attr import dataclass

class Annotation:

    def __init__(self, id, type_id, ann_type, name):
        self.id = id
        self.type_id = type_id
        self.ann_type = ann_type
        self.ann_name = name

    def get_type(self):
        return self.ann_type