import torch
from transformers import ReformerTokenizer, ReformerModel, ReformerForMaskedLM, AdamW
import logging

class Reformer:
    
    def train(path_pretrained):

        if os.path.exists(output_dir) and os.listdir(output_dir):
            raise ValueError("Output directory (%s) already exists and is not empty." % output_dir)
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO,
                        filename=os.path.join(output_dir, "log.txt"))

        # GARBAGE TRUCK
        configuration = ReformerConfig()
        model = ReformerModel(configuration)
        model.train()
        optimizer = AdamW(model.parameters(), lr=1e-5)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5)

        tokenizer = ReformerTokenizer.from_pretrained(path_pretrained)
        