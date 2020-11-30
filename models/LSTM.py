import torch.nn as nn
import torch.nn.functional as F
from allennlp.commands.elmo import ElmoEmbedder
from utils.data_utils import get_batch
import torch
from pytorch_transformers import AdamW, WarmupLinearSchedule
import logging
import os
import sys


class LSTM(nn.Module):

    def __init__(self, n_labels, hidden_size, embedding_path, dropout=0.2, label_ignore_idx=0,
                batch_size=32, head_init_range=0.04, device='cuda',
                vocab_size=320, input_size=300, num_layers=2, embed_size=1024):
        super().__init__()
        self.input_size = input_size
        self.batch_size = batch_size
        self.n_labels = n_labels
        self.label_ignore_idx = label_ignore_idx
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embed_size = embed_size
        self.init_linear = nn.Linear(self.embed_size, self.embed_size)
        self.lstm = nn.LSTM(self.embed_size, self.hidden_size, self.num_layers, batch_first=True, bidirectional=True)
        self.classification_head = nn.Linear(self.num_layers*self.hidden_size, self.n_labels)
        self.classification_head.weight.data.normal_(mean=0.0, std=head_init_range)

        options_file = embedding_path + "/options.json"
        weight_file = embedding_path + "/weights.hdf5"
        self.elmo = ElmoEmbedder(options_file, weight_file, 1)


    def init_hidden(self):
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_size),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_size))


    def forward(self, inputs_ids, labels, labels_mask, valid_mask):
        print(inputs_ids.size())
        linear_input = self.init_linear(inputs_ids)
        lstm_out, self.hidden = self.lstm(linear_input)
        logits = self.classification_head(lstm_out)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.label_ignore_idx)
            if labels_mask is not None:
                active_loss = valid_mask.view(-1) == 1

                active_logits = logits.view(-1, self.n_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(
                    logits.view(-1, self.n_labels), labels.view(-1))
            return loss
        else:
            return logits


    def encode_word(self, sentence):
        """
        takes a string and returns a list of token ids
        """
        tensor_ids = self.elmo.embed_sentence(sentence)[2]
        # remove <s> and </s> ids
        # TODO Test
        return tensor_ids


class Trainer:


    def train(self, model, x_train, y_train, label_map, epochs, train_batch_size, seed, gradient_accumulation_steps, output_dir, max_seq_length=128,
              weight_decay=0.01, warmup_proportion=0.1, learning_rate=5e-5, adam_epsilon=1e-8, no_cuda=False):
        # TODO batches
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        logger = logging.getLogger(__name__)

        num_train_optimization_steps = int(
            len(x_train) / train_batch_size / gradient_accumulation_steps) * epochs
        params = list(model.named_parameters())
        no_decay = ['bias', 'final_layer_norm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in params if not any(
                nd in n for nd in no_decay)], 'weight_decay': weight_decay},
            {'params': [p for n, p in params if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        warmup_steps = int(warmup_proportion * num_train_optimization_steps)
        optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps, t_total=num_train_optimization_steps)

        device = 'cuda:3' if (torch.cuda.is_available() and not no_cuda) else 'cpu'
        logger.info(device)
        model.to(device)
        print('-')
        print(device)
        print(label_map)
        steps = len(x_train)%train_batch_size
        for epoch_no in range(1, epochs+1):
            logger.info("Epoch %d" % epoch_no)
            tr_loss = 0
            model.train()
            for step in range(0, steps):
                div = int(step*train_batch_size)
                if len(x_train) > div+train_batch_size:
                    input_ids, label_ids, l_mask, valid_ids, = get_batch(x_train[div:div+train_batch_size], y_train[div:div+train_batch_size],
                    device=device, embed_method=model.encode_word, max_seq_length=128, label_map=label_map)
                else:
                    input_ids, label_ids, l_mask, valid_ids, = get_batch(x_train[div:], y_train[div:], device=device,
                    embed_method=model.encode_word, max_seq_length=128, label_map=label_map, last_one=True)
                loss = model(input_ids, label_ids, l_mask, valid_ids)
                print(loss)
                if gradient_accumulation_steps > 1:
                    loss = loss / gradient_accumulation_steps
                loss.backward()
                print(" I AM SUPREME ")
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_grad_norm)
                tr_loss += loss.item()
                if step % 10 == 0:
                    logger.info('Step = %d/%d; Loss = %.4f' % (step+1, steps, tr_loss / (step+1)))
                if (step + 1) % gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()