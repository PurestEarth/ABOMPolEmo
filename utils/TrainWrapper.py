import os
from .data_utils import get_examples, convert_examples_to_features, create_dataset, save_params, get_examples_from_json, read_label_file, get_examples_from_xml, get_examples_from_motherfile
from .train_utils import evaluate_model
from models.xlmr_for_token_classification import XLMRForTokenClassification
from models.herbert_for_token_calssification import AutoTokenizerForTokenClassification
from models.bert_base_multilingual_cased import BertBaseMultilingualCased
from models.polish_roberta import PolishRoberta
from models.reformer import Reformer
from pytorch_transformers import AdamW, WarmupLinearSchedule
from torch.utils.data import DataLoader, RandomSampler
import random
import numpy as np
import torch
import logging
import sys


class TrainWrapper:

    def train(self, output_dir, train_batch_size, gradient_accumulation_steps, seed,
              epochs, data_path, pretrained_path, valid_path=None, no_cuda=False, dropout=0.3,
              weight_decay=0.01, warmup_proportion=0.1, learning_rate=5e-5, adam_epsilon=1e-8,
              max_seq_length=128, squeeze=True, max_grad_norm=1.0, eval_batch_size=32, epoch_save_model=False,
              model_name='XLMR', embedding_path=None, split_train_data=False, data_divider=0.6, wandb=None, save=True,
              logger = None, json_dataset=False, label_file=None, xlm_dataset=False, div=None, div_2 = None,
              motherfile=False, multi_source_labels=False, device = 0):
        if wandb:
            import wandb
            print(wandb)
            wandb.init(project='xlmr-multitask',
                    config={
                        "epochs": epochs,
                        "language_model": pretrained_path,
                        "batch_size": train_batch_size,
                        "max_seq_length": max_seq_length,
                        "warmup_proportion": warmup_proportion,
                        "learning_rate": learning_rate,
                        "gradient_accumulation_steps": gradient_accumulation_steps,
                        "squeeze": squeeze,
                        "dropout": dropout,
                        "output_dit": output_dir
                    })
        if save and os.path.exists(output_dir) and os.listdir(output_dir):
            raise ValueError("Output directory (%s) already exists and is not empty." % output_dir)
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if not logger:
            logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S',
                            level=logging.INFO,
                            filename=os.path.join(output_dir, "log.txt"))
            logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

            logger = logging.getLogger(__name__)

        if gradient_accumulation_steps < 1:
            raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1"
                         % gradient_accumulation_steps)

        train_batch_size = train_batch_size // gradient_accumulation_steps
    
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if split_train_data:
            if json_dataset:
                examples, label_list = get_examples_from_json(data_path)
            elif motherfile:
                examples, label_list = get_examples_from_motherfile(data_path)
            elif xlm_dataset:
                examples, label_list = get_examples_from_xml(data_path)
            else:
                examples, label_list = get_examples(data_path, 'train')
            random.shuffle(examples)
            train_examples = examples[0:int(len(examples)*data_divider)]
            val_examples = examples[int(len(examples)*data_divider):]
            eval_examples = examples[(int(len(examples)*data_divider)+int(len(examples)*((1-data_divider)/2))):]
        else:
            train_examples = None
            if json_dataset:
                examples, label_list = get_examples_from_json(data_path)
            elif motherfile:
                train_examples, train_label_list = get_examples_from_motherfile(data_path, 'train')
                val_examples, val_label_list = get_examples_from_motherfile(data_path, 'test')
                train_label_list.extend(val_label_list)
                label_list = list(set(train_label_list))
            elif xlm_dataset:
                examples, label_list = get_examples_from_xml(data_path)
            else:
                train_examples, label_list = get_examples(data_path, 'train')
        logger.info("\nDATA SIZE\n")
        logger.info("\Train  = %d\n" % len(train_examples))
        logger.info("\Val  = %d\n" % len(val_examples))
        
        num_train_optimization_steps = 0
        num_labels = len(label_list) + 1
        num_train_optimization_steps = int(
            len(train_examples) / train_batch_size / gradient_accumulation_steps) * epochs
        
        hidden_size = 300 if pretrained_path == None else 768 if 'base' in pretrained_path else 1024
        device = 'cuda:1' if (torch.cuda.is_available() and not no_cuda) else 'cpu'
        logger.info(device)

        if model_name == 'HERBERT':
            model = AutoTokenizerForTokenClassification(
                pretrained_path=pretrained_path, n_labels=num_labels, hidden_size=hidden_size, dropout_p=dropout,
                device=device)
        elif model_name == 'BERT_MULTILINGUAL':
            model = BertBaseMultilingualCased(
                pretrained_path=pretrained_path, n_labels=num_labels, hidden_size=hidden_size, dropout_p=dropout,
                device=device)
        elif model_name == 'REFORMER':
            model = Reformer(n_labels=num_labels, hidden_size=512,
                             dropout=dropout, device=device, max_seq_length=max_seq_length,
                             batch_size=train_batch_size)
        elif model_name == 'POLISH_ROBERTA':
            model = PolishRoberta(
                pretrained_path=pretrained_path, n_labels=num_labels, hidden_size=hidden_size, dropout_p=dropout,
                device=device)
        else:
            model = XLMRForTokenClassification(pretrained_path=pretrained_path,
                                n_labels=num_labels, hidden_size=hidden_size,
                                dropout=dropout, device=device)

        model.to(device)
        if wandb:
            wandb.watch(model)
        no_decay = ['bias', 'final_layer_norm.weight']

        params = list(model.named_parameters())

        optimizer_grouped_parameters = [
            {'params': [p for n, p in params if not any(
                nd in n for nd in no_decay)], 'weight_decay': weight_decay},
            {'params': [p for n, p in params if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        warmup_steps = int(warmup_proportion * num_train_optimization_steps)
        optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps, t_total=num_train_optimization_steps)

        train_features = convert_examples_to_features(
            train_examples, label_list, max_seq_length, model.encode_word)

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)

        train_data = create_dataset(train_features)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(
            train_data, sampler=train_sampler, batch_size=train_batch_size)
        if not split_train_data and not val_examples and not motherfile:
            val_examples, _ = get_examples(valid_path, 'valid')
        val_features = convert_examples_to_features(
            val_examples, label_list, max_seq_length, model.encode_word)

        val_data = create_dataset(val_features)
        
        best_val_f1 = 0.0

        for epoch_no in range(1, epochs+1):
            epoch_stats = {"epoch": epoch_no}
            logger.info("Epoch %d" % epoch_no)
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            
            model.train()
            steps = len(train_dataloader)
            for step, batch in enumerate(train_dataloader):
                batch = tuple(t.to(device) for t in batch)
                input_ids, label_ids, l_mask, valid_ids, = batch
                loss = model(input_ids, label_ids, l_mask, valid_ids)
                if gradient_accumulation_steps > 1:
                    loss = loss / gradient_accumulation_steps

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_grad_norm)
                tr_loss += loss.item()
                epoch_stats["loss"] = loss
                if wandb:
                    wandb.log({"loss": loss})
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if step % 5 == 0:
                    logger.info('Step = %d/%d; Loss = %.4f' % (step+1, steps, tr_loss / (step+1)))
                if (step + 1) % gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()
                del batch
            logger.info("\nTesting on validation set...")
            f1, report, entity_scores = evaluate_model(model, val_data, label_list, eval_batch_size, device)
            epoch_stats["validation_F1"] = f1
            print(report)
            if f1 > best_val_f1:
                best_val_f1 = f1
                logger.info("\nFound better f1=%.4f on validation set. Saving model\n" % f1)
                logger.info("%s\n" % report)
                if save:
                    torch.save(model.state_dict(), open(os.path.join(output_dir, 'model.pt'), 'wb'))
                    save_params(output_dir, dropout, num_labels, label_list)

            if epoch_save_model:
                epoch_output_dir = os.path.join(output_dir, "e%03d" % epoch_no)
                os.makedirs(epoch_output_dir)
                if save:
                    torch.save(model.state_dict(), open(os.path.join(epoch_output_dir, 'model.pt'), 'wb'))
                    save_params(epoch_output_dir, dropout, num_labels, label_list)
            if wandb:
                wandb.log(epoch_stats)
        model.cpu()
        del model, logger
        torch.cuda.empty_cache()
        return best_val_f1, entity_scores


    def evaluate(self, pretrained_path, dropout, path_model, device, num_labels, 
             data_path, label_list, max_seq_length=128, squeeze=True, eval_batch_size=32, model_name="XLMR", split_train_data=False,
             multi_source_labels=False, motherfile=False):
        hidden_size = 768 if 'base' in pretrained_path else 1024
        if model_name == 'HERBERT':
            model = AutoTokenizerForTokenClassification(
                pretrained_path=pretrained_path, n_labels=num_labels, hidden_size=hidden_size, dropout_p=dropout,
                device=device)
        elif model_name == 'BERT_MULTILINGUAL':
            model = BertBaseMultilingualCased(
                pretrained_path=pretrained_path, n_labels=num_labels, hidden_size=hidden_size, dropout_p=dropout,
                device=device)
        elif model_name == 'Reformer':
            model = Reformer(n_labels=num_labels, hidden_size=512,
                             dropout=dropout, device=device, max_seq_length=max_seq_length,
                             batch_size=train_batch_size)
        else:
            model = XLMRForTokenClassification(pretrained_path=pretrained_path,
                                n_labels=num_labels, hidden_size=hidden_size,
                                dropout=dropout, device=device)
        output_dir = path_model
        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                datefmt='%m/%d/%Y %H:%M:%S',
                level=logging.INFO,
                filename=os.path.join(output_dir, "log.txt"))
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        logger = logging.getLogger(__name__)
        state_dict = torch.load(open(os.path.join(path_model, 'model.pt'), 'rb'))
        model.load_state_dict(state_dict)
        logger.info("Loaded saved model")

        model.to(device)
        if not split_train_data:
            if motherfile:
                eval_examples, _ = get_examples_from_motherfile(data_path, 'validate')
            else: 
                eval_examples, _ = get_examples(data_path)

        eval_features = convert_examples_to_features(
            eval_examples, label_list, max_seq_length, model.encode_word)
        
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", eval_batch_size)
        eval_data = create_dataset(eval_features)
        f1_score, report, entity_scores = evaluate_model(model, eval_data, label_list, eval_batch_size, device)
        logger.info("\n%s", report)

        
        output_eval_file = os.path.join(output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Writing results to file *****")
            writer.write(report)
            logger.info("Done.")
