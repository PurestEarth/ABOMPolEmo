
from torch.utils.data import SequentialSampler, DataLoader
import torch
from collections import defaultdict
import numpy as np

def add_xlmr_args(parser):
    """
    Adds training and validation arguments to the passed parser
    """
    parser.add_argument('--data_dir', required=True, metavar='PATH', help='path to input')
    parser.add_argument('--output_dir', required=True, metavar='PATH', help='directory where model shall be saved')
    parser.add_argument('--valid', metavar='PATH', help='directory to validation data')
    parser.add_argument('--model_name', required=True, metavar='PATH', help='name of the model to train - LSTM | POLISHROBERTA | Reformer | HERBERT | BERT_MULTILINGUAL | XLMR')
    parser.add_argument('--embedding', metavar='PATH', help='path to embeddings')
    parser.add_argument('--pretrained_path', metavar='PATH', help='path to pretrained model')
    parser.add_argument('--seed', type=int, default=44, help='seed')
    parser.add_argument('--wandb', type=str,
                            help="Wandb project id. If present the training data will be logged using wandb api.")  
    parser.add_argument('--max_seq_length', type=int, default=128, help='max sequence length')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='gradient accumulation steps')
    parser.add_argument('--epochs', required=True, default=32, type=int, metavar='num', help='number of epochs')
    parser.add_argument('-g', nargs='+', help='which GPUs to use', default=0)
    parser.add_argument("--train_batch_size", default=32, type=int, help="Total batch size for training.")
    parser.add_argument("--warmup_proportion",default=0.1,type=float,
                        help="Proportion of training to perform linear learning rate warmup for. " "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--split_train_data", default=False, help="Split train data into multiple batches.")
    parser.add_argument('--motherfile', action='store_true', default=False, help = "whether used dataset is motherfile")  
    return parser

def get_entities(seq, suffix=False):
    """Gets entities from sequence.

    Args:
        seq (list): sequence of labels.

    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).

    Example:
        >>> from seqeval.metrics.sequence_labeling import get_entities
        >>> seq = ['B-PER', 'I-PER', 'O', 'B-LOC']
        >>> seq = ['B-PER', 'I-PER', 'O', 'B-LOC']
        >>> get_entities(seq)
        [('PER', 0, 1), ('LOC', 3, 3)]
    """
    # for nested list
    if any(isinstance(s, list) for s in seq):
        seq = [item for sublist in seq for item in sublist]

    existing_tags = []
    chunks = []
    for i, chunk in enumerate(seq):
        for et in existing_tags:
            et['continued'] = False
        active_types = map(lambda x: x['type'] ,existing_tags)
        i_chunk = []
        if '#' in chunk:
            for ann in chunk.split('#'):
               i_chunk.append(ann) 
        else:
            i_chunk.append(chunk)  
        for subchunk in i_chunk:
            if 'B-' in subchunk or 'I-' in subchunk:
                tag, type_ = get_tag_type(suffix, subchunk)
                if start_of_chunk(tag,type_) and (tag == 'B' or type_ not in active_types):
                    existing_tags.append( {'begin': i, 'continued': True, 'type': type_} )
                if tag == 'I':
                    for et in existing_tags:
                        if et['type'] == type_:
                            et['continued'] = True
            elif subchunk != 'O':
                chunks.append((subchunk, i, i))
        notFinished = []
        for et in existing_tags:
            if et['continued'] :
                notFinished.append(et)
            else:
                chunks.append((et['type'], et['begin'], i-1))
        existing_tags = notFinished
    return chunks


def get_tag_type(suffix, chunk):
    if suffix:
        tag = chunk[-1]
        type_ = chunk.split('-')[0]
    else:
        tag = chunk[0]
        type_ = '-'.join(chunk.split('-')[1:])
    return tag, type_


def start_of_chunk(tag, type_, prev_type=None, prev_tag='O'):
    """Checks if a chunk started between the previous and current word.

    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
        prev_type: previous type.
        type_: current type.

    Returns:
        chunk_start: boolean.
    """
    chunk_start = False

    if tag == 'B': chunk_start = True
    if tag == 'S': chunk_start = True

    if prev_tag == 'E' and tag == 'E': chunk_start = True
    if prev_tag == 'E' and tag == 'I': chunk_start = True
    if prev_tag == 'S' and tag == 'E': chunk_start = True
    if prev_tag == 'S' and tag == 'I': chunk_start = True
    if prev_tag == 'O' and tag == 'E': chunk_start = True
    if prev_tag == 'O' and tag == 'I': chunk_start = True

    if tag != 'O' and tag != '.' and prev_type != type_:
        chunk_start = True

    return chunk_start


def classification_report(y_true, y_pred, digits=2, suffix=False):
    """Build a text report showing the main classification metrics.

    Args:
        y_true : 2d array. Ground truth (correct) target values.
        y_pred : 2d array. Estimated targets as returned by a classifier.
        digits : int. Number of digits for formatting output floating point values.

    Returns:
        report : string. Text summary of the precision, recall, F1 score for each class.

    Examples:
        >>> from seqeval.metrics import classification_report
        >>> y_true = [['O', 'O', 'O', 'a_zero', 'a_minus_m', 'O', 'O'], ['O', 'O', 'O']]
        >>> y_pred = [['O', 'O', 'a_zero', 'a_plus_s', 'O', 'O', 'O'], ['O', 'O', 'O']]
        >>> print(classification_report(y_true, y_pred))
                     precision    recall  f1-score   support
        <BLANKLINE>
          a_minus_m       0.00      0.00      0.00         1
          a_zero       1.00      1.00      1.00         1
        <BLANKLINE>
          micro avg       0.50      0.50      0.50         2
          macro avg       0.50      0.50      0.50         2
        <BLANKLINE>
    """
    true_entities = set(get_entities(y_true))
    pred_entities = set(get_entities(y_pred))
    entity_scores = []
    name_width = 0
    d1 = defaultdict(set)
    d2 = defaultdict(set)
    for e in true_entities:
        d1[e[0]].add((e[1]))
        name_width = max(name_width, len(e[0]))
    for e in pred_entities:
        d2[e[0]].add((e[1]))

    last_line_heading = 'macro avg'
    width = max(name_width, len(last_line_heading), digits)

    headers = ["precision", "recall", "f1-score", "support"]
    head_fmt = u'{:>{width}s} ' + u' {:>9}' * len(headers)
    report = head_fmt.format(u'', *headers, width=width)
    report += u'\n\n'

    row_fmt = u'{:<{width}s} ' + u' {:>9.{digits}f}' * 3 + u' {:>9}\n'

    ps, rs, f1s, s = [], [], [], []
    for type_name, true_entities in sorted(d1.items(), key=lambda v: v[0]):
        pred_entities = d2[type_name]
        nb_correct = len(true_entities & pred_entities)
        nb_pred = len(pred_entities)
        nb_true = len(true_entities)

        p = nb_correct / nb_pred if nb_pred > 0 else 0
        r = nb_correct / nb_true if nb_true > 0 else 0
        f1 = 2 * p * r / (p + r) if p + r > 0 else 0

        report += row_fmt.format(*[type_name, p, r, f1, nb_true], width=width, digits=digits)

        ps.append(p)
        rs.append(r)
        f1s.append(f1)
        s.append(nb_true)
        entity_scores.append({'name': type_name,'precision': p, 'recall': r, 'f1': f1, 'support': nb_true})
    report += u'\n'

    # compute averages
    report += row_fmt.format('micro avg',
                             precision_score(y_true, y_pred, suffix=suffix),
                             recall_score(y_true, y_pred, suffix=suffix),
                             f1_score(y_true, y_pred, suffix=suffix)[0],
                             np.sum(s),
                             width=width, digits=digits)
    report += row_fmt.format(last_line_heading,
                             np.average(ps, weights=s),
                             np.average(rs, weights=s),
                             np.average(f1s, weights=s),
                             np.sum(s),
                             width=width, digits=digits)
    return report, entity_scores


def evaluate_model(model, eval_dataset, label_list, batch_size, device):
     """
     Evaluates an NER model on the eval_dataset provided.
     Returns:
          F1_score: Macro-average f1_score on the evaluation dataset.
          Report: detailed classification report 
     """

     eval_sampler = SequentialSampler(eval_dataset)
     eval_dataloader = DataLoader(
          eval_dataset, sampler=eval_sampler, batch_size=batch_size)

     model.eval() # turn of dropout
     y_true = []
     y_pred = []
     ignored_label = "IGNORE"
     label_map = {i: label for i, label in enumerate(label_list, 1)}
     label_map[0] = ignored_label # 0 label is to be ignored
     label_dict = {}
     for input_ids, label_ids, l_mask, valid_ids in eval_dataloader:
        input_ids = input_ids.to(device)
        label_ids = label_ids.to(device)

        valid_ids = valid_ids.to(device)
        l_mask = l_mask.to(device)

        with torch.no_grad():
            logits = model(input_ids, labels=None, labels_mask=None,
                            valid_mask=valid_ids)
        logits = torch.argmax(logits, dim=2)
        logits = logits.detach().cpu().numpy()
        label_ids = label_ids.cpu().numpy()

        for i, cur_label in enumerate(label_ids):
            temp_1 = []
            temp_2 = []

            for j, m in enumerate(cur_label):
                if valid_ids[i][j]:  # if it's a valid label
                        temp_1.append(label_map[m])
                        temp_2.append(label_map[logits[i][j]])

            assert len(temp_1) == len(temp_2)
            y_true.append(temp_1)
            y_pred.append(temp_2)

        del input_ids, label_ids, valid_ids, l_mask
        torch.cuda.empty_cache()
     report, entity_scores  = classification_report(y_true, y_pred, digits=4)
     f1, precision = f1_score(y_true, y_pred, average='Macro')
     recall = recall_score(y_true, y_pred, average='Macro')
     return f1, report, entity_scores, precision, recall


def precision_score(y_true, y_pred, average='micro', suffix=False):
    """Compute the precision.

    The precision is the ratio ``tp / (tp + fp)`` where ``tp`` is the number of
    true positives and ``fp`` the number of false positives. The precision is
    intuitively the ability of the classifier not to label as positive a sample.

    The best value is 1 and the worst value is 0.

    Args:
        y_true : 2d array. Ground truth (correct) target values.
        y_pred : 2d array. Estimated targets as returned by a tagger.

    Returns:
        score : float.

    Example:
        >>> from seqeval.metrics import precision_score
        >>> y_true = [['O', 'O', 'O', 'a_minus_m', 'a_zero', 'O', 'O'], ['O', 'O', 'O']]
        >>> y_pred = [['O', 'O', 'a_minus_m', 'a_zero', 'O', 'O', 'O'], ['O', 'O', 'O']]
        >>> precision_score(y_true, y_pred)
        0.50
    """
    true_entities = set(get_entities(y_true))
    pred_entities = set(get_entities(y_pred))

    nb_correct = len(true_entities & pred_entities)
    nb_pred = len(pred_entities)

    score = nb_correct / nb_pred if nb_pred > 0 else 0

    return score


def recall_score(y_true, y_pred, average='micro', suffix=False):
    """Compute the recall.

    The recall is the ratio ``tp / (tp + fn)`` where ``tp`` is the number of
    true positives and ``fn`` the number of false negatives. The recall is
    intuitively the ability of the classifier to find all the positive samples.

    The best value is 1 and the worst value is 0.

    Args:
        y_true : 2d array. Ground truth (correct) target values.
        y_pred : 2d array. Estimated targets as returned by a tagger.

    Returns:
        score : float.

    Example:
        >>> from seqeval.metrics import recall_score
        >>> y_true = [['O', 'O', 'O', 'a_zero', 'a_minus_m', 'O', 'O'], ['O', 'O', 'O']]
        >>> y_pred = [['O', 'O', 'a_zero', 'a_plus_s', 'O', 'O', 'O'], ['O', 'O', 'O']]
        >>> recall_score(y_true, y_pred)
        0.50
    """
    true_entities = set(get_entities(y_true))
    pred_entities = set(get_entities(y_pred))

    nb_correct = len(true_entities & pred_entities)
    nb_true = len(true_entities)

    score = nb_correct / nb_true if nb_true > 0 else 0

    return score


def f1_score(y_true, y_pred, average='micro', suffix=False):
    """Compute the F1 score.

    The F1 score can be interpreted as a weighted average of the precision and
    recall, where an F1 score reaches its best value at 1 and worst score at 0.
    The relative contribution of precision and recall to the F1 score are
    equal. The formula for the F1 score is::

        F1 = 2 * (precision * recall) / (precision + recall)

    Args:
        y_true : 2d array. Ground truth (correct) target values.
        y_pred : 2d array. Estimated targets as returned by a tagger.

    Returns:
        score : float.
        precision: float.

    Example:
        >>> from seqeval.metrics import f1_score
        >>> y_true = [['O', 'O', 'O', 'a_zero', 'a_minus_m', 'O', 'O'], ['O', 'O', 'O']]
        >>> y_pred = [['O', 'O', 'a_zero', 'a_plus_s', 'O', 'O', 'O'], ['O', 'O', 'O']]
        >>> f1_score(y_true, y_pred)
        0.50
    """
    true_entities = set(get_entities(y_true))
    pred_entities = set(get_entities(y_pred))

    nb_correct = len(true_entities & pred_entities)
    nb_pred = len(pred_entities)
    nb_true = len(true_entities)

    p = nb_correct / nb_pred if nb_pred > 0 else 0
    r = nb_correct / nb_true if nb_true > 0 else 0
    score = 2 * p * r / (p + r) if p + r > 0 else 0

    return score, p


def process(model, sentences, label_list, max_seq_length, device, show_progress=False):
        """
        @param sentences -- array of array of words, [['Jan', 'z', 'Warszawy'], ['IBM', 'i', 'Apple']]
        @param max_seq_length -- the maximum total input sequence length after WordPiece tokenization
        @param squeeze -- boolean enabling squeezing multiple sentences into one Input Feature
        """
        examples = []
        for idx, tokens in enumerate(sentences):
            guid = str(idx)
            text_a = ' '.join(tokens)
            text_b = None
            label = ["O"] * len(tokens)
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

        eval_features = convert_examples_to_features(examples, label_list, max_seq_length,
                                                    model.encode_word)
        eval_dataset = create_dataset(eval_features)
        eval_dataloader = DataLoader(eval_dataset, batch_size=1)

        y_pred = []
        sum_pred = []
        label_map = {i: label for i, label in enumerate(label_list, 1)}

        if show_progress:
            outer = tqdm.tqdm(total=len(eval_dataloader), desc='Processing', position=0)
        for input_ids, label_ids, l_mask, valid_ids in eval_dataloader:
            if show_progress:
                outer.update(1)
            input_ids = input_ids.to(device)
            label_ids = label_ids.to(device)
            valid_ids = valid_ids.to(device)

            with torch.no_grad():
                logits = model(input_ids, labels=None, labels_mask=None, valid_mask=valid_ids)

            logits = torch.argmax(logits, dim=2)
            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.cpu().numpy()
            for i, cur_label in enumerate(label_ids):
                temp_1 = []
                temp_2 = []

                for j, m in enumerate(cur_label):
                    if valid_ids[i][j]:
                        temp_1.append(label_map[m])
                        temp_2.append(label_map[logits[i][j]])

                assert len(temp_1) == len(temp_2)
                y_pred.append(temp_2)
        pointer = 0
        for sentence in sentences:
            y_pred.append(sum_pred[pointer: (pointer+len(sentence))])
            pointer += len(sentence)
        return y_pred