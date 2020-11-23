from seqeval.metrics import f1_score
from torch.utils.data import SequentialSampler, DataLoader
import torch


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
        >>> y_true = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> y_pred = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> print(classification_report(y_true, y_pred))
                     precision    recall  f1-score   support
        <BLANKLINE>
               MISC       0.00      0.00      0.00         1
                PER       1.00      1.00      1.00         1
        <BLANKLINE>
          micro avg       0.50      0.50      0.50         2
          macro avg       0.50      0.50      0.50         2
        <BLANKLINE>
    """
    true_entities = set(get_entities(y_true, suffix))
    pred_entities = set(get_entities(y_pred, suffix))

    name_width = 0
    d1 = defaultdict(set)
    d2 = defaultdict(set)
    for e in true_entities:
        d1[e[0]].add((e[1], e[2]))
        name_width = max(name_width, len(e[0]))
    for e in pred_entities:
        d2[e[0]].add((e[1], e[2]))

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

    report += u'\n'

    # compute averages
    report += row_fmt.format('micro avg',
                             precision_score(y_true, y_pred, suffix=suffix),
                             recall_score(y_true, y_pred, suffix=suffix),
                             f1_score(y_true, y_pred, suffix=suffix),
                             np.sum(s),
                             width=width, digits=digits)
    report += row_fmt.format(last_line_heading,
                             np.average(ps, weights=s),
                             np.average(rs, weights=s),
                             np.average(f1s, weights=s),
                             np.sum(s),
                             width=width, digits=digits)

    return report


def evaluate_model(model, eval_dataset, label_list, batch_size, device):
     """
     Evaluates an NER model on the eval_dataset provided.
     Returns:
          F1_score: Macro-average f1_score on the evaluation dataset.
          Report: detailed classification report 
     """

     # Run prediction for full data
     eval_sampler = SequentialSampler(eval_dataset)
     eval_dataloader = DataLoader(
          eval_dataset, sampler=eval_sampler, batch_size=batch_size)

     model.eval() # turn of dropout

     y_true = []
     y_pred = []

     label_map = {i: label for i, label in enumerate(label_list, 1)}

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
               

     report = classification_report(y_true, y_pred, digits=4)
     f1 = f1_score(y_true, y_pred, average='Macro')

     return f1, report