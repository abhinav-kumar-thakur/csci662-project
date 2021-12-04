from collections import defaultdict
from itertools import product

import torch

def FindEntLen(ent_head_idx, ent_tags):
    count = 1
    for i in range(ent_head_idx + 1, len(ent_tags)):
        if ent_tags[i] == 2:
            count += 1
            continue
        break
    return count

def GenerateTriples(sentence_rel_map, subj_pred_tag, obj_pred_tag, pred_corres_matrix, attention_mask, lambda_2):
    triples = {i: [] for i in range(pred_corres_matrix.shape[0])}
    pred_corres_matrix = torch.sigmoid(pred_corres_matrix).squeeze(-1) * (attention_mask.unsqueeze(-1) * attention_mask.unsqueeze(1))
    for idx, sentence_rel in enumerate(sentence_rel_map):
        rel_id = sentence_rel[1].item()
        sub_tags = torch.argmax(subj_pred_tag[idx], dim=1)
        obj_tags = torch.argmax(obj_pred_tag[idx], dim=1)

        sub_head_candidates = (sub_tags == 1).nonzero().squeeze(1).tolist()
        obj_head_candidates = (obj_tags == 1).nonzero().squeeze(1).tolist()
        subj_obj_corres = (pred_corres_matrix[sentence_rel[0]] > lambda_2)

        possible_pairs = list(product(sub_head_candidates, obj_head_candidates))
        triple = [((s, s + FindEntLen(s, sub_tags) - 1), rel_id, (o, o + FindEntLen(o, obj_tags) - 1)) for s, o in possible_pairs if subj_obj_corres[s, o]]

        triples[sentence_rel[0].item()].extend(triple)
    return [v for k, v in triples.items()]

def FindMatches(true_triples, pred_triples):
    gold_num = sum([len(triples) for triples in true_triples])
    pred_num = sum([len(triples) for triples in pred_triples])
    correct_num = 0
    for i in range(len(true_triples)):
        correct_num += len( set(true_triples[i]) & set(pred_triples[i]) )
    return correct_num, pred_num, gold_num

def get_metrics(correct_num, predict_num, gold_num):
    p = correct_num / predict_num if predict_num > 0 else 0
    r = correct_num / gold_num if gold_num > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    return {
        'correct_num': correct_num,
        'predict_num': predict_num,
        'gold_num': gold_num,
        'precision': p,
        'recall': r,
        'f1': f1
    }

def val_collate_fn(labels_list):
    input_ids = torch.stack([f[0] for f in labels_list])
    attention_mask = torch.stack([f[1] for f in labels_list])
    triples = [f[2] for f in labels_list]
    return [input_ids, attention_mask, triples]

# Class to log file for each epoch
class EpochMetricFileLogger():
    def __init__(self, file_path):
        self.file_path = file_path
        self.metrics = defaultdict(list)

    def append(self, metric_name, value):
        self.metrics[metric_name].append(value)

    def log(self):
        with open(self.file_path, 'w+') as f:
            for metric_name, values in self.metrics.items():
                f.write(f'{metric_name}: {sum(values) / len(values)}\n')
            f.write('\n')
