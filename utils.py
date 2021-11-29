import json
import os
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


def GenerateTriples(sentence_rel_map, subj_pred_tag, obj_pred_tag, pred_corres_matrix, lambda_2):
    triples = {i: [] for i in range(pred_corres_matrix.shape[0])}
    for idx, sentence_rel in enumerate(sentence_rel_map):
        rel_id = sentence_rel[1].item()
        sub_tags = torch.argmax(subj_pred_tag[idx], dim=1)
        obj_tags = torch.argmax(obj_pred_tag[idx], dim=1)

        sub_head_candidates = (sub_tags == 1).nonzero().squeeze(1).tolist()
        obj_head_candidates = (obj_tags == 1).nonzero().squeeze(1).tolist()
        subj_obj_corres = (pred_corres_matrix[sentence_rel[0]].squeeze(2) > lambda_2)

        possible_pairs = list(product(sub_head_candidates, obj_head_candidates))
        triple = [((s, s + FindEntLen(s, sub_tags) - 1), rel_id, (o, o + FindEntLen(o, obj_tags) - 1)) for s, o in possible_pairs if subj_obj_corres[s, o]]

        triples[sentence_rel[0].item()].extend(triple)
    return [v for k, v in triples.items()]


def FindMatches(true_triples, pred_triples):
    gold_num = sum([len(triples) for triples in true_triples])
    pred_num = sum([len(triples) for triples in pred_triples])
    correct_num = 0
    for i in range(len(true_triples)):
        correct_num += len(set(true_triples[i]) & set(pred_triples[i]))
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


def ReadData(Dataset, type):
    type = type + '.json' if type == 'rel2id' else type + '_triples.json'
    pathtoData = os.path.join(os.sep, os.path.dirname(__file__), 'data4PRGC', Dataset)
    pathtoFile = os.path.join(os.sep, pathtoData, type)
    with open(pathtoFile, "r") as file:
        data = json.load(file)
    if type == 'rel2id.json':
        return data[0]
    else:
        text = [];
        labels = []
        for example in data:
            text.append(example['text']);
            labels.append(example['triple_list'])
        return text, labels


def FindHeadIdx(head_token_id, input_ids):
    return list(input_ids).index(head_token_id)


def GetTrainFeatures(rel2id, str_labels, input_ids, attention_mask, tokenizer, seq_len):
    rel_num = len(rel2id)
    tot_ex_len = sum([len(relations) for relations in str_labels])

    max_triplet_per_ex = max([len(label) for label in str_labels])
    max_subj_ent_token_per_triplet = max([len(tokenizer.tokenize(label[0])) for example in str_labels for label in example])
    max_obj_ent_token_per_triplet = max([len(tokenizer.tokenize(label[2])) for example in str_labels for label in example])

    subj_entity_labels = -1 * torch.ones(size=(len(str_labels), max_triplet_per_ex, max_subj_ent_token_per_triplet), dtype=torch.long)
    obj_entity_labels = -1 * torch.ones(size=(len(str_labels), max_triplet_per_ex, max_obj_ent_token_per_triplet), dtype=torch.long)

    labels_list = [{} for i in range(tot_ex_len)]
    tot_ex_counter = 0;
    tot_ex_counter_lagged = 0
    for idx1, example in enumerate(str_labels):
        relation_labels = torch.zeros(rel_num, dtype=torch.long)
        corres_matrix = torch.zeros(size=(seq_len, seq_len), dtype=torch.long)
        for idx2, label in enumerate(example):
            subj_seq_tag = torch.zeros(seq_len, dtype=torch.long)
            obj_seq_tag = torch.zeros(seq_len, dtype=torch.long)
            target_rel = rel2id[label[1]]

            subj_token_ids = torch.tensor(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(label[0])))
            subj_entity_labels[idx1, idx2, :len(subj_token_ids)] = subj_token_ids
            subj_head_idx = FindHeadIdx(subj_token_ids[0], input_ids[idx1])
            subj_seq_tag[subj_head_idx] = 1;
            subj_seq_tag[(subj_head_idx + 1):(subj_head_idx + len(subj_token_ids))] = 2

            relation_labels[rel2id[label[1]]] = 1

            obj_token_ids = torch.tensor(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(label[2])))
            obj_entity_labels[idx1, idx2, :len(obj_token_ids)] = obj_token_ids
            obj_head_idx = FindHeadIdx(obj_token_ids[0], input_ids[idx1])
            obj_seq_tag[obj_head_idx] = 1;
            obj_seq_tag[(obj_head_idx + 1):(obj_head_idx + len(obj_token_ids))] = 2

            corres_matrix[subj_head_idx, obj_head_idx] = 1

            labels_list[tot_ex_counter]['input_ids'] = input_ids[idx1];
            labels_list[tot_ex_counter]['attention_mask'] = attention_mask[idx1]
            labels_list[tot_ex_counter]['target_rel'] = target_rel
            labels_list[tot_ex_counter]['subj_seq_tag'] = subj_seq_tag
            labels_list[tot_ex_counter]['obj_seq_tag'] = obj_seq_tag

            tot_ex_counter += 1

        for idx in range(len(example)):
            labels_list[tot_ex_counter_lagged]['corres_matrix'] = corres_matrix
            labels_list[tot_ex_counter_lagged]['relation_labels'] = relation_labels
            tot_ex_counter_lagged += 1

    return labels_list


def GetEntRange(input_ids, ent_str, tokenizer):
    ent_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(ent_str))
    head_idx = FindHeadIdx(ent_ids[0], input_ids)
    range_ent = (head_idx, head_idx + len(ent_ids) - 1)
    return range_ent


def GetValFeatures(rel2id, str_labels, input_ids, attention_mask, tokenizer, seq_len):
    labels_list = [{} for i in range(len(str_labels))]

    for idx1, triples in enumerate(str_labels):
        triples_list = []
        for triple in triples:
            subj_ent_range = GetEntRange(input_ids[idx1], triple[0], tokenizer)
            rel_id = rel2id[triple[1]]
            obj_ent_range = GetEntRange(input_ids[idx1], triple[2], tokenizer)
            triples_list.append((subj_ent_range, rel_id, obj_ent_range))
        labels_list[idx1]['input_ids'] = input_ids[idx1]
        labels_list[idx1]['attention_mask'] = attention_mask[idx1]
        labels_list[idx1]['triples'] = triples_list

    return labels_list


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
        self.metrics = defaultdict[list]
