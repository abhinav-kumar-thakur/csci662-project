from torch.utils.data import Dataset
import torch
import os
import json
import utils

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

def GetEntRange(input_ids, ent_str, tokenizer):
    ent_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(ent_str))
    head_idx = FindHeadIdx(ent_ids[0], input_ids)
    range_ent = (head_idx, head_idx + len(ent_ids) - 1)
    return range_ent

def GetTrainFeatures(rel2id, str_labels, input_ids, attention_mask, tokenizer, seq_len):
    rel_num = len(rel2id)

    labels_list = []
    tot_ex_counter = 0
    tot_ex_counter_lagged = 0
    for idx1, example in enumerate(str_labels):
        relation_labels = torch.zeros(rel_num, dtype=torch.long)
        corres_matrix = torch.zeros(size=(seq_len, seq_len), dtype=torch.long)
        sentence_rel = []
        for label in example:
            subj_seq_tag = torch.zeros(seq_len, dtype=torch.long)
            obj_seq_tag = torch.zeros(seq_len, dtype=torch.long)
            target_rel = rel2id[label[1]]

            subj_token_ids = torch.tensor(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(label[0])))
            subj_head_idx = FindHeadIdx(subj_token_ids[0], input_ids[idx1])
            subj_seq_tag[subj_head_idx] = 1
            subj_seq_tag[(subj_head_idx + 1):(subj_head_idx + len(subj_token_ids))] = 2

            relation_labels[rel2id[label[1]]] = 1

            obj_token_ids = torch.tensor(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(label[2])))
            obj_head_idx = FindHeadIdx(obj_token_ids[0], input_ids[idx1])
            obj_seq_tag[obj_head_idx] = 1
            obj_seq_tag[(obj_head_idx + 1):(obj_head_idx + len(obj_token_ids))] = 2

            corres_matrix[subj_head_idx, obj_head_idx] = 1

            if (target_rel in sentence_rel):
                # Same relation occured twice in sentence for different entities, don't add as new train example,
                # only extend set of entity tags (model should detect this relation and all corresponding subj/obj tags)
                idx_rel_example = tot_ex_counter - len(sentence_rel) + sentence_rel.index(target_rel)
                labels_list[idx_rel_example]['subj_seq_tag'] = torch.where((labels_list[idx_rel_example]['subj_seq_tag']==0) & (subj_seq_tag!=0),
                                                                           subj_seq_tag, labels_list[idx_rel_example]['subj_seq_tag'])
                labels_list[idx_rel_example]['obj_seq_tag'] = torch.where((labels_list[idx_rel_example]['obj_seq_tag']==0) & (obj_seq_tag!=0),
                                                                           obj_seq_tag, labels_list[idx_rel_example]['obj_seq_tag'])
            else:
                # new train example for this target relation
                labels_list.append({})
                labels_list[tot_ex_counter]['input_ids'] = input_ids[idx1]
                labels_list[tot_ex_counter]['attention_mask'] = attention_mask[idx1]
                labels_list[tot_ex_counter]['target_rel'] = target_rel
                labels_list[tot_ex_counter]['subj_seq_tag'] = subj_seq_tag 
                labels_list[tot_ex_counter]['obj_seq_tag'] = obj_seq_tag 
                tot_ex_counter += 1

            sentence_rel.append(target_rel)

        for idx in range(len(set(sentence_rel))):
            labels_list[tot_ex_counter_lagged]['corres_matrix'] = corres_matrix
            labels_list[tot_ex_counter_lagged]['relation_labels'] = relation_labels
            tot_ex_counter_lagged += 1

    return labels_list

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

class TrainDataset(Dataset):
    def __init__(self, text, labels, rel2id, tokenizer, max_plm_seq_len):
        self.text = text
        self.labels = labels
        self.rel2id = rel2id
        self.tokenizer = tokenizer
        self.encoded_text = self.tokenizer(
            text,
            add_special_tokens=False,
            padding=True,
            max_length=max_plm_seq_len,
            truncation=True,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors='pt',
        )
        self.seq_len = self.encoded_text['input_ids'].shape[1]
        self.labels = GetTrainFeatures(rel2id, labels, self.encoded_text['input_ids'], self.encoded_text['attention_mask'], self.tokenizer, self.seq_len)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.labels[idx]['input_ids'],
            'attention_mask': self.labels[idx]['attention_mask'],
            'subj_seq_tag': self.labels[idx]['subj_seq_tag'],
            'obj_seq_tag': self.labels[idx]['obj_seq_tag'],
            'target_rel': self.labels[idx]['target_rel'],
            'corres_matrix': self.labels[idx]['corres_matrix'],
            'relation_labels': self.labels[idx]['relation_labels']
        }

class ValDataset(Dataset):
    def __init__(self, text, labels, rel2id, tokenizer, seq_len):
        self.text = text
        self.labels = labels
        self.rel2id = rel2id
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.encoded_text = self.tokenizer(
            text,
            add_special_tokens=False,
            padding='max_length',
            max_length=seq_len,
            truncation=True,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors='pt',
        )
        self.labels = GetValFeatures(rel2id, labels, self.encoded_text['input_ids'], self.encoded_text['attention_mask'], self.tokenizer, self.seq_len)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.labels[idx]['input_ids'], self.labels[idx]['attention_mask'], self.labels[idx]['triples']
