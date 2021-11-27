import os
import json
import torch

def ReadData(Dataset,type):
    type = type + '.json'  if type=='rel2id' else type + '_triples.json'
    pathtoData = os.path.join(os.sep,os.path.dirname(__file__),'data4PRGC',Dataset)
    pathtoFile = os.path.join(os.sep,pathtoData,type)
    with open(pathtoFile,"r") as file:
        data=json.load(file)
    if type=='rel2id.json':
        return data[0]
    else:
        text = []; labels = []
        for example in data:
            text.append(example['text']); labels.append(example['triple_list'])
        return text , labels

def FindHeadIdx(head_token_id,input_ids):
    return list(input_ids).index(head_token_id)

def GetAllLabels(rel2id,str_labels,input_ids,attention_mask,tokenizer,seq_len):
    rel_num = len(rel2id)
    tot_ex_len = sum([len(relations) for relations in str_labels])
    max_triplet_per_ex = max([len(label) for label in str_labels])
    max_subj_ent_token_per_triplet = max([len(tokenizer.tokenize(label[0])) for example in str_labels for label in example])
    max_obj_ent_token_per_triplet = max([len(tokenizer.tokenize(label[2])) for example in str_labels for label in example])
    
    
    
    subj_entity_labels = -1 *  torch.ones(size=(len(str_labels),max_triplet_per_ex,max_subj_ent_token_per_triplet),dtype=torch.long)
    obj_entity_labels = -1 *  torch.ones(size=(len(str_labels),max_triplet_per_ex,max_obj_ent_token_per_triplet),dtype=torch.long)



    labels_list = [{} for i in range(tot_ex_len)]
    tot_ex_counter = 0; tot_ex_counter_lagged = 0
    for idx1,example in enumerate(str_labels):
        relation_labels = torch.zeros(rel_num,dtype=torch.long)
        corres_matrix = torch.zeros(size=(seq_len,seq_len),dtype=torch.long)
        for idx2,label in enumerate(example):
            
            subj_seq_tag = torch.zeros(seq_len,dtype=torch.long)
            obj_seq_tag = torch.zeros(seq_len,dtype=torch.long)
            target_rel = rel2id[label[1]]

            subj_token_ids = torch.tensor(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(label[0])))
            subj_entity_labels[idx1,idx2,:len(subj_token_ids)] = subj_token_ids
            subj_head_idx = FindHeadIdx(subj_token_ids[0],input_ids[idx1])
            subj_seq_tag[subj_head_idx] = 1; subj_seq_tag[(subj_head_idx+1):(subj_head_idx+len(subj_token_ids))] = 2

            relation_labels[rel2id[label[1]]] = 1

            obj_token_ids = torch.tensor(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(label[2])))
            obj_entity_labels[idx1,idx2,:len(obj_token_ids)] = obj_token_ids
            obj_head_idx = FindHeadIdx(obj_token_ids[0],input_ids[idx1])
            obj_seq_tag[obj_head_idx] = 1; obj_seq_tag[(obj_head_idx+1):(obj_head_idx+len(obj_token_ids))] = 2

            corres_matrix[subj_head_idx,obj_head_idx] = 1

            labels_list[tot_ex_counter]['input_ids'] = input_ids[idx1]; 
            labels_list[tot_ex_counter]['attention_mask'] = attention_mask[idx1]
            labels_list[tot_ex_counter]['target_rel'] = target_rel
            labels_list[tot_ex_counter]['subj_seq_tag'] = subj_seq_tag
            labels_list[tot_ex_counter]['obj_seq_tag'] = obj_seq_tag

            tot_ex_counter+=1

        for idx in range(len(example)):
            labels_list[tot_ex_counter_lagged]['corres_matrix'] = corres_matrix
            labels_list[tot_ex_counter_lagged]['relation_labels'] = relation_labels
            tot_ex_counter_lagged+=1

    return labels_list