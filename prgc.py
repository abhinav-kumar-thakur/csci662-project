import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel

class PRGC(BertPreTrainedModel):
    def __init__(self, config, params):
        super().__init__(config)

        self.fusion_type = params['fusion_type']

        # Bert Embedding
        self.bert = BertModel(config)
        hidden_dim = self.bert.config.hidden_size
        self.lambda_1 = params['lambda_1']
        self.lambda_2 = params['lambda_2']

        # Stage 2
        fusion_multiplier = 2 if self.fusion_type == 'concat' else 1
        self.subj_hidden_tag = nn.Linear(fusion_multiplier * hidden_dim, fusion_multiplier * hidden_dim // 2)
        self.dropout_subj = nn.Dropout(p=0.3)
        self.tag_subject = nn.Linear(fusion_multiplier * hidden_dim // 2, 3)
        self.obj_hidden_tag = nn.Linear(fusion_multiplier * hidden_dim, fusion_multiplier * hidden_dim // 2)
        self.dropout_obj = nn.Dropout(p=0.3)
        self.tag_object = nn.Linear(hidden_dim * fusion_multiplier // 2, 3)


        # Stage 3
        self.hidden_corres = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout_corres = nn.Dropout(p=0.3)
        self.global_corr = nn.Linear(hidden_dim, 1)

        # Stage 1
        self.hidden_relational = nn.Linear(hidden_dim, hidden_dim // 2)
        self.dropout_rel = nn.Dropout(p=0.3)
        self.rel_judge = nn.Linear(hidden_dim // 2, params['rel_num'])
        
        # Relation Trainable Embedding Matrix
        self.rel_embedding = nn.Embedding(num_embeddings=params['rel_num'], embedding_dim=hidden_dim)

        self.init_weights()

    def forward(self, input_ids, attention_mask, target_rel, mode):
        # Bert Embedding
        # with torch.no_grad():
        embed_tokens = self.bert(input_ids=input_ids, attention_mask=attention_mask)[0]
        # Stage 1
        embedded_masked = embed_tokens * attention_mask[:, :, None]
        num_nonmasked = torch.sum(attention_mask, dim=1)
        stage1_avgpool = torch.sum(embedded_masked, dim=1) / num_nonmasked[:, None]
        rel_hidden = torch.relu(self.hidden_relational(stage1_avgpool))
        rel_hidden_drop = self.dropout_rel(rel_hidden)
        stage1 = self.rel_judge(rel_hidden_drop)

        # Stage 3
        # TODO: Check if masked embed_tokens is better
        #sub_matrix = embed_tokens.unsqueeze(1).repeat((1, input_ids.shape[1], 1, 1))
        #concat_matrix = torch.cat([sub_matrix, torch.transpose(sub_matrix, 1, 2)], axis=3)
        sub_matrix = embed_tokens.unsqueeze(2).expand(-1, -1, input_ids.shape[1], -1)  # (bs, s, s, h)
        obj_matrix = embed_tokens.unsqueeze(1).expand(-1, input_ids.shape[1], -1, -1)  # (bs, s, s, h)
        concat_matrix = torch.cat([sub_matrix, obj_matrix], 3)
        corres_hidden = torch.relu(self.hidden_corres(concat_matrix))
        corres_hidden_drop = self.dropout_corres(corres_hidden)
        pred_corres_matrix = self.global_corr(corres_hidden_drop)

        # Stage 2
        if mode == 'train':
            target_rel_emb = self.rel_embedding(target_rel)

        elif mode == 'eval':
            potential_relations = torch.sigmoid(stage1) > self.lambda_1

            for ii in range(stage1.shape[0]): # if not a single relation passes the threshold, force the one with largest logit to be selected
                if 1 not in potential_relations[ii]:
                    max_idx = torch.argmax(stage1[ii])
                    potential_relations[ii][max_idx] = 1

            sentence_ids, rel_ids = torch.nonzero(potential_relations, as_tuple=True)
            sentence_embedded = []
            for sentence_id, rel_id in zip(sentence_ids, rel_ids):
                sentence_embedded.append(embed_tokens[sentence_id])

            embed_tokens = torch.stack(sentence_embedded)
            target_rel_emb = self.rel_embedding(rel_ids)

        # TODO: Check if masked embed_tokens is better
        target_rel_emb = target_rel_emb.unsqueeze(1).repeat((1, input_ids.shape[1], 1))
        fusion = torch.cat([embed_tokens, target_rel_emb], dim=-1) if self.fusion_type == 'concat' else target_rel_emb + embed_tokens
        subj_pred_tag = self.tag_subject(self.dropout_subj(torch.relu(self.subj_hidden_tag(fusion))))
        obj_pred_tag = self.tag_object(self.dropout_obj(torch.relu(self.obj_hidden_tag(fusion))))

        if mode == 'eval':
            return torch.stack((sentence_ids, rel_ids), dim=1), subj_pred_tag, obj_pred_tag, pred_corres_matrix

        return stage1, subj_pred_tag, obj_pred_tag, pred_corres_matrix
