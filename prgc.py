import torch
import torch.nn as nn

class PRGC(nn.Module):
    def __init__(self, bert, rel_num, lambda_1, lambda_2, fusion_type):
        super().__init__()

        self.fusion_type = fusion_type

        # Bert Embedding
        self.bert = bert
        hidden_dim = bert.config.hidden_size

        # Stage 1
        self.hidden_relational = nn.Linear(hidden_dim, hidden_dim // 2)
        self.dropout_rel = nn.Dropout(p=0.3)
        self.rel_judge = nn.Linear(hidden_dim // 2, rel_num)
        self.lambda_1 = lambda_1

        # Stage 2
        fusion_multiplier = 2 if self.fusion_type == 'concat' else 1
        self.subj_hidden_tag = nn.Linear(fusion_multiplier * hidden_dim, fusion_multiplier * hidden_dim // 2)
        self.obj_hidden_tag = nn.Linear(fusion_multiplier * hidden_dim, fusion_multiplier * hidden_dim // 2)
        self.tag_subject = nn.Linear(fusion_multiplier * hidden_dim // 2, 3)
        self.tag_object = nn.Linear(hidden_dim * fusion_multiplier // 2, 3)
        self.dropout_subj = nn.Dropout(p=0.3)
        self.dropout_obj = nn.Dropout(p=0.3)
        # Relation Trainable Embedding Matrix
        self.rel_embedding = nn.Embedding(num_embeddings=rel_num, embedding_dim=hidden_dim)

        # Stage 3
        self.hidden_corres = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout_corres = nn.Dropout(p=0.3)
        self.global_corr = nn.Linear(hidden_dim, 1)
        self.lambda_2 = lambda_2

    def forward(self, input_ids, attention_mask, target_rel, mode):
        # Bert Embedding
        # with torch.no_grad():
        embed_tokens = self.bert(input_ids=input_ids, attention_mask=attention_mask)['last_hidden_state']
        # Stage 1
        embedded_masked = embed_tokens * attention_mask[:, :, None]
        num_nonmasked = torch.sum(attention_mask, dim=1)
        stage1_avgpool = torch.sum(embedded_masked, dim=1) / num_nonmasked[:, None]
        rel_hidden = torch.relu(self.hidden_relational(stage1_avgpool))
        rel_hidden_drop = self.dropout_rel(rel_hidden)
        stage1 = self.rel_judge(rel_hidden_drop)

        # Stage 3
        # TODO: Check if masked embed_tokens is better
        sub_matrix = embed_tokens.unsqueeze(1).repeat((1, input_ids.shape[1], 1, 1))
        concat_matrix = torch.cat([sub_matrix, torch.transpose(sub_matrix, 1, 2)], axis=3)
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
            sentence_mask = []
            for sentence_id, rel_id in zip(sentence_ids, rel_ids):
                sentence_embedded.append(embed_tokens[sentence_id])
                sentence_mask.append(attention_mask[sentence_id])

            embed_tokens = torch.stack(sentence_embedded)
            attention_mask = torch.stack(sentence_mask)
            target_rel_emb = self.rel_embedding(rel_ids)

        # TODO: Check if masked embed_tokens is better
        target_rel_emb = target_rel_emb.unsqueeze(1).repeat((1, input_ids.shape[1], 1))
        fusion = torch.cat([target_rel_emb, embed_tokens], dim=-1) if self.fusion_type == 'concat' else target_rel_emb + embed_tokens
        subj_pred_tag = self.tag_subject(self.dropout_subj(torch.relu(self.subj_hidden_tag(fusion))))
        obj_pred_tag = self.tag_object(self.dropout_obj(torch.relu(self.obj_hidden_tag(fusion))))

        if mode == 'eval':
            return torch.stack((sentence_ids, rel_ids), dim=1), subj_pred_tag, obj_pred_tag, pred_corres_matrix

        return stage1, subj_pred_tag, obj_pred_tag, pred_corres_matrix
