import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, BertConfig, BertTokenizer
import argparse
import utils
from tqdm import tqdm
import os


class TrainDataset(Dataset):
    def __init__(self, text, labels, rel2id, tokenizer, max_seq_len):
        self.text = text
        self.labels = labels
        self.rel2id = rel2id
        self.tokenizer = tokenizer
        self.encoded_text = self.tokenizer(
            text,
            add_special_tokens=True,
            padding=True,
            max_length=max_seq_len,
            truncation=True,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors='pt',
        )
        self.seq_len = self.encoded_text['input_ids'].shape[1]
        self.labels = utils.GetTrainFeatures(rel2id, labels, self.encoded_text['input_ids'], self.encoded_text['attention_mask'], self.tokenizer, self.seq_len)

    def __len__(self):
        return len(self.text)

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
    def __init__(self, text, labels, rel2id, tokenizer, max_seq_len):
        self.text = text
        self.labels = labels
        self.rel2id = rel2id
        self.tokenizer = tokenizer
        self.encoded_text = self.tokenizer(
            text,
            add_special_tokens=True,
            padding=True,
            max_length=max_seq_len,
            truncation=True,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors='pt',
        )
        self.seq_len = self.encoded_text['input_ids'].shape[1]
        self.labels = utils.GetValFeatures(rel2id, labels, self.encoded_text['input_ids'], self.encoded_text['attention_mask'], self.tokenizer, self.seq_len)

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        return self.labels[idx]['input_ids'], self.labels[idx]['attention_mask'], self.labels[idx]['triples']


class PRGC(nn.Module):
    def __init__(self, bert, rel_num, lambda_1, lambda_2, fusion_type):
        super().__init__()

        self.fusion_type = fusion_type

        # Bert Embedding
        self.bert = bert
        hidden_dim = bert.config.hidden_size

        # Stage 1
        self.hidden_relational = nn.Linear(hidden_dim, hidden_dim // 2)
        self.rel_judge = nn.Linear(hidden_dim // 2, rel_num)
        self.lambda_1 = lambda_1

        # Stage 2
        fusion_multiplier = 2 if self.fusion_type == 'concat' else 1
        self.hidden_tag = nn.Linear(fusion_multiplier * hidden_dim, fusion_multiplier * hidden_dim // 2)
        self.tag_subject = nn.Linear(fusion_multiplier * hidden_dim // 2, 3)
        self.tag_object = nn.Linear(hidden_dim * fusion_multiplier // 2, 3)
        # Relation Trainable Embedding Matrix
        self.rel_embedding = nn.Embedding(num_embeddings=rel_num, embedding_dim=hidden_dim)

        # Stage 3
        self.hidden_corres = nn.Linear(hidden_dim * 2, hidden_dim)
        self.global_corr = nn.Linear(hidden_dim, 1)
        self.lambda_2 = lambda_2

        # Optimizer
        self.optimizer = optim.AdamW([
            {'params': self.bert.parameters(), 'lr': 5 * 1e-5},
            {'params': self.hidden_relational.parameters(), 'lr': 0.001},
            {'params': self.rel_judge.parameters(), 'lr': 0.001},
            {'params': self.hidden_tag.parameters(), 'lr': 0.001},
            {'params': self.tag_subject.parameters(), 'lr': 0.001},
            {'params': self.tag_object.parameters(), 'lr': 0.001},
            {'params': self.rel_embedding.parameters(), 'lr': 0.001},
            {'params': self.hidden_corres.parameters(), 'lr': 0.001},
            {'params': self.global_corr.parameters(), 'lr': 0.001},
        ], weight_decay=0.01
        )

    def forward(self, input_ids, attention_mask, target_rel, mode):
        # Bert Embedding
        # with torch.no_grad():
        embed_tokens = self.bert(input_ids=input_ids, attention_mask=attention_mask)['last_hidden_state']
        # Stage 1
        embedded_masked = embed_tokens * attention_mask[:, :, None]
        num_nonmasked = torch.sum(attention_mask, dim=1)
        stage1_avgpool = torch.sum(embedded_masked, dim=1) / num_nonmasked[:, None]
        rel_hidden = torch.relu(self.hidden_relational(stage1_avgpool))
        stage1 = self.rel_judge(rel_hidden)

        # Stage 3
        # TODO: Check if masked embed_tokens is better
        sub_matrix = embed_tokens.unsqueeze(1).repeat((1, input_ids.shape[1], 1, 1))
        concat_matrix = torch.cat([sub_matrix, torch.transpose(sub_matrix, 1, 2)], axis=3)
        corres_hidden = torch.relu(self.hidden_corres(concat_matrix))
        pred_corres_matrix = self.global_corr(corres_hidden)

        # Stage 2
        if mode == 'train':
            target_rel_emb = self.rel_embedding(target_rel)

        elif mode == 'eval':
            potential_relations = torch.sigmoid(stage1) > self.lambda_1
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
        fusion = self.hidden_tag(fusion)
        subj_pred_tag = self.tag_subject(fusion)
        obj_pred_tag = self.tag_object(fusion)

        if mode == 'eval':
            return torch.stack((sentence_ids, rel_ids), dim=1), subj_pred_tag, obj_pred_tag, pred_corres_matrix

        return stage1, subj_pred_tag, obj_pred_tag, pred_corres_matrix


def train_epoch(model, iterator, epoch, device):
    file_logger = utils.EpochMetricFileLogger('epoch_train.log')

    epoch_loss = 0
    epoch_loss_rel = 0;
    epoch_loss_tag = 0;
    epoch_loss_corres = 0
    model.train()
    criterion_BCE = nn.BCEWithLogitsLoss()
    criterion_BCE_NoReduction = nn.BCEWithLogitsLoss(reduction='none')
    criterion_CE = nn.CrossEntropyLoss(reduction='none')
    tepoch = tqdm(iterator, desc=f'Epoch {epoch}', total=len(iterator))
    for bi, batch in enumerate(tepoch):
        model.optimizer.zero_grad()
        ids = batch['input_ids'].to(device);
        attention_mask = batch['attention_mask'].to(device)
        subj_seq_tag = batch['subj_seq_tag'].to(device)
        obj_seq_tag = batch['obj_seq_tag'].to(device)
        target_rel = batch['target_rel'].to(device)
        corres_matrix = batch['corres_matrix'].to(device)
        relation_labels = batch['relation_labels'].to(device)

        stage1, subj_pred_tag, obj_pred_tag, pred_corres_matrix = model(ids, attention_mask, target_rel, mode='train')

        # Stage 1 Loss
        loss_rel = criterion_BCE(stage1, relation_labels.float())
        file_logger.append('rec_rel', ((torch.sigmoid(stage1) > model.lambda_1) & relation_labels.bool()).sum().item() / relation_labels.sum().item())
        file_logger.append('prec_rel', ((torch.sigmoid(stage1) > model.lambda_1) & relation_labels.bool()).sum().item() / max(1, (torch.sigmoid(stage1) > model.lambda_1).sum().item()))

        # Stage 2 Loss
        loss_tag = (criterion_CE(subj_pred_tag.view(-1, 3), subj_seq_tag.flatten()) + criterion_CE(obj_pred_tag.view(-1, 3), obj_seq_tag.flatten()))
        loss_tag = 0.5 * (loss_tag * attention_mask.flatten()).sum() / attention_mask.sum()

        file_logger.append('prec_subj_tag_1', ((subj_seq_tag == 1) & (torch.argmax(subj_pred_tag, 2) == 1)).sum().item() / max(1, (torch.argmax(subj_pred_tag, 2) == 1).sum().item()))
        file_logger.append('prec_subj_tag_2', ((subj_seq_tag == 2) & (torch.argmax(subj_pred_tag, 2) == 2)).sum().item() / max(1, (torch.argmax(subj_pred_tag, 2) == 2).sum().item()))
        file_logger.append('rec_subj_tag_1', ((subj_seq_tag == 1) & (torch.argmax(subj_pred_tag, 2) == 1)).sum().item() / max(1, (subj_seq_tag == 1).sum().item()))
        file_logger.append('rec_subj_tag_2', ((subj_seq_tag == 2) & (torch.argmax(subj_pred_tag, 2) == 2)).sum().item() / max(1, (subj_seq_tag == 2).sum().item()))
        file_logger.append('prec_obj_tag_1', ((obj_seq_tag == 1) & (torch.argmax(obj_pred_tag, 2) == 1)).sum().item() / max(1, (torch.argmax(obj_pred_tag, 2) == 1).sum().item()))
        file_logger.append('prec_obj_tag_2', ((obj_seq_tag == 2) & (torch.argmax(obj_pred_tag, 2) == 2)).sum().item() / max(1, (torch.argmax(obj_pred_tag, 2) == 2).sum().item()))
        file_logger.append('rec_obj_tag_1', ((obj_seq_tag == 1) & (torch.argmax(obj_pred_tag, 2) == 1)).sum().item() / max(1, (obj_seq_tag == 1).sum().item()))
        file_logger.append('rec_obj_tag_2', ((obj_seq_tag == 2) & (torch.argmax(obj_pred_tag, 2) == 2)).sum().item() / max(1, (obj_seq_tag == 2).sum().item()))

        # Stage 3 Loss
        matrix_mask = attention_mask.unsqueeze(-1) * attention_mask.unsqueeze(1)
        loss_corres = criterion_BCE_NoReduction(pred_corres_matrix.squeeze(3), corres_matrix.float())
        loss_corres = (loss_corres * matrix_mask).sum() / matrix_mask.sum()
        file_logger.append('rec_corres', ((torch.sigmoid(pred_corres_matrix).squeeze(-1) > model.lambda_2) & corres_matrix.bool()).sum().item() / corres_matrix.sum().item())
        file_logger.append('prec_corres', ((torch.sigmoid(pred_corres_matrix).squeeze(-1) > model.lambda_2) & corres_matrix.bool()).sum().item() / max(1, (
                torch.sigmoid(pred_corres_matrix) > model.lambda_2).sum().item()))

        loss = loss_rel + loss_tag + loss_corres
        loss.backward()
        model.optimizer.step()
        epoch_loss += loss.item()
        epoch_loss_rel += loss_rel.item()
        epoch_loss_tag += loss_tag.item()
        epoch_loss_corres += loss_corres.item()

        tepoch.set_postfix({'loss': epoch_loss / (bi + 1), 'loss_rel': epoch_loss_rel / (bi + 1), 'loss_tag': epoch_loss_tag / (bi + 1), 'loss_corres': epoch_loss_corres / (bi + 1)})

    file_logger.log()
    return epoch_loss / len(iterator)


def evaluate_epoch(model, iterator, device):
    model.eval()
    correct_num = 0;
    gold_num = 0;
    pred_num = 0
    with torch.no_grad():
        for batch in tqdm(iterator):
            ids = batch[0].to(device);
            attention_mask = batch[1].to(device)
            triples = batch[2]

            sentence_rel_map, subj_pred_tag, obj_pred_tag, pred_corres_matrix = model(ids, attention_mask, None, mode='eval')
            pred_triples = utils.GenerateTriples(sentence_rel_map, subj_pred_tag, obj_pred_tag, pred_corres_matrix, model.lambda_2)

            sent_correct_num, sent_predict_num, sent_gold_num = utils.FindMatches(triples, pred_triples)

            correct_num += sent_correct_num;
            gold_num += sent_gold_num;
            pred_num += sent_predict_num

    metrics = utils.get_metrics(correct_num, pred_num, gold_num)
    metrics_str = "; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics.items())
    print(metrics_str)
    return metrics


def get_arguments():
    parser = argparse.ArgumentParser(description="PRGC Model")
    parser.add_argument("-dataset", type=str, default='WebNLG', help="Dataset Choice out of {'NYT','NYT-star','WebNLG','WebNLG-star'}")
    parser.add_argument("-checkpoint", type=str, default='bert-base-uncased', help="chepoint for a pre-trained language model, from https://huggingface.co/models")
    parser.add_argument("-nepochs", type=int, default='100', help="number of training epochs")
    parser.add_argument("-batchsize", type=int, default='6', help="size of each batch")
    parser.add_argument("-lambda1", type=float, default='0.1', help="threshold for relation judgement, in [0,1]")
    parser.add_argument("-lambda2", type=float, default='0.5', help="threshold for global correspondence, in [0,1]")
    parser.add_argument("-gpuid", type=str, default='0', help="GPU id ")
    parser.add_argument("-seed", type=int, default='2021', help="RNG seed")
    parser.add_argument("-fusion", type=str, default='concat', help="Fusion type concat or sum")
    return parser.parse_args()


if __name__ == "__main__":

    rel_num = 216

    args = get_arguments()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuid

    torch.manual_seed(args.seed)

    # plm_tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    plm_tokenizer = BertTokenizer(vocab_file=os.path.join(os.path.dirname(__file__), 'pretrained_model', 'vocab.txt'),
                                  do_lower_case=False)
    configuration = BertConfig()
    pathtoModel = 'pretrained_model'
    plm_weights = AutoModel.from_pretrained(pathtoModel)
    max_seq_len = plm_tokenizer.max_model_input_sizes[args.checkpoint] - 2  # -2 for [CLS] and [SEP]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Read data from file
    train_text, train_labels = utils.ReadData(args.dataset, 'train')
    val_text, val_labels = utils.ReadData(args.dataset, 'val')
    rel2id = utils.ReadData(args.dataset, 'rel2id')

    train_dataset = TrainDataset(train_text, train_labels, rel2id, plm_tokenizer, max_seq_len)
    val_dataset = ValDataset(val_text, val_labels, rel2id, plm_tokenizer, max_seq_len)

    train_loader = DataLoader(train_dataset, args.batchsize, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, args.batchsize, shuffle=True, drop_last=True, collate_fn=utils.val_collate_fn)

    prgc_model = PRGC(plm_weights, rel_num, args.lambda1, args.lambda2, args.fusion).to(device)

    for epoch in range(args.nepochs):
        train_epoch_loss = train_epoch(prgc_model, train_loader, epoch, device)
        metrics = evaluate_epoch(prgc_model, val_loader, device)
    pause = 1
