import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import BertModel, BertConfig, BertTokenizer
import argparse
import utils
from tqdm.auto import tqdm
from optimization import BertAdam
from prgc import PRGC
from datautils import TrainDataset, ValDataset, ReadData
import os

def train_epoch(model, iterator, optimizer, gradient_accumulation_steps,  epoch, device):
    file_logger = utils.EpochMetricFileLogger('epoch_train.log')

    epoch_loss = 0; epoch_loss_rel = 0; epoch_loss_tag = 0; epoch_loss_corres = 0
    model.train()
    criterion_BCE = nn.BCEWithLogitsLoss()
    criterion_BCE_NoReduction = nn.BCEWithLogitsLoss(reduction='none')
    criterion_CE = nn.CrossEntropyLoss(reduction='none')
    tepoch = tqdm(iterator, desc=f'Epoch {epoch}', total=len(iterator))
    model.zero_grad()
    for bi, batch in enumerate(tepoch):
        ids = batch['input_ids'].to(device); attention_mask = batch['attention_mask'].to(device)
        subj_seq_tag = batch['subj_seq_tag'].to(device); obj_seq_tag = batch['obj_seq_tag'].to(device)
        target_rel = batch['target_rel'].to(device); relation_labels = batch['relation_labels'].to(device)
        corres_matrix = batch['corres_matrix'].to(device)
        
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
        

        if isinstance(optimizer,BertAdam):
            loss = loss / gradient_accumulation_steps
            loss.backward()
            epoch_loss += loss.item() * gradient_accumulation_steps
            if (bi + 1) % gradient_accumulation_steps == 0:
                # performs updates using calculated gradients
                optimizer.step()
                model.zero_grad()
        else:
            loss.backward()
            optimizer.step()
            model.zero_grad()
            epoch_loss += loss.item()
        
        epoch_loss_rel += loss_rel.item()
        epoch_loss_tag += loss_tag.item()
        epoch_loss_corres += loss_corres.item()

        tepoch.set_postfix({'loss': epoch_loss / (bi + 1),'loss_mat': epoch_loss_corres / (bi + 1), 'loss_rel': epoch_loss_rel / (bi + 1), 'loss_seq': epoch_loss_tag / (bi + 1)})

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
            pred_triples = utils.GenerateTriples(sentence_rel_map, subj_pred_tag, obj_pred_tag, pred_corres_matrix,attention_mask, model.lambda_2)

            sent_correct_num, sent_predict_num, sent_gold_num = utils.FindMatches(triples, pred_triples)

            correct_num += sent_correct_num
            gold_num += sent_gold_num
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
    parser.add_argument("-gpuid", type=str, default='3', help="GPU id ")
    parser.add_argument("-seed", type=int, default='2021', help="RNG seed")
    parser.add_argument("-fusion", type=str, default='concat', help="Fusion type concat or sum")
    return parser.parse_args()

if __name__ == "__main__":

    rel_num = 216

    args = get_arguments()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuid

    torch.manual_seed(args.seed)

    plm_tokenizer = BertTokenizer(vocab_file=os.path.join(os.path.dirname(__file__), 'pretrained_model', 'vocab.txt'),
                                  do_lower_case=False)
    configuration = BertConfig()
    plm_weights = BertModel.from_pretrained('pretrained_model')
    max_plm_seq_len = plm_tokenizer.max_model_input_sizes[args.checkpoint] - 2  # -2 for [CLS] and [SEP]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Read data from file
    train_text, train_labels = ReadData(args.dataset, 'train')
    val_text, val_labels = ReadData(args.dataset, 'val')
    rel2id = ReadData(args.dataset, 'rel2id')

    train_dataset = TrainDataset(train_text, train_labels, rel2id, plm_tokenizer, max_plm_seq_len)
    val_dataset = ValDataset(val_text, val_labels, rel2id, plm_tokenizer, train_dataset.seq_len)

    train_loader = DataLoader(train_dataset, args.batchsize, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, args.batchsize, shuffle=True, drop_last=True, collate_fn=utils.val_collate_fn)

    model = PRGC(plm_weights, rel_num, args.lambda1, args.lambda2, args.fusion).to(device)

    if True:
        # Adam Optimizer
        optimizer = optim.AdamW([
            {'params': model.bert.parameters(), 'lr': 1e-4},
            {'params': model.hidden_relational.parameters(), 'lr': 0.001},
            {'params': model.rel_judge.parameters(), 'lr': 0.001},
            {'params': model.subj_hidden_tag.parameters(), 'lr': 0.001},
            {'params': model.obj_hidden_tag.parameters(), 'lr': 0.001},
            {'params': model.tag_subject.parameters(), 'lr': 0.001},
            {'params': model.tag_object.parameters(), 'lr': 0.001},
            {'params': model.rel_embedding.parameters(), 'lr': 0.001},
            {'params': model.hidden_corres.parameters(), 'lr': 0.001},
            {'params': model.global_corr.parameters(), 'lr': 0.001},
        ], weight_decay=0.01
        )
        gradient_accumulation_steps = None # placeholder
    else:
        # Prepare optimizer
        params = {'weight_decay_rate': 0.01, 'fin_tuning_lr': 1e-4, 'downs_en_lr': 1e-3, 'clip_grad': 2., 'warmup_prop': 0.1, 'gradient_accumulation_steps': 2 }
        # fine-tuning
        param_optimizer = list(model.named_parameters())
        # pretrain model param
        param_pre = [(n, p) for n, p in param_optimizer if 'bert' in n]
        # downstream model param
        param_downstream = [(n, p) for n, p in param_optimizer if 'bert' not in n]
        no_decay = ['bias', 'LayerNorm', 'layer_norm']
        optimizer_grouped_parameters = [
            # pretrain model param
            {'params': [p for n, p in param_pre if not any(nd in n for nd in no_decay)],
            'weight_decay': params['weight_decay_rate'], 'lr': params['fin_tuning_lr']
            },
            {'params': [p for n, p in param_pre if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0, 'lr': params['fin_tuning_lr']
            },
            # downstream model
            {'params': [p for n, p in param_downstream if not any(nd in n for nd in no_decay)],
            'weight_decay': params['weight_decay_rate'], 'lr': params['downs_en_lr']
            },
            {'params': [p for n, p in param_downstream if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0, 'lr': params['downs_en_lr']
            }
        ]
        gradient_accumulation_steps = params['gradient_accumulation_steps']
        num_train_optimization_steps = len(train_loader) // gradient_accumulation_steps * args.nepochs
        optimizer = BertAdam(optimizer_grouped_parameters, warmup=params['warmup_prop'], schedule="warmup_cosine",
                            t_total=num_train_optimization_steps, max_grad_norm=params['clip_grad'])
    
    for epoch in range(args.nepochs):
        train_epoch_loss = train_epoch(model, train_loader, optimizer, gradient_accumulation_steps, epoch, device)
        metrics = evaluate_epoch(model, val_loader, device)
