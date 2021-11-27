from os import truncate
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer
import argparse
import utils

class myDataset(Dataset):
    def __init__(self,text,labels,rel2id,tokenizer,max_seq_len):             
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
        self.labels = utils.GetAllLabels(rel2id,labels,self.encoded_text['input_ids'],self.encoded_text['attention_mask'],self.tokenizer,self.seq_len)
   
    def __len__(self):
        return len(self.text)
    
    def __getitem__(self,idx):
        return {
            'input_ids': self.labels[idx]['input_ids'],
            'attention_mask': self.labels[idx]['attention_mask'],
            'subj_seq_tag': self.labels[idx]['subj_seq_tag'],
            'obj_seq_tag': self.labels[idx]['obj_seq_tag'],
            'target_rel': self.labels[idx]['target_rel'],
            'corres_matrix': self.labels[idx]['corres_matrix'],
            'relation_labels': self.labels[idx]['relation_labels']
        }

class PRGC(nn.Module):
    def __init__(self,plm_model,rel_num,lambda_1,lambda_2):
        super().__init__()
        
        # Bert Embedding 
        self.plm_model = plm_model
        hidden_dim = plm_model.config.hidden_size
        
        # Relation Trainable Embedding Matrix
        self.rel_embedding = nn.Embedding(num_embeddings=rel_num,embedding_dim=hidden_dim)

        # Stage 1
        self.rel_judge = nn.Linear(hidden_dim,rel_num)
        self.lambda_1 = lambda_1
        # Stage 2
        self.tag_subject = nn.Linear(hidden_dim,3)
        self.tag_object = nn.Linear(hidden_dim,3)
        # Stage 3
        self.global_corr = nn.Linear(2*hidden_dim,1)
        self.lambda_2 = lambda_2
    
    def forward(self,input_ids,attention_mask,subj_seq_tag,obj_seq_tag,target_rel,corres_matrix,relation_labels):
        # Bert Embedding
        with torch.no_grad():
            embedded = self.plm_model(input_ids=input_ids,attention_mask=attention_mask)['last_hidden_state']
        # Stage 1
        embedded_masked = embedded * attention_mask[:,:,None]
        num_nonmasked = torch.sum(attention_mask,dim=1)
        stage1_avgpool = torch.sum(embedded_masked,dim=1) / num_nonmasked[:,None]
        stage1 = torch.sigmoid(self.rel_judge(stage1_avgpool))
        pot_rel_mask = stage1 >= self.lambda_1

        # Stage 2


        # Stage 3

        return True

def train_epoch(model,iterator,optimizer,criterion,device):
    epoch_loss = 0
    correct_preds = 0
    model.train()
    for idx, batch in enumerate(iterator):
        print(f'\n\n {idx}')
        optimizer.zero_grad()
        ids = batch['input_ids'].to(device); 
        attention_mask = batch['attention_mask'].to(device)
        subj_seq_tag = batch['subj_seq_tag'].to(device)
        obj_seq_tag = batch['obj_seq_tag'].to(device)
        target_rel = batch['target_rel'].to(device)
        corres_matrix = batch['corres_matrix'].to(device)
        relation_labels = batch['relation_labels'].to(device)

        logits = model(ids,attention_mask,subj_seq_tag,obj_seq_tag,target_rel,corres_matrix,relation_labels)
        loss = criterion(logits,relation_labels)
        loss.backward()
        optimizer.step()
        epoch_loss+=loss.item()
        correct_preds+=torch.sum(torch.argmax(logits,dim=1)==true_labels)
    return epoch_loss/len(iterator), correct_preds/(len(iterator)*iterator.batch_size)

def evaluate_epoch(model,iterator,criterion,device):
    epoch_loss = 0
    correct_preds = 0
    model.eval()
    with torch.no_grad():
        for batch in iterator:
            ids = batch['input_ids'].to(device); true_labels = batch['labels'].to(device); attention_mask = batch['attention_mask'].to(device)
            logits = model(ids,attention_mask)
            loss = criterion(logits,true_labels)
            epoch_loss+=loss.item()
            correct_preds+=torch.sum(torch.argmax(logits,dim=1)==true_labels)
    return epoch_loss/len(iterator), correct_preds/(len(iterator)*iterator.batch_size)

def get_arguments():
    parser = argparse.ArgumentParser(description="PRGC Model")
    parser.add_argument("-dataset",type=str, help="Dataset Choice out of {'NYT','NYT-star','WebNLG','WebNLG-star'}")
    parser.add_argument("-checkpoint",type=str, help="chepoint for a pre-trained language model, from https://huggingface.co/models")
    parser.add_argument("-nepochs", type=int, help="number of training epochs")
    parser.add_argument("-batchsize", type=int, help="size of each batch")
    parser.add_argument("-lambda1", type=float, help="threshold for relation judgement, in [0,1]")
    parser.add_argument("-lambda2", type=float, help="threshold for global correspondence, in [0,1]") 
    parser.add_argument("-seed", type=int, help="RNG seed")
    return parser.parse_args()

if __name__ == "__main__":
    
    rel_num = 216

    args = get_arguments()

    torch.manual_seed(args.seed)

    plm_tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    plm_weights = AutoModel.from_pretrained(args.checkpoint)
    max_seq_len = plm_tokenizer.max_model_input_sizes[args.checkpoint]-2 # -2 for [CLS] and [SEP]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Read data from file
    train_text,train_labels = utils.ReadData(args.dataset,'train')
    val_text,val_labels = utils.ReadData(args.dataset,'val')
    rel2id = utils.ReadData(args.dataset,'rel2id')

    train_dataset = myDataset(train_text,train_labels,rel2id,plm_tokenizer,max_seq_len)
    val_dataset = myDataset(val_text,val_labels,rel2id,plm_tokenizer,max_seq_len)

    train_loader = DataLoader(train_dataset,args.batchsize,shuffle=True,drop_last=True)
    val_loader = DataLoader(val_dataset,args.batchsize,shuffle=True,drop_last=True)

    
    prgc_model = PRGC(plm_weights,rel_num,args.lambda1,args.lambda2).to(device)
    optimizer = optim.AdamW(prgc_model.parameters())
    criterion = nn.CrossEntropyLoss().to(device)

    for epoch in range(args.nepochs):
        train_epoch_loss, train_epoch_acc = train_epoch(prgc_model,train_loader,optimizer,criterion,device)
        valid_epoch_loss, valid_epoch_acc = evaluate_epoch(prgc_model,val_loader,criterion,device)
        print(f'Epoch: {epoch+1}\nTrain: Loss: {train_epoch_loss}, Accuracy: {train_epoch_acc} \
                                \nValid: Loss: {valid_epoch_loss}, Accuracy: {valid_epoch_acc}')
    pause=1

