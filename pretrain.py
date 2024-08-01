import os
import pandas as pd
import numpy as np
import pickle
from torch.utils.data import DataLoader, Dataset
from finmodel import SimpleCLIP, ModalitySpecific
from import_tool import CFG

from pprint import pprint
from tqdm import tqdm
import torch
from torch.optim import Adam, SGD, AdamW
from torch.utils.data import DataLoader, Dataset
import transformers
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

#print(f"transformers.__version__: {transformers.__version__}")
transformers.logging.set_verbosity_error()

news_feature_path = "./Feature/"
ROOT = "./import_csv/"
SPLIT_DATE = "2021-08-01"

class TrainDataset(Dataset):
    def __init__(self, merge_data):
        self.stock_df = merge_data.iloc[:,0:32]
        self.en_df = merge_data.iloc[:,32:800]
        self.ch_df = merge_data.iloc[:,800:]
        '''
        print(stock_data.iloc[0].values)
        with pd.ExcelWriter('output.xlsx') as writer:  
            stock_data.to_excel(writer, sheet_name='Sheet_name_1')
        with pd.ExcelWriter('output.xlsx') as writer:  
            news_data.to_excel(writer, sheet_name='Sheet_name_2')
        with pd.ExcelWriter('output.xlsx') as writer:  
            news_en_data.to_excel(writer, sheet_name='Sheet_name_3')
        '''

    def __len__(self):
        return len(self.stock_df)

    def __getitem__(self, idx):
        stockVector = self.stock_df.iloc[idx].values
        enVector = self.en_df.iloc[idx].values
        chVector = self.ch_df.iloc[idx].values
        return {'stock':stockVector, 'en':enVector, 'ch':chVector}

def eval(model, valid_dataloader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        tk0 = tqdm(enumerate(valid_dataloader),total=len(valid_dataloader), desc="[Dev]")
        for step, batch in tk0:
            for k,v in enumerate(batch['stock']):
                batch['stock'][k] = v
            for k,v in enumerate(batch['en']):
                batch['en'][k] = v
            for k,v in enumerate(batch['ch']):
                batch['ch'][k] = v
            with torch.cuda.amp.autocast(enabled=CFG.apex):
                loss, _, _, _, _ = model(batch['stock'].to(device), batch['en'].to(device), batch['ch'].to(device), outputLoss=True)
                total_loss += loss
            tk0.set_postfix(Loss=total_loss.item()/(step+1))
    return total_loss

def train_eval(model, train_dataloader, valid_dataloader, save_path):
    assert CFG.device.startswith('cuda') or CFG.device == 'cpu', ValueError("Invalid device.")
    device = torch.device(CFG.device)
    best_score = 100000
    total_step = 0
    model = model.to(device)
    scaler = torch.cuda.amp.GradScaler(enabled=CFG.apex)
    if not len(train_dataloader):
        raise EOFError("Empty train_dataloader.")

    # 过滤掉冻结的权重
    param_optimizer = [(n, p) for n, p in model.named_parameters() if p.requires_grad]

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    # 设置权重decay
    optimizer_grouped_parameters = [
        {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": CFG.weight_decay},
        {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0}]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=CFG.learning_rate, weight_decay=CFG.weight_decay)
    
    num_train_steps = int(len(train_dataloader) * CFG.epochs / CFG.accumulation_steps)
    if CFG.scheduler=='cosine':
        scheduler = get_cosine_schedule_with_warmup(
                    optimizer, 
                    num_warmup_steps=CFG.num_warmup_steps, 
                    num_training_steps=num_train_steps, 
                    num_cycles=CFG.num_cycles
                )
    else:
        scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=CFG.num_warmup_steps, num_training_steps=num_train_steps
            )
    
    for cur_epc in range(int(CFG.epochs)):
        training_loss = 0
        model.train()
        tk0 = tqdm(enumerate(train_dataloader),total=len(train_dataloader), desc="Epoch: {}".format(cur_epc))
        for step, batch in tk0:
            if batch['stock'].size(0) == 1:
                continue
            total_step += 1
            for k,v in enumerate(batch['stock']):
                batch['stock'][k] = v
            for k,v in enumerate(batch['en']):
                batch['en'][k] = v
            for k,v in enumerate(batch['ch']):
                batch['ch'][k] = v
            with torch.cuda.amp.autocast(enabled=CFG.apex):
                loss, _, _, _, _ = model(batch['stock'].to(device), batch['en'].to(device), batch['ch'].to(device), outputLoss=True)
            scaler.scale(loss).backward()
            if (total_step) % CFG.accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if CFG.batch_scheduler:
                    scheduler.step()
            training_loss += loss.item()
            tk0.set_postfix(Epoch=cur_epc, Loss=training_loss/(step+1))
        evalloss = eval(model,valid_dataloader)
        if evalloss/len(valid_dataloader)<=best_score:
            model_save_path = os.path.join(save_path+'/pretrain',f'{evalloss/len(valid_dataloader)}_{training_loss/len(train_dataloader)}_epoch{cur_epc}') # 保留所有checkpoint
            if os.path.exists(model_save_path) is False:
                os.mkdir(model_save_path)
            model.saveState(model_save_path)
            model.saveState('checkpoints/bestmodel')
            print(f'save at {model_save_path}')
        best_score = min(best_score,evalloss/len(valid_dataloader))
    torch.cuda.empty_cache()

if __name__ == '__main__':
    np.random.seed(CFG.seed)  
    torch.manual_seed(CFG.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(CFG.seed)
        torch.cuda.manual_seed_all(CFG.seed) 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    #load chinese news bert data
    with open(news_feature_path + "news_features.pkl", "rb") as f: 
        news_data = pickle.load(f)

    news_data = news_data[news_data["time"] > "01/01/2015"]
    # news_data = news_data.drop(["titlelist", "titlecount", "summlen"], axis=1)
    news_data = news_data.drop(["titlecount", "summlen"], axis=1)
    news_data.rename(columns={"time": "date"}, inplace=True)

    #load english news data
    with open(news_feature_path + "news_en_cased_features.pkl", "rb") as f:       # cased news
        news_en_data = pickle.load(f)
    news_en_data = news_en_data[news_en_data["time"] > "2015-01-01"]
    # news_en_data = news_en_data.drop(["titlelist", "titlecount", "summlen"], axis=1)
    news_en_data = news_en_data.drop(["titlecount", "summlen"], axis=1)
    news_en_data.rename(columns={"time": "date"}, inplace=True)
    # print(news_en_data)

    with open(ROOT + "price_pre.pickle", "rb") as f:
        data = pickle.load(f)
    data.dropna(inplace=True)
    data = data[data["date"] > "2015-01-01"]

    # lasso col
    lasso_col = pd.read_csv(ROOT + 'coef.csv')
    lasso_col = list(lasso_col['name'])
    lasso_col = ["date", 'twf_log_rtn','twf_norm_o','twf_norm_h','twf_norm_l','twf_norm_c'] + lasso_col

    stock_data = data[lasso_col]
    merge_data = pd.merge(stock_data, news_en_data)
    merge_data = pd.merge(merge_data, news_data, on=['date'])
    merge_data = merge_data.drop(["summary_x", "summary_y"], axis=1)

    train_data = merge_data[merge_data["date"] < SPLIT_DATE].drop(["date"], axis=1).astype('float32')
    valid_data = merge_data[merge_data["date"] >= SPLIT_DATE].drop(["date"], axis=1).astype('float32')
    data.to_csv('valid_data.csv')
    device = torch.device(CFG.device)
    train_dataset = TrainDataset(train_data)
    train_dataloader = DataLoader(train_dataset, batch_size=CFG.batch_size, num_workers=5,shuffle=True)
    valid_dataset = TrainDataset(valid_data)
    valid_dataloader = DataLoader(valid_dataset, batch_size=CFG.batch_size, num_workers=5,shuffle=True)
    # 加载模型
    clipModel = SimpleCLIP(CFG.model_ptm, device=device, pretrained=CFG.pretrained)

    # 训练
    train_eval(clipModel, train_dataloader,valid_dataloader, CFG.output_dir)