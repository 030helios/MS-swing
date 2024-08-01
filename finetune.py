import os
import pandas as pd
import numpy as np
import pickle
from torch.utils.data import DataLoader, Dataset
from finmodel import SimpleCLIP, FinClip
from import_tool import CFG

from pprint import pprint
from tqdm import tqdm
import torch
from torch.optim import Adam, SGD, AdamW
import transformers
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

#print(f"transformers.__version__: {transformers.__version__}")
transformers.logging.set_verbosity_error()

news_feature_path = "./Feature/"
ROOT = "./import_csv/"
SPLIT_DATE = "2021-08-01"

def labelApply(x):
    if x>0:
        return 1
    else:
        return 0

def label_gen_trend(df,futureDay):
    # could change to df["open"]
    percent = (df['close'].pct_change(futureDay)).shift(-futureDay)
    return percent.apply(labelApply)

def eval(model, valid_dataloader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        tk0 = tqdm(enumerate(valid_dataloader),total=len(valid_dataloader), desc="[Eval]")
        for step, batch in tk0:
            with torch.cuda.amp.autocast(enabled=CFG.apex):
                loss, output, ytrain, cated = model(batch['data'].to(device), batch['label'].to(device), outputLoss=True)
                total_loss += loss
            tk0.set_postfix(Loss=total_loss.item()/(step+1))

    return total_loss, output, ytrain, cated

def train_eval(model, train_dataloader, valid_dataloader, save_path):
    assert CFG.device.startswith('cuda') or CFG.device == 'cpu', ValueError("Invalid device.")
    device = torch.device(CFG.device)
    train_eval.best_score = 100000.0
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
    optimizer = torch.optim.SGD(optimizer_grouped_parameters, lr=CFG.learning_rate, momentum=0.9, nesterov=True)
    
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
                optimizer, num_warmup_steps=CFG.num_warmup_steps, num_training_steps=num_train_steps)
    
    optimizer.zero_grad()
    for cur_epc in range(int(CFG.epochs)):
        if(cur_epc==CFG.epochs/3):
            model.unfreeze()
        training_loss = 0
        model.train()
        tk0 = tqdm(enumerate(train_dataloader),total=len(train_dataloader), desc="Epoch: {}".format(cur_epc))
        for step, batch in tk0:
            if batch['data'].size(0) == 1:
                continue
            total_step += 1
            with torch.cuda.amp.autocast(enabled=CFG.apex):
                loss, _, _ , _ = model(batch['data'].to(device), batch['label'].to(device), outputLoss=True)
            scaler.scale(loss).backward()
            if (total_step) % CFG.accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                if CFG.batch_scheduler:
                    scheduler.step()
                optimizer.zero_grad()
            training_loss += loss.item()
            tk0.set_postfix(Epoch=cur_epc, Loss=training_loss/(step+1))
        evalloss,_,_, _ = eval(model,valid_dataloader)
        if evalloss/len(valid_dataloader)<=train_eval.best_score or cur_epc%10==9:
            model_save_path = os.path.join(save_path,f'{evalloss/len(valid_dataloader)}_{training_loss/len(train_dataloader)}_epoch{cur_epc}') # 保留所有checkpoint
            if os.path.exists(model_save_path) is False:
                os.mkdir(model_save_path)
            model.saveState(model_save_path)
            model.saveState('checkpoints/bestmodel')
            print(f'save at {model_save_path}')
        train_eval.best_score = min(train_eval.best_score,evalloss/len(valid_dataloader))
    torch.cuda.empty_cache()

def buildTrain(train):
    X_train, Y_train = [], []
    for i in range(train.shape[0]-9):
        tenDay = []
        for j in range(10):
            tenDay.extend(train.iloc[i+j,2:34])
            tenDay.extend(train.iloc[i+j,34:802])
            tenDay.extend(train.iloc[i+j,802:])
        label = train.iloc[i+9,1]
        Y_train.append(label)
        X_train.append(tenDay)
        # break
    return X_train, Y_train

class TrainDataset(Dataset):
    def __init__(self, train):
        self.X_train, self.Y_train = [], []
        for i in range(train.shape[0]-9):
            tenDay = []
            for j in range(10):
                tenDay.extend(train.iloc[i+j,2:])
            label = train.iloc[i+9,1]
            self.X_train.append(torch.FloatTensor(tenDay))
            self.Y_train.append(label.astype(np.float32))

    def __len__(self):
        return len(self.X_train)

    def __getitem__(self, idx):
        return {'data':self.X_train[idx], 'label':self.Y_train[idx]}

if __name__ == '__main__':
    np.random.seed(CFG.seed)  
    torch.manual_seed(CFG.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(CFG.seed)
        torch.cuda.manual_seed_all(CFG.seed) 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    twf = pd.read_csv(ROOT + "TWF_price.csv")
    twf['label'] = label_gen_trend(twf, CFG.futureDay)
    twf.to_csv("label.csv",index=0)

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
    data = pd.merge(data, twf)
    data.dropna(inplace=True)
    data = data[data["date"] > "2015-01-01"]

    # lasso col
    lasso_col = pd.read_csv(ROOT + 'coef.csv')
    lasso_col = list(lasso_col['name'])
    lasso_col = ["date", 'label','twf_log_rtn','twf_norm_o','twf_norm_h','twf_norm_l','twf_norm_c'] + lasso_col

    stock_data = data[lasso_col]
    merge_data = pd.merge(stock_data, news_en_data)
    merge_data = pd.merge(merge_data, news_data, on=['date'])
    merge_data = merge_data.drop(["summary_x", "summary_y"], axis=1)

    '''
    xten, yten = buildTrain(merge_data)
    with open('xten.pickle', 'wb') as handle:
        pickle.dump(xten, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('yten.pickle', 'wb') as handle:
        pickle.dump(yten, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('merge_data.pickle', 'wb') as handle:
        pickle.dump(merge_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    '''

    train_data = merge_data[merge_data["date"] < SPLIT_DATE]
    valid_data = merge_data[merge_data["date"] >= SPLIT_DATE]

    train_dataset = TrainDataset(train_data)
    train_dataloader = DataLoader(train_dataset, batch_size=CFG.batch_size, num_workers=5,shuffle=True)
    # 加载模型
    device = torch.device(CFG.device)
    clipModel = FinClip(CFG.model_ptm, device=device, pretrained=CFG.pretrained, freeze= CFG.freeze)
    clipModel = clipModel.to(device)

    if CFG.inference == False:
        valid_dataset = TrainDataset(valid_data)
        valid_dataloader = DataLoader(valid_dataset, batch_size=CFG.batch_size, num_workers=5,shuffle=False)
        # 训练
        train_eval(clipModel, train_dataloader,valid_dataloader, CFG.output_dir +'/finetune')
    else:
        predList  = []
        ytrainLIst  = []
        
        valid_dataset = TrainDataset(merge_data)
        valid_dataloader = DataLoader(valid_dataset, batch_size=CFG.batch_size, num_workers=5,shuffle=False)
        clipModel.eval()
        with open("cated4.txt", "w") as f:
            with torch.no_grad():
                tk0 = tqdm(enumerate(valid_dataloader),total=len(valid_dataloader), desc="[Dev]")
                for step, batch in tk0:
                    with torch.cuda.amp.autocast(enabled=CFG.apex):
                        loss, prediction, ytrain, cated = clipModel(batch['data'].to(device), batch['label'].to(device), outputLoss=True)
                        predList.extend(prediction)
                        ytrainLIst.extend(ytrain)
                        np.savetxt(f,torch.round(torch.flatten(cated,start_dim=1).cpu(),decimals=3), fmt='%+1.3f', delimiter='\t')

        y_pred = predList[0]
        for i in range(1,len(predList)):
            y_pred = torch.cat((y_pred,predList[i]))
        y_pred = y_pred.cpu()    
        #print(y_pred)

        ytrain = ytrainLIst[0]
        for i in range(1,len(ytrainLIst)):
            ytrain = torch.cat((ytrain,ytrainLIst[i]))
        ytrain = ytrain.cpu()    
        #print(ytrain)

        mc = pd.DataFrame(data)
        mc = mc.reset_index()
        drop_row = np.arange(mc.shape[0] - len(y_pred))
        mc = mc.drop(drop_row)
        mc['pred'] = y_pred*1000
        mc = mc[['date','pred']]
        twfprice = pd.read_csv(ROOT+'TWF_price.csv')
        to_mc = pd.merge(twfprice,mc,on='date',how='outer')
        to_mc['pred'] = to_mc['pred'].fillna(2000)
        #to_mc = to_mc[to_mc['date']>=SPLIT_DATE]
        to_mc.drop(['volume'],axis=1,inplace=True)
        to_mc.rename(columns={'pred':'volume'},inplace=True)
        to_mc[['open','high','low','close','volume']] = to_mc[['open','high','low','close','volume']].astype(int)
        csv_name = './pred.csv'
        to_mc.to_csv(csv_name,index=0)