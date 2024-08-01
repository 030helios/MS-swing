#!/usr/bin/env python
# coding: utf-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from import_tool import *
from torch import nn, Tensor
from import_tool import CFG

def stripPrefix(ptm_name, prefix = 'model.'):
    state_dict = torch.load(ptm_name)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if prefix in k:
            name = k[len(prefix):] # remove module.
            new_state_dict[name] = v
    return new_state_dict

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        #m.bias.data.fill_(0.1)
        nn.init.constant_(m.bias, 0.01)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)/10
        pe[0, :, 1::2] = torch.cos(position * div_term)/10
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.pe[:,:x.size(1)]
        return self.dropout(x)


#32->32
class SharedEncoder(nn.Module):
    def __init__(self, ptm_name, pretrained, d_model=32):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(32,32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Linear(32,32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Linear(32,32),
            nn.LayerNorm(32)
        )
        if pretrained:
            try:
                self.model.load_state_dict(stripPrefix(ptm_name,'model.'))
            except:
                print(f"{ptm_name} not found")
                self.model.apply(init_weights)
                pass
        else:
            self.model.apply(init_weights)

    def forward(self, inputs):
        feature = self.model(inputs)
        feature = F.normalize(feature, dim=-1)
        return feature
#768->32
class NewsEncoder(nn.Module):
    def __init__(self, ptm_name, pretrained, d_model=32):
        super().__init__()        
        self.model = nn.Sequential(
        nn.Linear(768,384),
        nn.BatchNorm1d(384),
        nn.LeakyReLU(),
        nn.Linear(384,192),
        nn.BatchNorm1d(192),
        nn.LeakyReLU(),
        nn.Linear(192,32),
        nn.LayerNorm(32)
        )
        if pretrained:
            try:
                self.model.load_state_dict(stripPrefix(ptm_name,'model.'))
            except:
                print(f"{ptm_name} not found")
                self.model.apply(init_weights)
                pass
        else:
            self.model.apply(init_weights)

    def forward(self, inputs):
        feature = self.model(inputs)
        feature = F.normalize(feature, dim=-1)
        return feature

#10 days
#960->1
class Transformer960(nn.Module):
    def __init__(self, ptm_name,pretrained):
        super().__init__()
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=32, nhead=8, activation=F.leaky_relu, batch_first=True)
        for name, param in encoder_layer.named_parameters():
            if 'weight' in name and param.data.dim() == 2:
                nn.init.kaiming_uniform_(param)
        
        self.model = nn.Sequential(
        nn.TransformerEncoder(encoder_layer, num_layers=8),
        nn.Flatten(),
        nn.Linear(960,480),
        nn.BatchNorm1d(480),
        nn.LeakyReLU(),
        nn.Linear(480,240),
        nn.BatchNorm1d(240),
        nn.LeakyReLU(),
        nn.Linear(240,1)
        #nn.Sigmoid()
        )
        
        self.pos_encoder = PositionalEncoding(32)
        if pretrained:
            try:
                self.model.load_state_dict(stripPrefix(ptm_name,'model.'))
                self.pos_encoder.load_state_dict(stripPrefix(ptm_name,'pos_encoder.'))
            except:
                print(f"{ptm_name} not found")
                self.model.apply(init_weights)
                pass
        else:
            self.model.apply(init_weights)
    def forward(self, inputs):
        inputs = self.pos_encoder(inputs)
        return self.model(inputs)
    
class SimpleCLIP(nn.Module):
    def __init__(self, ptm, device, pretrained):
        super().__init__()
        self.device = device
        self.stockencoder = SharedEncoder(ptm+'/stock', pretrained=pretrained)
        self.en_newsEncoder = NewsEncoder(ptm+'/en', pretrained=pretrained)
        self.ch_newsEncoder = NewsEncoder(ptm+'/ch', pretrained=pretrained)
        self.sharedLayer = SharedEncoder(ptm+'/shared', pretrained=pretrained)

        self.logit_scale = nn.Parameter(torch.ones([]))
        self.init_parameters()
    
    def init_parameters(self):
        nn.init.constant_(self.logit_scale, np.log(1 / 0.07))

    def loss(self, stock_feat, en_news_feat, ch_news_feat, logit_scale):
        labels = torch.arange(en_news_feat.shape[0], device=self.device, dtype=torch.long)
        logits_per_stock = logit_scale * stock_feat @ en_news_feat.T   # [batch_size, batch_size]
        logits_per_en_text = logit_scale * en_news_feat @ ch_news_feat.T   # [batch_size, batch_size]
        logits_per_ch_text = logit_scale * ch_news_feat @ stock_feat.T   # [batch_size, batch_size]

        total_loss = (
            F.cross_entropy(logits_per_stock, labels) +
            F.cross_entropy(logits_per_stock.T, labels) +
            F.cross_entropy(logits_per_en_text, labels)+
            F.cross_entropy(logits_per_en_text.T, labels)+
            F.cross_entropy(logits_per_ch_text, labels)+
            F.cross_entropy(logits_per_ch_text.T, labels)
            ) / 6
        return total_loss

    def forward(self,  stock_inputs, en_news_inputs, ch_news_inputs,outputLoss=False):
        #32 -> 32
        stock_feat1 = self.stockencoder(stock_inputs)# @ self.stock_projection # [batch_size, dim]
        #768->32
        en_news_feat1 = self.en_newsEncoder(en_news_inputs)
        #768->32
        ch_news_feat1 = self.ch_newsEncoder(ch_news_inputs)

        stock_feat = self.sharedLayer(stock_feat1)
        en_news_feat = self.sharedLayer(en_news_feat1)
        ch_news_feat = self.sharedLayer(ch_news_feat1)
        #clip

        logit_scale = self.logit_scale.exp()
        if outputLoss:
            loss = self.loss(stock_feat, en_news_feat, ch_news_feat,logit_scale)
            return loss, stock_feat, en_news_feat, ch_news_feat, logit_scale
        else:
            return stock_feat, en_news_feat, ch_news_feat, logit_scale
        
    def saveState(self, model_save_path):
        torch.save(self.stockencoder.state_dict(), model_save_path+'\\stock')
        torch.save(self.en_newsEncoder.state_dict(), model_save_path+'\\en')
        torch.save(self.ch_newsEncoder.state_dict(), model_save_path+'\\ch')
        torch.save(self.sharedLayer.state_dict(), model_save_path+'\\shared')

class SpecificUpper(nn.Module):
    def __init__(self, ptm_name, pretrained, d_model=32):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(32,32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU()
        )
        if pretrained:
            try:
                self.model.load_state_dict(stripPrefix(ptm_name,'model.'))
            except:
                print(f"{ptm_name} not found")
                self.model.apply(init_weights)
                pass
        else:
            self.model.apply(init_weights)

    def forward(self, inputs):
        feature = self.model(inputs)
        feature = F.normalize(feature, dim=-1)
        return feature

class SpecificLower(nn.Module):
    def __init__(self, ptm_name, pretrained, d_model=32):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(32,32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Linear(32,32),
            nn.LayerNorm(32)
        )
        if pretrained:
            try:
                self.model.load_state_dict(stripPrefix(ptm_name,'model.'))
            except:
                print(f"{ptm_name} not found")
                self.model.apply(init_weights)
                pass
        else:
            self.model.apply(init_weights)

    def forward(self, inputs):
        feature = self.model(inputs)
        feature = F.normalize(feature, dim=-1)
        return feature

class ModalitySpecific(nn.Module):
    def __init__(self, ptm, device, pretrained):
        super().__init__()
        self.device = device
        self.stockencoder = SharedEncoder(ptm+'/stock', pretrained=pretrained)
        self.en_newsEncoder = NewsEncoder(ptm+'/en', pretrained=pretrained)
        self.ch_newsEncoder = NewsEncoder(ptm+'/ch', pretrained=pretrained)

        self.stockLower = SharedEncoder('null', pretrained=pretrained)
        self.en_newsLower = SharedEncoder('null', pretrained=pretrained)
        self.ch_newsLower = SharedEncoder('null', pretrained=pretrained)

        self.logit_scale = nn.Parameter(torch.ones([]))
        self.init_parameters()
    
    def init_parameters(self):
        nn.init.constant_(self.logit_scale, np.log(1 / 0.07))

    def loss(self, stock_feat, en_news_feat, ch_news_feat, logit_scale):
        labels = torch.arange(en_news_feat.shape[0], device=self.device, dtype=torch.long)
        logits_per_stock = logit_scale * stock_feat @ en_news_feat.T   # [batch_size, batch_size]
        logits_per_en_text = logit_scale * en_news_feat @ ch_news_feat.T   # [batch_size, batch_size]
        logits_per_ch_text = logit_scale * ch_news_feat @ stock_feat.T   # [batch_size, batch_size]

        total_loss = (
            F.cross_entropy(logits_per_stock, labels) +
            F.cross_entropy(logits_per_stock.T, labels) +
            F.cross_entropy(logits_per_en_text, labels)+
            F.cross_entropy(logits_per_en_text.T, labels)+
            F.cross_entropy(logits_per_ch_text, labels)+
            F.cross_entropy(logits_per_ch_text.T, labels)
            ) / 6
        return total_loss

    def forward(self,  stock_inputs, en_news_inputs, ch_news_inputs,outputLoss=False):
        #32 -> 32
        stock_feat1 = self.stockencoder(stock_inputs)# @ self.stock_projection # [batch_size, dim]
        #768->32
        en_news_feat1 = self.en_newsEncoder(en_news_inputs)
        #768->32
        ch_news_feat1 = self.ch_newsEncoder(ch_news_inputs)
        
        stock_feat = self.stockLower(stock_feat1)
        en_news_feat = self.en_newsLower(en_news_feat1)
        ch_news_feat = self.ch_newsLower(ch_news_feat1)
        #clip

        logit_scale = self.logit_scale.exp()
        if outputLoss:
            loss = self.loss(stock_feat, en_news_feat, ch_news_feat,logit_scale)
            return loss, stock_feat, en_news_feat, ch_news_feat, logit_scale
        else:
            return stock_feat, en_news_feat, ch_news_feat, logit_scale
        
    def saveState(self, model_save_path):
        torch.save(self.stockencoder.state_dict(), model_save_path+'\\stock')
        torch.save(self.en_newsEncoder.state_dict(), model_save_path+'\\en')
        torch.save(self.ch_newsEncoder.state_dict(), model_save_path+'\\ch')
        #torch.save(self.sharedLayer.state_dict(), model_save_path+'\\shared')

class SemiModalitySpecific(nn.Module):
    def __init__(self, ptm, device, pretrained):
        super().__init__()
        self.device = device
        self.stockencoder = SharedEncoder(ptm+'/stock', pretrained=pretrained)
        self.en_newsEncoder = NewsEncoder(ptm+'/en', pretrained=pretrained)
        self.ch_newsEncoder = NewsEncoder(ptm+'/ch', pretrained=pretrained)
        self.sharedLayer = SpecificUpper('null', pretrained=pretrained)

        self.stockLower = SpecificLower('null', pretrained=pretrained)
        self.en_newsLower = SpecificLower('null', pretrained=pretrained)
        self.ch_newsLower = SpecificLower('null', pretrained=pretrained)

        self.logit_scale = nn.Parameter(torch.ones([]))
        self.init_parameters()
    
    def init_parameters(self):
        nn.init.constant_(self.logit_scale, np.log(1 / 0.07))

    def loss(self, stock_feat, en_news_feat, ch_news_feat, logit_scale):
        labels = torch.arange(en_news_feat.shape[0], device=self.device, dtype=torch.long)
        logits_per_stock = logit_scale * stock_feat @ en_news_feat.T   # [batch_size, batch_size]
        logits_per_en_text = logit_scale * en_news_feat @ ch_news_feat.T   # [batch_size, batch_size]
        logits_per_ch_text = logit_scale * ch_news_feat @ stock_feat.T   # [batch_size, batch_size]

        total_loss = (
            F.cross_entropy(logits_per_stock, labels) +
            F.cross_entropy(logits_per_stock.T, labels) +
            F.cross_entropy(logits_per_en_text, labels)+
            F.cross_entropy(logits_per_en_text.T, labels)+
            F.cross_entropy(logits_per_ch_text, labels)+
            F.cross_entropy(logits_per_ch_text.T, labels)
            ) / 6
        return total_loss

    def forward(self,  stock_inputs, en_news_inputs, ch_news_inputs,outputLoss=False):
        #32 -> 32
        stock_feat1 = self.stockencoder(stock_inputs)# @ self.stock_projection # [batch_size, dim]
        #768->32
        en_news_feat1 = self.en_newsEncoder(en_news_inputs)
        #768->32
        ch_news_feat1 = self.ch_newsEncoder(ch_news_inputs)

        stock_feat2 = self.sharedLayer(stock_feat1)
        en_news_feat2 = self.sharedLayer(en_news_feat1)
        ch_news_feat2 = self.sharedLayer(ch_news_feat1)

        
        stock_feat = self.stockLower(stock_feat2)
        en_news_feat = self.en_newsLower(en_news_feat2)
        ch_news_feat = self.ch_newsLower(ch_news_feat2)
        #clip

        logit_scale = self.logit_scale.exp()
        if outputLoss:
            loss = self.loss(stock_feat, en_news_feat, ch_news_feat,logit_scale)
            return loss, stock_feat, en_news_feat, ch_news_feat, logit_scale
        else:
            return stock_feat, en_news_feat, ch_news_feat, logit_scale
        
    def saveState(self, model_save_path):
        torch.save(self.stockencoder.state_dict(), model_save_path+'\\stock')
        torch.save(self.en_newsEncoder.state_dict(), model_save_path+'\\en')
        torch.save(self.ch_newsEncoder.state_dict(), model_save_path+'\\ch')
        #torch.save(self.sharedLayer.state_dict(), model_save_path+'\\shared')

class FinClip(nn.Module):
    def __init__(self, ptm, device, pretrained, freeze=True):
        super().__init__()
        self.device = device
        self.stockencoder = SharedEncoder(ptm+'/stock', pretrained=pretrained)
        self.en_newsEncoder = NewsEncoder(ptm+'/en', pretrained=pretrained)
        self.ch_newsEncoder = NewsEncoder(ptm+'/ch', pretrained=pretrained)
        self.sharedLayer = SharedEncoder(ptm+'/shared', pretrained=pretrained)
        self.transformer32 = Transformer960(ptm+'/t960', pretrained=pretrained)
        if(freeze):
            self.stockencoder.requires_grad_(False)
            self.en_newsEncoder.requires_grad_(False)
            self.ch_newsEncoder.requires_grad_(False)
        self.logit_scale = nn.Parameter(torch.ones([]))
        self.init_parameters()
    
    def unfreeze(self):
        self.stockencoder.requires_grad_(True)
        self.en_newsEncoder.requires_grad_(True)
        self.ch_newsEncoder.requires_grad_(True)

    def init_parameters(self):
        nn.init.constant_(self.logit_scale, np.log(1 / 0.07))

    def forward(self, xtrain, ytrain=None ,outputLoss=False):
        section = [32,768,768,32,768,768,32,768,768,32,768,768,32,768,768,32,768,768,32,768,768,32,768,768,32,768,768,32,768,768]
        tendays = torch.split(xtrain,section,1)
        features = []
        for i in range(10):
            features.append(self.stockencoder(tendays[i*3]))
            features.append(self.en_newsEncoder(tendays[i*3+1]))
            features.append(self.ch_newsEncoder(tendays[i*3+2]))

        cated = torch.stack(features,dim=1)

        output = self.transformer32(cated)

        if outputLoss:
            bce = nn.BCEWithLogitsLoss()
            ytrain = ytrain.unsqueeze(1).to(torch.float32)
            return bce(output,ytrain), torch.sigmoid(output), ytrain, cated
        else:
            return output
        
    def saveState(self, model_save_path):
        torch.save(self.stockencoder.state_dict(), model_save_path+'\\stock')
        torch.save(self.en_newsEncoder.state_dict(), model_save_path+'\\en')
        torch.save(self.ch_newsEncoder.state_dict(), model_save_path+'\\ch')
        torch.save(self.sharedLayer.state_dict(), model_save_path+'\\shared')
        torch.save(self.transformer32.state_dict(), model_save_path+'\\t960')
        


class dummy(nn.Module):
    def __init__(self, ptm, device, pretrained, freeze=True):
        super().__init__()
        self.device = device
        self.model = nn.Sequential(
            nn.Linear(1568,784),
            nn.LeakyReLU(),
            nn.Linear(784,392),
            nn.LeakyReLU(),
            nn.Linear(392,1)
        )
        if pretrained:
            self.model.load_state_dict(torch.load(ptm+"\\model"))

    def unfreeze(self):
        self.model.requires_grad_(True)

    def forward(self, xtrain):
        output = self.model(xtrain)
        return output
        
    def saveState(self, model_save_path):
        torch.save(self.model.state_dict(), model_save_path+'\\model')
        

'''
#372->186
class RNNLayer(nn.Module):
    def __init__(self, ptm_name,pretrained):
        super().__init__()
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=372, nhead=8)
        encoder_layer2 = nn.TransformerEncoderLayer(d_model=186, nhead=6)

        self.model = nn.Sequential(nn.TransformerEncoder(encoder_layer, num_layers=4),
                                    nn.Linear(372,186),nn.ReLU(),nn.BatchNorm1d(186),
                                    nn.TransformerEncoder(encoder_layer2, num_layers=4))

        if pretrained:
            try:
                self.model.load_state_dict(stripPrefix(ptm_name))
            except:
                print(f"{ptm_name} not found")
                pass
    def forward(self, inputs):
        feature = self.model(inputs)
        feature = F.normalize(feature, dim=-1)
        return feature
    
#186->1
class Classifier(nn.Module):
    def __init__(self, ptm_name, pretrained):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
        nn.Linear(186,93),nn.ReLU(),nn.BatchNorm1d(93),
        nn.Linear(93,48),nn.ReLU(),nn.BatchNorm1d(48),
        nn.Linear(48,1), nn.Sigmoid())
        if pretrained:
            try:
                 self.model.load_state_dict(stripPrefix(ptm_name))
            except:
                print(f"{ptm_name} not found")
                pass

    def forward(self, inputs):
        feature = self.linear_relu_stack(inputs)
        return feature
    
class FinClip(nn.Module):
    def __init__(self, dim, stock_ptm, en_news_ptm, ch_news_ptm, shared_ptm, forward_ptm, classifier_ptm, device, pretrained):
        super().__init__()
        self.device = device
        self.stockencoder = SharedEncoder(stock_ptm, pretrained=pretrained)
        self.en_newsEncoder = NewsEncoder(en_news_ptm, pretrained=pretrained)
        self.ch_newsEncoder = NewsEncoder(ch_news_ptm, pretrained=pretrained)
        self.sharedLayer = SharedEncoder(shared_ptm, pretrained=pretrained)
        self.forwardLayer = RNNLayer(forward_ptm, pretrained=pretrained)
        self.classifier = Classifier(classifier_ptm, pretrained=pretrained)

        self.logit_scale = nn.Parameter(torch.ones([]))
        self.init_parameters()
    
    def init_parameters(self):
        nn.init.constant_(self.logit_scale, np.log(1 / 0.07))

    def loss(self, stock_feat, en_news_feat, ch_news_feat, logit_scale):
        labels = torch.arange(en_news_feat.shape[0], device=self.device, dtype=torch.long)
        logits_per_stock = logit_scale * stock_feat @ en_news_feat.T   # [batch_size, batch_size]
        logits_per_en_text = logit_scale * en_news_feat @ ch_news_feat.T   # [batch_size, batch_size]
        logits_per_ch_text = logit_scale * ch_news_feat @ stock_feat.T   # [batch_size, batch_size]

        total_loss = (
            F.cross_entropy(logits_per_stock, labels) +
            F.cross_entropy(logits_per_stock.T, labels) +
            F.cross_entropy(logits_per_en_text, labels)+
            F.cross_entropy(logits_per_en_text.T, labels)+
            F.cross_entropy(logits_per_ch_text, labels)+
            F.cross_entropy(logits_per_ch_text.T, labels)
            ) / 6
        return total_loss

    def forward(self,  stock_inputs, en_news_inputs, ch_news_inputs,outputLoss=False):
        stock_feat1 = self.stockencoder(stock_inputs)# @ self.stock_projection # [batch_size, dim]
        #32 -> 32
        en_news_feat1 = self.en_newsEncoder(en_news_inputs)
        #768->32
        ch_news_feat1 = self.ch_newsEncoder(ch_news_inputs)
        #768->32

        stock_feat = self.sharedLayer(stock_feat1)
        en_news_feat = self.sharedLayer(en_news_feat1)
        ch_news_feat = self.sharedLayer(ch_news_feat1)
        #clip

        logit_scale = self.logit_scale.exp()
        if outputLoss:
            loss = self.loss(stock_feat, en_news_feat, ch_news_feat,logit_scale)
            return loss, stock_feat, en_news_feat, ch_news_feat, logit_scale
        else:
            return stock_feat, en_news_feat, ch_news_feat, logit_scale
        
    def saveState(self, model_save_path):
        torch.save(self.stockencoder.state_dict(), model_save_path+'\\stock')
        torch.save(self.en_newsEncoder.state_dict(), model_save_path+'\\en')
        torch.save(self.ch_newsEncoder.state_dict(), model_save_path+'\\ch')
        torch.save(self.sharedLayer.state_dict(), model_save_path+'\\shared')
'''