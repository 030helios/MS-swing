# -*- coding: utf-8 -*-
'''
@ Author: HsinWei
@ Create Time: 2023-01-25 09:15:47
@ Description:
'''

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
import pandas as pd
import numpy as np
import datetime
import pickle
import os
import psycopg2
import math
import time
import matplotlib.pyplot as plt

plt.style.use("default")

# from tool import *

from transformers import BertTokenizer, TFBertModel
from bs4 import BeautifulSoup
import requests
import functools as ft
import yfinance as yf

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.linear_model import lasso_path, enet_path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, LassoCV, LassoLarsCV
from sklearn.model_selection import train_test_split

from itertools import cycle


# SET PARAMS
ROOT = "./import_csv/"
PIC = "./pic/"

NORMAL_COL = ["open", "close", "low", "high", "volume"]

# SET TIME
start_date = "2007-07-10"
end_date = "2024-08-01"

# SET GPU
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

class CFG:
    inference = True
    futureDay = 1
    output_dir = 'checkpoints'
    pretrained = True                    # 是否加载预训练模型权重, False为仅加载模型结构随机初始化权重
    freeze = True
    model_ptm = "checkpoints/bestmodel"
    # 训练参数
    device = 'cuda:0'
    epochs = 50
    learning_rate = 2e-4                 # 0.5e-4 for large; 2.5e-4 for base
    batch_size = 16
    accumulation_steps = 1               # 梯度累加
    apex = True                          # 是否使用混合精度加速
    seed = 42 
    # scheduler参数
    scheduler = 'cosine'                 # ['linear', 'cosine'] # lr scheduler 类型
    batch_scheduler = True               # 是否每个step结束后更新 lr scheduler
    weight_decay = 0.01
    num_warmup_steps = 3
    num_cycles = 0.5                     # 如果使用 cosine lr scheduler， 该参数决定学习率曲线的形状，0.5代表半个cosine曲线
