# MS-swing


python .\get_data.py; python .\get_news_data_ch.py; python .\get_news_data_en.py; python .\data_pre.py; python .\news_data_pre.py;
python .\pretrain.py;python .\finetune.py
## Introduction
0. [Update date](#import_csv/txf_settlement.csv) update the settlement dates
1. [Update date](#import_tool.py) change the end_date to current date 
2. [Download](#get_data.py) the latest data from stock price        
3. [Download](#get_news_data_ch.py) the latest Chinese news data
4. [Download](#get_news_data_en.py) the latest English news data
5. Data [preprocess](#data_pre.py)  preprocess the data. Includes candlestick, expiration date, ...
6. News Data [preprocess](#news_data_pre.py)  preprocess the News data. Includes aggregate news data, clean news data, finally do the BERT extractation... (could take a while due to BERT features extraction)
7. Get [result](#rebuild_new_data.py)   (with our pre-trained TBM model in `/result/`)
8. Put the output file in `/pred/` to [multichart](#Multichart) to backtest the strategy. 
    ([Multichart] each optimized perameters can be found inside of every `/result/{modelname}` file)

