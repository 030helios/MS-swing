# -*- coding: utf-8 -*-
'''
@ Author: HsinWei
@ Create Time: 2023-01-10 12:47:52
@ Description:
'''
import pandas as pd
import os
import random
import numpy as np
from datetime import date, timedelta
from import_tool import *
from bert_extraction import extractBertFeatures
from en_bert_extraction import extractCasedBertFeatures


Textfilepath = "./News_data/"
writepath = './import_csv/'
MaxBertInputSize = 510


def AggreTitle_ch(sourcepath, newsfilename):
    Minstrlimit = 5     # title length limit
    testNews = pd.read_excel(sourcepath + newsfilename)
    PretestNews = testNews[testNews["title"].str.len() >= Minstrlimit]
    titlelistdf = (
        PretestNews.groupby("time")["title"].apply(list).reset_index(name="titlelist")
    )
    titlelistdf["titlecount"] = titlelistdf["titlelist"].str.len()
    return titlelistdf


def randomconcate(list, size):
    list = [i.replace("'", "") for i in list]
    try:
        sample = random.sample(list, size)
    except:
        result_string = ",".join(map(str, list))
        return result_string
    result_string = ",".join(map(str, sample))
    while len(result_string) > MaxBertInputSize:
        sample = random.sample(list, size)
        result_string = ",".join(map(str, sample))
    return result_string

def ch_news_pre():
    Maxtitlenums = 38  # maximum number of news headline per day
    start_year, end_year = 2015, 2025
    need_years = [f"news_{i}.xlsx" for i in range(start_year, end_year)]
    need_year_spd = [
        AggreTitle_ch(Textfilepath, i) for i in os.listdir(Textfilepath) if i in need_years
    ]
    news_df = pd.concat(need_year_spd, ignore_index=True)
    news_df["summary"] = news_df.titlelist.apply(lambda x: randomconcate(x, Maxtitlenums))
    news_df["summlen"] = news_df.summary.str.len()

    news_df = news_df.drop(['titlelist'], axis=1)
    df = pd.DataFrame(pd.date_range(start='2015-01-01', end=end_date), columns=['time'])
    news_df["time"] = pd.to_datetime(news_df["time"])
    df["time"] = pd.to_datetime(df["time"])

    news_df = pd.merge(news_df, df, on='time', how='outer')
    news_df["summary"] = news_df["summary"].apply(lambda x: 'x' if type(x) == type(0.0) else x)


    news_df = news_df.sort_values(by="time")
    news_df.to_csv(writepath + "News_data.csv", encoding="utf_8_sig", index=False)
    print("ch new finished")

def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)


def AggreTitleEn(sourcepath, newsfilename):
    Minstrlimit = 5  # title length limit
    testNews = pd.read_csv(sourcepath + newsfilename, encoding= 'UTF-8')
    testNews = testNews.rename(columns={'Date': 'time', 'Title': 'title'})
    PretestNews = testNews[testNews["title"].str.len() >= Minstrlimit]
    titlelistdf = (
        PretestNews.groupby("time")["title"].apply(list).reset_index(name="titlelist")
    )
    
    titlelistdf["titlecount"] = titlelistdf["titlelist"].str.len()
    

    return titlelistdf

def en_news_pre():
    Maxtitlenums = 5
    news_df = AggreTitleEn(Textfilepath, "cnbc.csv") 
    news_df["summary"] = news_df.titlelist.apply(lambda x: randomconcate(x, Maxtitlenums))
    news_df["summlen"] = news_df.summary.str.len()

    news_df = news_df.drop(['titlelist'], axis=1)

    df = pd.DataFrame(pd.date_range(start='2015-01-01', end=end_date), columns=['time'])
    news_df["time"] = pd.to_datetime(news_df["time"])
    df["time"] = pd.to_datetime(df["time"])

    news_df = pd.merge(news_df, df, on='time', how='outer')
    news_df["summary"] = news_df["summary"].apply(lambda x: 'x' if type(x) == type(0.0) else x)
    # df['new column name'] = df['column name'].apply(lambda x: 'value if condition is met' if x condition else 'value if condition is not met')


    news_df = news_df.sort_values(by="time")
    news_df.to_csv(writepath + "en_News_data.csv", encoding="utf_8_sig", index=False)
    print("en new finished")


# news preprocess
en_news_pre()
ch_news_pre()

# Exacute bert_extraction.py  / en_bert_extraction.py
extractBertFeatures()
extractCasedBertFeatures()