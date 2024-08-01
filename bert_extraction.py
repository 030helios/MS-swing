# -*- coding: utf-8 -*-
'''
@ Author: HsinWei
@ Create Time: 2023-05-10 22:32:18
@ Description:
'''
from import_tool import *


def extractBertFeatures():
  writepath = './import_csv/'
  news_data = 'News_data.csv'
  modelPATH = './bert-base-chinese/'
  Featurepath = './Feature/'
  batch_size = 4
  news_data_sample = pd.read_csv(writepath + news_data)
  tokenizer = BertTokenizer.from_pretrained(modelPATH)
  model = TFBertModel.from_pretrained(modelPATH)
  textfeature = []

  for i in range(0, len(news_data_sample)):
    news_data_test = news_data_sample['summary'][i]
    inputs = tokenizer(news_data_test, return_tensors='tf')
    outputs = model(inputs)
    textfeature.append(outputs[1])
    print("{} finished".format(news_data_sample['time'][i]))
  textfeature = np.concatenate(textfeature)
  print(textfeature.shape)

  features_column = [f'TextF{i}' for i in range(1,769)] #BERT FEATURES
  features_df = pd.DataFrame(textfeature, columns=features_column)
  features = pd.concat([news_data_sample, features_df], axis=1)
  print(features.head())
  features.to_pickle(Featurepath+"./news_features.pkl")
  print("Extract bert features finished")


'''
def extractroBertFeatures():
      writepath = './import_csv/'
  news_data = 'News_data.csv'
  modelPATH = './chinese-roberta-wwm-ext/'
  Featurepath = './Feature/'
  batch_size = 4
  news_data_sample = pd.read_csv(writepath + news_data)
  tokenizer = BertTokenizer.from_pretrained(modelPATH)
  model = TFBertModel.from_pretrained(modelPATH)
  textfeature = []

  for i in range(batch_size, len(news_data_sample), batch_size):
    news_data_test = news_data_sample['summary'][i - batch_size:i].tolist()
    inputs = tokenizer(news_data_test, padding=True, truncation=True, return_tensors="np")
    outputs = model(**inputs)
    textfeature.append(outputs[1])
    print("{} finished".format(news_data_sample['time'][i]))
  textfeature = np.concatenate(textfeature)
  print(textfeature.shape)

  features_column = [f'TextF{i}' for i in range(1,769)] #BERT FEATURES
  features_df = pd.DataFrame(textfeature, columns=features_column)
  features = pd.concat([news_data_sample, features_df], axis=1)
  print(features.head())
  features.to_pickle(Featurepath+"./news_roberta_features.pkl")
  print("Extract bert features finished")
'''


# extractBertFeatures()