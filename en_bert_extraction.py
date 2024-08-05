from import_tool import *
from transformers import BertTokenizer, TFBertModel

def extractUncasedBertFeatures():
    writepath = './import_csv/'
    modelPATH = './bert-base-uncased/'
    #modelPATH = './finbert/'
    news_data = 'en_News_data.csv'
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

    features_column = [f'EnTextF{i}' for i in range(1,769)] #BERT FEATURES
    features_df = pd.DataFrame(textfeature, columns=features_column)
    features = news_data_sample.merge(features_df, left_index=True, right_index=True)
    print(features)
    features.to_pickle(Featurepath+"./news_en_features.pkl")
    # features.to_pickle(Featurepath+"./news_en_cased_features.pkl")

    print("Extract uncased bert features finished")

def extractCasedBertFeaturesCLS():
    writepath = './import_csv/'
    modelPATH = './bert-base-cased/'
    #modelPATH = './finbert/'
    news_data = 'en_News_data.csv'
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
        textfeature.append(outputs[0][0][1])
        #textfeature.append(outputs[1])
        print("{} finished".format(news_data_sample['time'][i]))
    textfeature = np.stack(textfeature)
    print(textfeature.shape)

    features_column = [f'EnTextF{i}' for i in range(1,769)] #BERT FEATURES
    features_df = pd.DataFrame(textfeature, columns=features_column)
    features = news_data_sample.merge(features_df, left_index=True, right_index=True)
    print(features)
    features.to_pickle(Featurepath+"./news_en_cased_features.pkl")

    print("Extract cased bert features finished")

def extractCasedBertFeaturesMEAN():
    writepath = './import_csv/'
    modelPATH = './bert-base-cased/'
    #modelPATH = './finbert/'
    news_data = 'en_News_data.csv'
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
        #print(np.mean(np.array(outputs[0][0]),axis=0))
        textfeature.append(np.mean(np.array(outputs[0][0]),axis=0))
        print("{} finished".format(news_data_sample['time'][i]))
    textfeature = np.stack(textfeature)
    print(textfeature.shape)

    features_column = [f'EnTextF{i}' for i in range(1,769)] #BERT FEATURES
    features_df = pd.DataFrame(textfeature, columns=features_column)
    features = news_data_sample.merge(features_df, left_index=True, right_index=True)
    print(features)
    features.to_pickle(Featurepath+"./news_en_cased_features.pkl")

    print("Extract cased bert features finished")

def extractCasedBertFeatures():
    writepath = './import_csv/'
    modelPATH = './bert-base-cased/'
    #modelPATH = './finbert/'
    news_data = 'en_News_data.csv'
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

    features_column = [f'EnTextF{i}' for i in range(1,769)] #BERT FEATURES
    features_df = pd.DataFrame(textfeature, columns=features_column)
    features = news_data_sample.merge(features_df, left_index=True, right_index=True)
    print(features)
    features.to_pickle(Featurepath+"./news_en_cased_features.pkl")

    print("Extract cased bert features finished")
    features.to_excel(Featurepath+"news_en.xlsx")

#extractCasedBertFeaturesMEAN()