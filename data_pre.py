# -*- coding: utf-8 -*-
'''
@ Author: HsinWei
@ Create Time: 2023-04-05 16:19:48
@ Description:
'''

from import_tool import *
# from bert_extraction import extractBertFeatures
# from en_bert_extraction import extractCasedBertFeatures
# from news_data_pre import en_news_pre, ch_news_pre

# SET PARAMS
ROOT = './import_csv/'
# SET TIME


def expiration_cal(x):
    settlement = pd.read_csv(ROOT+"txf_settlement.csv")
    settlement['txf_settlement'] = pd.to_datetime(settlement['txf_settlement'])
    settlement = settlement.set_index(settlement['txf_settlement'])
    remain = (settlement[x.strftime('%Y/%m')].index - x)[0]
    if remain >= pd.Timedelta("0 days"):
        return remain
    else:
        return (settlement[(x + pd.Timedelta(15, unit="d")).strftime('%Y/%m')].index - x)[0]


def settlement_cal(d):
    tmp = pd.DataFrame()
    tmp['date'] = pd.to_datetime(d['date'])
    d['until_expiration'] = tmp.date.apply(lambda x: expiration_cal(x))
    d['until_expiration'] = d['until_expiration'].apply(lambda x: x.days)
    print("Settlement preprocessing finished")
    return d

def price_data_pre(data, target):
    data = data.rename({'Date': 'date'}, axis='columns')
    # feature_generate
    COL = ['date', 'open', 'high', 'low', 'close', 'volume']
    data.columns = COL[:len(data.columns)]

    tmp_o = np.log(data['open'])
    data['log_rtn'] = np.log(data['close']/data['close'].shift(1)).round(5)
    data['norm_o'] = (tmp_o - np.log(data['close'].shift(1))).round(4)
    data['norm_h'] = (np.log(data['high']) - tmp_o).round(4)
    data['norm_l'] = (np.log(data['low']) - tmp_o).round(4)
    data['norm_c'] = (np.log(data['close']) - tmp_o).round(4)
    if 'volume' in data.columns:
        data['vol_deg_change'] = data['volume'].diff(1).apply(
            lambda x: (np.arctan2(x, 100) / np.pi) + 0.5).round(4)
        data['vol_percentile'] = data['volume'].rolling(len(data), min_periods=1).apply(
            lambda x: pd.Series(x).rank(pct=True).values[-1], raw=False).round(3)
        data.drop(['volume'], axis=1, inplace=True)

    data['range'] = data['high'] - data['low']
    data['body'] = (np.abs(data['open'] - data['close']) /
                    data['range']).fillna(1).round(2)
    data['upper_shadow'] = (
        (data['high'] - data[['open', 'close']].max(axis=1)) / data['range']).fillna(0).round(2)
    data['lower_shadow'] = ((data[['open', 'close']].min(
        axis=1) - data['low']) / data['range']).fillna(0).round(2)

    data.drop(COL[1:5], axis=1, inplace=True)
    data.drop(columns=['range'], axis=1, inplace=True)
    colname = ['date']
    for col in data.columns:
        if col != 'date':
            colname.append(target+'_'+col)

    data.columns = colname
    print("Price data preprocessing finished")
    return data


naq = pd.read_csv(ROOT+'naq.csv')
naq = price_data_pre(naq, 'naq')
sp500 = pd.read_csv(ROOT+'sp500.csv')
sp500 = price_data_pre(sp500, 'sp500')
phlx = pd.read_csv(ROOT+'phlx.csv')
phlx = price_data_pre(phlx, 'phlx')

sha = pd.read_csv(ROOT+'sha.csv')
sha = price_data_pre(sha, 'sha')
hsi = pd.read_csv(ROOT+'hsi.csv')
hsi = price_data_pre(hsi, 'hsi')
nik = pd.read_csv(ROOT+'nik.csv')
nik = price_data_pre(nik, 'nik')

twf = pd.read_csv(ROOT+'TWF_price.csv')
# twf['date'] = pd.to_datetime(twf['date'])

taiex = pd.read_csv(ROOT+"taiex.csv")
taiex['basis'] = twf['close'] - taiex['close']
twf = price_data_pre(twf, 'twf')
twf['twf_basis'] = taiex['basis'] / 1000 # divide by const
tmp = settlement_cal(twf[['date']])
twf['twf_until_expiration'] = tmp['until_expiration']

merge = [twf, naq, sp500, phlx, sha, hsi, nik]
price_all = ft.reduce(lambda left, right: pd.merge(
    left, right, on='date'), merge)

price_all.dropna(inplace=True)
with open(ROOT+'price_pre.pickle', 'wb') as f:
    pickle.dump(price_all, f)

# %%
oi = pd.read_csv(ROOT+'3_L_all_oi.csv')


def oi_preprocess(d):
    out = d[['date']]
    for col in d.columns:
        if col == 'date':
            continue
        if not 'oi' in col:
            continue
        name = col+'_deg_change'
        temp = d[col].diff(1).apply(lambda x: (
            np.arctan2(x, 100) / np.pi) + 0.5).round(4)
        out.insert(1, name, temp)
        name = col+'_percentile'
        temp = d[col].rolling(len(d), min_periods=1).apply(
            lambda x: pd.Series(x).rank(pct=True).values[-1], raw=False).round(3)
        out.insert(1, name, temp)
    print("OI preprocessing finished")
    return out


oi_all = oi_preprocess(oi)
oi_all.dropna(inplace=True)
with open(ROOT+'oi_pre.pickle', 'wb') as f:
    pickle.dump(oi_all, f)



print("All data preprocessing finished")