from import_tool import *
from crawler_fut import get_price
import config

ROOT = './import_csv/'

# the db config file please refer to the islab finance
#---connect database---
conn = psycopg2.connect(user=config.server['user'],
                        password=config.server['password'],
                        host=config.server['host'],
                        port=config.server['port'],
                        database=config.server['database'])
cur = conn.cursor()

def get_taiex(start_date,end_date):
    start_date:str = start_date
    end_date:str = end_date
    sql = '''SELECT DISTINCT ON(date)date, open, high, low, close , volume FROM api_twse_taiex where date >= '{start_date}' and date <= '{end_date}' 
    ORDER BY date asc'''.format(start_date=start_date,end_date =end_date)
    cur.execute(sql)
    df_taiex = pd.DataFrame(cur.fetchall(),columns = [col[0] for col in cur.description])

    df_taiex.to_csv(ROOT+"taiex.csv",index = 0)
    return df_taiex

def get_big3_futures_oi(start_date,end_date,target):
    start_date:str = start_date
    end_date:str = end_date
    target:str = target
    
    
    sql_TXF_oi_all = '''select date,sum(long_oi_volume) AS {target}_long_oi, 
    sum(short_oi_volume) AS {target}_short_oi, sum(net_oi_volume) AS {target}_net_oi from api_taifex_big3_futures 
    where contract = '{target}' and date >= '{start_date}' and date <= '{end_date}'
    group by date order by date '''.format(target=target,start_date=start_date,end_date =end_date)
    
    cur.execute(sql_TXF_oi_all)
    rows = cur.fetchall()
    
    df_all = pd.DataFrame(rows,columns = [col[0] for col in cur.description])
    # df_all= pd.read_sql(sql_TXF_oi_all, conn)
    
    for item in ['trust','foreigner','dealer']:
        sql = '''select date,sum(long_oi_volume) AS {target}_{type3}_long_oi, 
        sum(short_oi_volume) AS {target}_{type3}_short_oi, sum(net_oi_volume) AS {target}_{type3}_net_oi from api_taifex_big3_futures 
        where contract = '{target}' and date > '{start_date}'  
        and date <= '{end_date}' and type = '{type3}'
        group by date order by date '''.format(target=target,start_date=start_date,end_date =end_date,type3=item)
        # df = pd.read_sql(sql, conn)
        cur.execute(sql)
        df = pd.DataFrame(cur.fetchall(),columns = [col[0] for col in cur.description])
        # df_temp = df_all
        df_all = pd.merge(df,df_all)
    return df_all 


# +

def get_large_trader_oi():
    #買權 所有契約 指標
    sql = "select *  from api_taifex_large_trader_options where cp='c' and contract_month LIKE 'all' order by date asc;"
    cur.execute(sql)
    df_c=pd.DataFrame(cur.fetchall(),columns = [col[0] for col in cur.description])
    df_c.drop(['contract', 'contract_month', 'cp'],axis=1,inplace=True)
    colname = ['date']
    for c in df_c.columns:
        if "date" not in c:
            colname.append(c+"_call")
        
    df_c.columns = colname
    #賣權 所有契約 指標
    sql = "select *  from api_taifex_large_trader_options where cp='p' and contract_month LIKE 'all' order by date asc;"
    cur.execute(sql)
    df_p=pd.DataFrame(cur.fetchall(),columns = [col[0] for col in cur.description])
    df_p.drop(['contract', 'contract_month', 'cp'],axis=1,inplace=True)
    colname = ['date']
    for c in df_p.columns:
        if "date" not in c:
            colname.append(c+"_put")
        
    df_p.columns = colname
    
    # large trader futures
    cur.execute("select * FROM api_taifex_large_trader_futures WHERE contract_month = 'all' order by date asc;")
    df = pd.DataFrame(cur.fetchall(), columns = [col[0] for col in cur.description])
    df.drop(['contract', 'contract_month'],axis=1,inplace=True)
    
    colname = ['date']
    for c in df.columns:
        if "date" not in c:
            colname.append(c+"_future")
        
    df.columns = colname
    
    large_trader = pd.merge(df_c,df_p)
    large_trader = pd.merge(large_trader,df)
    
    return large_trader
# -


def get_ASIA_data():
    sha = yf.download('000001.SS',start=start_date,end=end_date)    #上海指數
    hsi = yf.download('^HSI',start=start_date,end=end_date)         #恆生指數
    nik = yf.download('^N225',start=start_date,end=end_date)        #日經指數

    sha.drop(['Adj Close'],inplace=True,axis=1)
    hsi.drop(['Adj Close'],inplace=True,axis=1)
    nik.drop(['Adj Close'],inplace=True,axis=1)
    
    sha = sha.reset_index(level=0)
    hsi = hsi.reset_index(level=0)
    nik = nik.reset_index(level=0)
    
    sha.to_csv(ROOT+'sha.csv',index=0)
    hsi.to_csv(ROOT+'hsi.csv',index=0)
    nik.to_csv(ROOT+'nik.csv',index=0)
    
    sha.columns = ['date','sha_open','sha_high','sha_low','sha_close','sha_volume']
    hsi.columns = ['date','hsi_open','hsi_high','hsi_low','hsi_close','hsi_volume']
    nik.columns = ['date','nik_open','nik_high','nik_low','nik_close','nik_volume']

    data = pd.merge(sha,hsi)
    data = pd.merge(data,nik)
    data = data.rename({'Date': 'date'}, axis='columns')
    
    data['date'] = pd.to_datetime(data['date'])
    data['date'] = data['date'].dt.strftime('%Y-%m-%d')
    
    data.to_csv(ROOT+'asia_data.csv',index=0)

get_ASIA_data()
print("download ASIA market data")


def get_usa_data():
    naq = yf.download('^IXIC',start=start_date,end=end_date)
    sp500 = yf.download('^GSPC',start=start_date,end=end_date)
    phlx = yf.download('^SOX',start=start_date,end=end_date)
    naq.drop(['Adj Close'],inplace=True,axis=1)
    sp500.drop(['Adj Close'],inplace=True,axis=1)
    phlx.drop(['Adj Close','Volume'],inplace=True,axis=1)
    
    naq = naq.reset_index(level=0)
    sp500 = sp500.reset_index(level=0)
    phlx = phlx.reset_index(level=0)
    
    naq.to_csv(ROOT+'naq.csv',index=0)
    sp500.to_csv(ROOT+'sp500.csv',index=0)
    phlx.to_csv(ROOT+'phlx.csv',index=0)
    
    naq.columns = ['date','naq_open','naq_high','naq_low','naq_close','naq_volume']
    sp500.columns = ['date','sp500_open','sp500_high','sp500_low','sp500_close','sp500_volume']
    phlx.columns = ['date','phlx_open','phlx_high','phlx_low','phlx_close']

    data = pd.merge(naq,sp500)
    data = pd.merge(data,phlx)
    data = data.rename({'Date': 'date'}, axis='columns')
    
    data['date'] = pd.to_datetime(data['date'])
    data['date'] = data['date'].dt.strftime('%Y-%m-%d')

    data.to_csv(ROOT+'usa_data.csv',index=0)

print("download USA data")
get_usa_data()
#%%
# +
print("download TWF price")
df_TWF_price = get_price(start_date,end_date)
print("download TAIEX")
df_taiex = get_taiex(start_date,end_date)

print("download TXF big3 oi")
target='TXF'
df_TXF = get_big3_futures_oi(start_date,end_date,target)
df_TXF.to_csv(ROOT+target+'_all_oi.csv',index=0)
print("download MXF big3 oi")
target='MXF'
df_MXF = get_big3_futures_oi(start_date,end_date,target)
df_MXF.to_csv(ROOT+target+'_all_oi.csv',index=0)
df = pd.merge(df_TXF,df_MXF)
print("download TXF Large Trader oi (option&future)")
largetrader = get_large_trader_oi()
data = pd.merge(df,largetrader)

print("data merged")
data = pd.merge(df,data)
data = pd.merge(data,df_TWF_price)
data.to_csv(ROOT+'3_L_all_oi.csv',index=0)






print("finished")
# -


