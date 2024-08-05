import requests
from bs4 import BeautifulSoup
from datetime import date, timedelta
import datetime
import pandas as pd
from import_tool import ROOT

def daterange(startDate, endDate):
    for n in range(int((endDate - startDate).days)+1):
        yield startDate + timedelta(n)

def get_price(start_date, end_date):
    root_url = "https://www.taifex.com.tw/cht/3/futDailyMarketReport?queryType=2&marketCode=0&commodity_id=TX&queryDate="
    res = []
    dates = []

    dt_start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    dt_end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d')

    for single_date in daterange(dt_start_date.date(), dt_end_date.date()):
        if(single_date.weekday()==5 or single_date.weekday()==6):
            #print(single_date)
            continue
        url = root_url + single_date.strftime("%Y/%m/%d")
        #print(url)

        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")

        # Extract the table data
        table_body = soup.find("tbody")
        if table_body:
            rows = table_body.find_all("tr")
        else:
            print(single_date.strftime("%Y/%m/%d"))
            continue
        dates.append(single_date.strftime("%Y-%m-%d"))

        # rows = soup.find_all("table")

        data = []
        ignore_list = ['\r', '\n', '\t', ',', ' ', '▼', '▲', '%']
        # for ignore in ignore_list:
        #     target_table = [item.replace(ignore, '') for item in target_table]
        for row in rows:
            cells = row.find_all("td")
            data_row = []
            for cell in cells:

                cellval = cell.text
                if cellval == '-':
                    cellval = None
                else:
                    for ignore in ignore_list:
                        cellval = cellval.replace(ignore, '')
                        # print(cellval)
                data_row.append(cellval)
            data.append(data_row)

        res.append(data[0][2:10])

        # Print the extracted data
        # for row in data:
        #     print(row)

    pdataframe = pd.DataFrame(res)
    for i in range(3):
        pdataframe=pdataframe.drop(pdataframe.columns[4], axis=1)

    pdataframe['date'] = dates

    cols = pdataframe.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    pdataframe = pdataframe[cols]

    df2 = pdataframe.set_axis(['date','open','high','low','close','volume'], axis=1)

    df2.to_csv(ROOT + 'TWF_price.csv',index=0)