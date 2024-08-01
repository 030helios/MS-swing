import requests
import pandas as pd
from import_tool import end_date


ROOT = './News_data/'

params = {
    "queryly_key": "31a35d40a9a64ab3",
    "query": "stock",                   #shearching key word
    "endindex": "0",
    "batchsize": "100",
    "callback": "",
    "showfaceted": "true",
    "timezoneoffset": "-120",
    "facetedfields": "formats",
    "facetedkey": "formats|",
    "facetedvalue": "!Press Release|",
    "needtoptickers": "1",
    "additionalindexes": "4cd6f71fbf22424d,937d600b0d0d4e23,3bfbe40caee7443e,626fdfcd96444f28",
}

goal = ["cn:title", "_pubDate", "cn:liveURL", "description"]




url = "https://api.queryly.com/cnbc/json.aspx"


def get_en_news(url):
    with requests.Session() as req:
        old_news = pd.read_csv(ROOT + "cnbc.csv", encoding= 'UTF-8')
        update_to_date = pd.to_datetime(old_news['Date'][0])    #上次抓到哪一個時間點
        Newest_date = pd.to_datetime(end_date)
        # print(update_to_date)

        allin = []
        for page, item in enumerate(range(0, 3000, 100)):       # range(0, 3000, 100)代表下載近3000筆資料，應該每個月更新一次就可以了，會自動載到 update_to_date 這個時間點
            print(f"Extracting Page  #{page + 1}")
            params["endindex"] = item
            r = req.get(url, params=params).json()

            for loop in r["results"]:
                allin.append([loop[x] for x in goal])
        new = pd.DataFrame(allin, columns=["Title", "Ori_Date", "Url", "Description"])
        ori_date = new["Ori_Date"].str.split(" ", n=1, expand=True)
        new["Date"] = ori_date[0]
        new = new.drop(columns=["Url", "Description", "Ori_Date"])

        new = new.loc[:, ["Date", "Title"]]
        
        new['Date'] = pd.to_datetime(new['Date']).dt.date
        new = new[pd.to_datetime(new["Date"]) >= update_to_date]
        new = new.sort_values(by=["Date"], ascending=False)
        
        ####    Check if the data is corrected update.  ####
        print("Updated to", new['Date'].iloc[0])
        print("Requested date", Newest_date)
        if new['Date'].iloc[0] != Newest_date :
            print("Data didn't update to the lastest date!")
        else :
            print("Completed update!")
        ####################################################

        res = pd.concat([new, old_news], axis=0)
        res.drop_duplicates(subset=None, inplace=True)
        res.to_csv(ROOT + "cnbc.csv", index=False, encoding= 'UTF-8')



get_en_news(url)

