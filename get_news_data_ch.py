from import_tool import *
import eventlet
ROOT = './News_data/'


def get_parsing_data(url_s):
    while True:
        with eventlet.Timeout(10):
            r = requests.get(url_s)
            break
    result = []
    
    if r.status_code == requests.codes.ok:
        soup = BeautifulSoup(r.text, "html.parser")
        stories = soup.find_all("div", class_="news")

        for s in stories: 
            date = s.find("span", class_="source")
            date = date.string[:10]
            idx, title = s.a.string.split(".", 1)
            # print(date + " " + title)
            result.append({"time": date, "title": title})
    return result

# https://udndata.com/ndapp/Index?cp=udn  
# 先在上面的網站 搜尋"台股" 自20XX年1月1日到20XX年12月31日
# 搜尋到的"第一頁URL 複製到line 36"，把中間的page=1改為 page={}
# 手動看一下該年總共有幾頁的新聞資料
# 再把總共頁數填進 line 35

with open('latestURL.txt') as f:
    latestURL = [line.rstrip() for line in f]
    latestURL = latestURL[0]
    splittedURL = latestURL.split('&')
    page = splittedURL[1][5:]
    page = int(page)

    splittedURL[1] = "page={}"

    unformatURL = '&'.join(splittedURL)
    unformatURL = 'https://'+unformatURL
    
output = []
for i in range(page):
    url = unformatURL.format(
        i + 1
    )
    result = get_parsing_data(url)
    output += result
    print("page{} is ok.".format(i + 1))


df = pd.DataFrame(output)


df.to_excel(ROOT + "news_2024.xlsx")   