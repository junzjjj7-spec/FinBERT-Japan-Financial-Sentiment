!pip install yfinance - -upgrade - -no - cache - dir - q
!pip install pandas requests tqdm openpyxl - q

import yfinance as yf
import pandas as pd
import requests
import time
import random
from io import StringIO
from tqdm import tqdm


# ==========================================
# 1. 获取海量代码 (双重保险策略)
# ==========================================
def get_sp500_tickers():
    """策略A: 伪装成浏览器去爬维基百科"""
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        # 关键修复：加上 User-Agent，假装自己是 Chrome 浏览器
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        r = requests.get(url, headers=headers)
        # 使用 StringIO 处理 HTML 文本
        dfs = pd.read_html(StringIO(r.text))
        tickers = dfs[0]['Symbol'].tolist()
        print(f"✅ 成功从维基百科获取 {len(tickers)} 家公司代码")
        return tickers
    except Exception as e:
        print(f"⚠️ 维基百科抓取失败: {e}")
        return []


# 获取列表
sp500_tickers = get_sp500_tickers()

# 策略B: 硬编码的保底列表 (涵盖美股/日股/科技/金融核心)
# 如果上面失败了，这些能保证至少有 1000+ 条新闻
fallback_tickers = [
    # 美国科技七巨头 + 热门
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "NFLX", "ADBE", "CRM",
    "INTC", "AMD", "QCOM", "AVGO", "TXN", "MU", "AMAT", "LRCX", "IBM", "ORCL",
    # 华尔街金融
    "JPM", "BAC", "WFC", "C", "GS", "MS", "BLK", "AXP", "V", "MA", "PYPL",
    # 知名中概股/ADR (阿里京东等新闻多)
    "BABA", "PDD", "JD", "BIDU", "TCEHY", "NIO", "XPEV", "LI",
    # 日本核心 (你的主场)
    "SONY", "TM", "HMC", "SFTBY", "IX", "MUFG", "SMFG", "MFG", "NMR", "TAK",
    "CAJ", "PCRFY", "FANUY", "NTDOY", "MZDAY", "NSANY", "SZKMY", "DNZOY", "FUJHY",
    # 欧洲/全球巨头
    "TSM", "ASML", "SAP", "NVO", "NVS", "AZN", "SHEL", "TTE", "HSBC", "UL"
]

# 合并所有代码 + 去重
all_tickers = list(set(sp500_tickers + fallback_tickers))
random.shuffle(all_tickers)  # 打乱顺序，防止只抓到同一个行业的

print(f"🎯 最终锁定目标公司: {len(all_tickers)} 家")
if len(all_tickers) < 100:
    print("❌ 警告：代码列表过短，可能无法凑够5000条，请检查网络。")


# ==========================================
# 2. 智能提取函数
# ==========================================
def extract_title_smart(news_item):
    if 'title' in news_item: return news_item['title']
    if 'headline' in news_item: return news_item['headline']
    if 'content' in news_item and isinstance(news_item['content'], dict):
        return extract_title_smart(news_item['content'])
    return None


# ==========================================
# 3. 极速抓取循环 (目标 5000)
# ==========================================
collected_news = []
seen_titles = set()
TARGET_COUNT = 5000

print(f"🚀 全力开动！目标抓取 {TARGET_COUNT} 条...")
pbar = tqdm(total=TARGET_COUNT)

for ticker in all_tickers:
    # 抓够了就提前收工
    if len(collected_news) >= TARGET_COUNT:
        break

    try:
        stock = yf.Ticker(ticker)
        news_list = stock.news

        if not news_list: continue

        for news in news_list:
            title = extract_title_smart(news)

            if title and title not in seen_titles:
                # 简单过滤：只要长度大于15个字符的正常句子
                if len(str(title)) > 15:
                    collected_news.append({
                        "company": ticker,
                        "text": title,
                        "label": -1
                    })
                    seen_titles.add(title)
                    pbar.update(1)

                    if len(collected_news) >= TARGET_COUNT:
                        break

        # 极速模式：把休眠时间调短，因为yfinance最近不太封Colab
        # 如果报错频繁，可以把这里改成 0.2
        time.sleep(0.05)

    except:
        continue

pbar.close()

# ==========================================
# 4. 保存
# ==========================================
df_final = pd.DataFrame(collected_news)
print(f"\n📊 最终战果: {len(df_final)} 条")

filename = "real_world_5k_raw.xlsx"
df_final.to_excel(filename, index=False)
print(f"💾 文件已保存: {filename}")

from google.colab import files

try:
    files.download(filename)
except:
    pass