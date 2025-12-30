# 1. 安装必要的库
!pip
install
openai
pandas
tqdm
openpyxl - q

import pandas as pd
from openai import OpenAI
from tqdm import tqdm
import json
from google.colab import files

# ================= 配置区 =================
# 🔴 请在这里填入你的 DeepSeek API Key
API_KEY = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

# 🔴 你的 Excel 文件名 (请确保已上传到 Colab)
INPUT_FILE = "real_world_final.xlsx"
# 输出文件名
OUTPUT_FILE = "real_world_labeled.xlsx"

# DeepSeek 配置
BASE_URL = "https://api.deepseek.com"
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


# =========================================

def get_sentiment_batch(texts):
    """发送给大模型进行批量标注"""
    # 经过优化的金融情感分析 Prompt
    prompt = f"""
    你是一个资深金融分析师。请分析以下财经新闻标题的情感极性。

    【判断标准】
    - 0 = Positive (利好): 盈利增长, 股价大涨, 收购/扩张, 获得合同, 新产品发布, 回购, 合作, 技术突破, 销量新高。
    - 1 = Negative (利空): 亏损/下滑, 股价暴跌, 诉讼/调查, 裁员/辞职, 故障/延误, 评级下调, 罢工, 停产。
    - 2 = Neutral (中性): 战略审查, 一般人事变动(无负面背景), 混合业绩(mixed results), 单纯的资产出售, 事实陈述, 维持评级。

    【特殊规则】
    1. "Record high sales/profit" (创新高) 必须标为 0 (利好)。
    2. "Loss narrowed" (亏损收窄) 视为 0 (利好)。
    3. 仅仅提到公司名而无具体事件的，标为 2 (中性)。
    4. "Lawsuit dismissed" (诉讼驳回/胜诉) 是 0 (利好)。

    【输入数据】
    {json.dumps(texts, ensure_ascii=False)}

    【输出格式】
    只返回一个 JSON 对象，格式为: {{"results": [0, 1, 2, ...]}}
    不要包含任何Markdown标记或解释。
    """

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a financial sentiment analyzer. Return JSON only."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        content = response.choices[0].message.content
        return json.loads(content)["results"]
    except Exception as e:
        print(f"⚠️ API 报错: {e}")
        return [2] * len(texts)  # 出错保底回中性，防止程序中断


# ================= 主程序 =================
# 1. 读取数据
try:
    print(f"正在读取 {INPUT_FILE} ...")
    df = pd.read_excel(INPUT_FILE)
    # 确保有一列叫 'text'，如果没有，尝试找 'title' 或第一列
    if 'text' not in df.columns:
        if 'title' in df.columns:
            df.rename(columns={'title': 'text'}, inplace=True)
        else:
            print("⚠️ 没找到 'text' 列，默认使用第一列作为文本...")
            df.rename(columns={df.columns[0]: 'text'}, inplace=True)

    # 物理清洗：删掉太短的垃圾数据 (少于5个字符的)
    df = df[df['text'].astype(str).str.len() > 5].copy()

except FileNotFoundError:
    print(f"❌ 错误：找不到文件 {INPUT_FILE}，请检查是否已上传！")
    df = pd.DataFrame()  # 空防止报错

if not df.empty:
    # 2. 开始跑批
    batch_size = 20  # 每次发20条
    new_labels = []
    texts = df['text'].astype(str).tolist()

    print(f"🚀 开始 AI 自动标注 {len(texts)} 条数据... (预计需要几分钟)")

    try:
        for i in tqdm(range(0, len(texts), batch_size)):
            batch = texts[i: i + batch_size]
            labels = get_sentiment_batch(batch)

            # 长度校验
            if len(labels) != len(batch):
                labels = [2] * len(batch)

            new_labels.extend(labels)

    except KeyboardInterrupt:
        print("\n🛑 用户手动停止。正在保存已完成的部分...")

    # 3. 数据对齐与保存
    # 如果中途停止，只保存跑完的部分
    if len(new_labels) < len(df):
        print(f"⚠️ 注意：只跑完了 {len(new_labels)} 条，将截取保存。")
        df = df.iloc[:len(new_labels)]

    df['label'] = new_labels

    # 保存结果
    df.to_excel(OUTPUT_FILE, index=False)
    print(f"\n✅ 标注完成！文件已保存为: {OUTPUT_FILE}")

    # 4. 自动下载
    try:
        files.download(OUTPUT_FILE)
        print("⬇️ 正在触发浏览器下载...")
    except Exception as e:
        print("自动下载失败，请在左侧文件栏手动右键下载。")