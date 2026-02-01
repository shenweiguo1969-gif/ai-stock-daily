import os
import json
from datetime import datetime
from dashscope import Generation
import akshare as ak
import pandas as pd

# 配置 Qwen3 API
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
Generation.api_key = DASHSCOPE_API_KEY

def load_stock_list():
    """从 STOCKS.txt 加载股票代码（每行一个6位A股代码）"""
    try:
        with open("STOCKS.txt", "r", encoding="utf-8") as f:
            stocks = [
                line.strip()
                for line in f
                if line.strip() and not line.startswith("#")
            ]
        return stocks
    except FileNotFoundError:
        print("⚠️ 未找到 STOCKS.txt，使用默认股票池")
        return ["600519", "000858", "300750"]

STOCKS = load_stock_list()

def calculate_rsi(prices, window=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1]

def get_stock_data(symbol):
    try:
        df = ak.stock_zh_a_hist(
            symbol=symbol,
            period="daily",
            start_date="20240101",
            adjust="qfq"
        )
        if df.empty:
            return None

        df.rename(columns={
            '日期': 'date',
            '开盘': 'open',
            '收盘': 'close',
            '最高': 'high',
            '最低': 'low',
            '成交量': 'volume'
        }, inplace=True)

        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
        df.dropna(subset=['close', 'volume'], inplace=True)
        df.reset_index(drop=True, inplace=True)

        if len(df) < 2:
            return None

        latest = df.iloc[-1]
        prev = df.iloc[-2]
        close_prices = df['close']

        rsi = calculate_rsi(close_prices) if len(close_prices) >= 14 else "N/A"
        ma20 = close_prices.tail(20).mean() if len(close_prices) >= 20 else "N/A"
        change_pct = ((latest['close'] - prev['close']) / prev['close']) * 100

        return {
            "symbol": symbol,
            "price": round(latest['close'], 2),
            "change_pct": round(change_pct, 2),
            "volume": int(latest['volume']),
            "rsi": round(rsi, 2) if isinstance(rsi, float) else rsi,
            "ma20": round(ma20, 2) if isinstance(ma20, float) else ma20,
            "last_5_days": close_prices.tail(5).round(2).tolist()
        }

    except Exception:
        return None

def generate_analysis(data):
    prompt = f"""
你是一位专业的中文股票分析师，请根据以下数据生成一段150字以内的简明分析：
- 股票代码: {data['symbol']}
- 当前价格: ¥{data['price']}
- 近5日收盘价: {data['last_5_days']}
- RSI指标: {data['rsi']}（>70超买，<30超卖）
- 20日均线: {data['ma20']}
- 今日涨跌幅: {data['change_pct']}%

要求：
1. 用中文回答，语气专业但易懂；
2. 包含趋势判断（如“短期偏强”、“震荡整理”）；
3. 提示风险（如“RSI接近超买区，注意回调”）；
4. 不要提及“AI”、“模型”等词。
"""
    try:
        response = Generation.call(
            model="qwen-max",
            prompt=prompt,
            max_tokens=200
        )
        if response.status_code == 200:
            return response.output.text.strip()
        else:
            return "分析生成失败"
    except Exception:
        return "分析服务异常"

def main():
    os.makedirs("output", exist_ok=True)

    results = []
    for symbol in STOCKS:
        data = get_stock_data(symbol)
        if data:
            analysis = generate_analysis(data)
            data["analysis"] = analysis
            results.append(data)

    output = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "stocks": results
    }
    with open("output/predictions.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
