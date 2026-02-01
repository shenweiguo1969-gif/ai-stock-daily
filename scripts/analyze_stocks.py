import os
import json
import yfinance as yf
from datetime import datetime
from dashscope import Generation

# 配置 Qwen3
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
Generation.api_key = DASHSCOPE_API_KEY

# 股票列表（支持 A股/港股/美股）
STOCKS = ["AAPL", "TSLA", "600519.SS", "00700.HK"]

def get_stock_data(symbol):
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="30d")
        if hist.empty:
            return None
        latest = hist.iloc[-1]
        prev = hist.iloc[-2] if len(hist) > 1 else latest
        # 计算 RSI
        rsi = calculate_rsi(hist['Close']) if len(hist) >= 14 else "N/A"
        ma20 = hist['Close'].tail(20).mean() if len(hist) >= 20 else "N/A"
        change_pct = ((latest['Close'] - prev['Close']) / prev['Close']) * 100
        return {
            "symbol": symbol,
            "price": round(latest['Close'], 2),
            "change_pct": round(change_pct, 2),
            "volume": int(latest['Volume']),
            "rsi": round(rsi, 2) if isinstance(rsi, float) else rsi,
            "ma20": round(ma20, 2) if isinstance(ma20, float) else ma20,
            "last_5_days": hist['Close'].tail(5).round(2).tolist()
        }
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return None

def calculate_rsi(prices, window=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1]

def generate_analysis(data):
    prompt = f"""
你是一位专业的中文股票分析师，请根据以下数据生成一段150字以内的简明分析：
- 股票代码: {data['symbol']}
- 当前价格: ¥{data['price']}（若为美股则单位为美元）
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
            return f"分析失败（错误码: {response.code}）"
    except Exception as e:
        return f"调用异常: {str(e)}"

def main():
    if not os.path.exists("output"):
        os.makedirs("output")
    results = []
    for symbol in STOCKS:
        print(f"Analyzing {symbol}...")
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
    print("✅ 分析完成！")

if __name__ == "__main__":
    main()
