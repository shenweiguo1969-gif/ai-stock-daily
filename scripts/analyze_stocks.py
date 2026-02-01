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
    with open("STOCKS.txt", "r", encoding="utf-8") as f:
        stocks = [
            line.strip()
            for line in f
            if line.strip() and not line.startswith("#")
        ]
    return stocks

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
        if df.empty or len(df) < 5:
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
        volumes = df['volume']

        change_pct = ((latest['close'] - prev['close']) / prev['close']) * 100
        rsi = calculate_rsi(close_prices) if len(close_prices) >= 14 else "N/A"
        ma20 = close_prices.tail(20).mean() if len(close_prices) >= 20 else "N/A"

        # === 量价关系 ===
        vol_5d_avg = volumes.tail(5).mean()
        price_up = latest['close'] > prev['close']
        vol_up = latest['volume'] > prev['volume']

        if price_up and vol_up:
            volume_price_signal = "价涨量增（趋势健康）"
        elif not price_up and latest['volume'] < prev['volume']:
            volume_price_signal = "价跌量缩（抛压减轻）"
        elif price_up and not vol_up:
            volume_price_signal = "缩量上涨（持续性存疑）"
        elif not price_up and vol_up:
            volume_price_signal = "放量下跌（主力出货或洗盘）"
        else:
            volume_price_signal = "量价中性"

        # === 资金流向 ===
        fund_direction = "数据暂无"
        net_inflow = "N/A"
        try:
            market = "sh" if symbol.startswith(("60", "68")) else "sz"
            fund_df = ak.stock_individual_fund_flow(stock=symbol, market=market)
            if not fund_df.empty:
                inflow_str = fund_df.iloc[0]['主力净流入-净额']
                if isinstance(inflow_str, str) and '万' in inflow_str:
                    net_inflow_val = float(inflow_str.replace('万', '').replace(',', ''))
                    net_inflow = f"{net_inflow_val:.1f}"
                    fund_direction = "资金净流入" if net_inflow_val > 0 else "资金净流出"
        except Exception:
            pass

        return {
            "symbol": symbol,
            "price": round(latest['close'], 2),
            "change_pct": round(change_pct, 2),
            "volume": int(latest['volume']),
            "rsi": round(rsi, 2) if isinstance(rsi, float) else rsi,
            "ma20": round(ma20, 2) if isinstance(ma20, float) else ma20,
            "last_5_days": close_prices.tail(5).round(2).tolist(),
            "volume_price_signal": volume_price_signal,
            "fund_direction": fund_direction,
            "net_inflow": net_inflow,
        }

    except Exception:
        return None

def generate_analysis(data):
    prompt = f"""
你是一位资深中文股票分析师，请基于以下多维数据生成150字以内简明分析：

- 股票代码: {data['symbol']}
- 当前价格: ¥{data['price']} | 涨跌幅: {data['change_pct']}%
- 近5日走势: {data['last_5_days']}
- 量价关系: {data['volume_price_signal']}
- 资金流向: {data['fund_direction']}（净流入/出: {data['net_inflow']}万）
- RSI: {data['rsi']}（>70超买，<30超卖）
- 20日均线: {data['ma20']}

要求：
1. 优先解读「量价配合」和「主力资金动向」；
2. 判断当前是「吸筹」「拉升」「派发」还是「洗盘」阶段；
3. 给出操作建议（如“可逢低布局”、“警惕高位放量滞涨”）；
4. 语言专业简洁，避免术语堆砌，不提“AI”或“模型”。
"""
    try:
        response = Generation.call(
            model="qwen-max",
            prompt=prompt,
            max_tokens=250
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
    for symbol in STOC
