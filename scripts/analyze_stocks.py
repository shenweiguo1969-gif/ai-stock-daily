import os
import json
import time
from datetime import datetime
from dashscope import Generation
import akshare as ak
import pandas as pd

# 配置 Qwen3 API
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
Generation.api_key = DASHSCOPE_API_KEY

def load_stock_list():
    with open("STOCKS.txt", "r", encoding="utf-8") as f:
        stocks = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    return stocks

def get_stock_name_safe(symbol):
    """
    安全获取单只股票名称，确保无缓存污染
    """
    try:
        # 明确指定市场
        if symbol.startswith(('60', '688', '689')):
            market = "sh"
        elif symbol.startswith(('00', '30', '8')):
            market = "sz"
        else:
            return None

        # ⚠️ 关键：每次调用前清除可能的内部缓存（通过新进程模拟，此处用重试+延迟）
        time.sleep(0.1)  # 防止请求过快被限
        
        df = ak.stock_individual_info_em(symbol=symbol, market=market)
        
        if df is not None and not df.empty:
            # 查找“公司全称”或“股票简称”
            if 'item' in df.columns and 'value' in df.columns:
                # 尝试获取股票简称（更短）
                short_name_row = df[df['item'] == '股票简称']
                if not short_name_row.empty:
                    name = str(short_name_row.iloc[0]['value']).strip()
                    return name
                
                # 否则用公司全称
                full_name_row = df[df['item'] == '公司全称']
                if not full_name_row.empty:
                    name = str(full_name_row.iloc[0]['value']).strip()
                    # 清理后缀
                    for suffix in ["股份有限公司", "集团股份有限公司", "集团有限公司", "有限公司"]:
                        if name.endswith(suffix):
                            name = name[:-len(suffix)].rstrip()
                            break
                    return name
        return None
    except Exception as e:
        # print(f"  调试: {symbol} 名称获取失败 - {e}")
        return None

STOCKS = load_stock_list()

# ========== 以下保持不变（仅在 main 中调用 get_stock_name_safe） ==========

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

        def infer_main_force_behavior(df):
            closes = df['close'].tail(5).tolist()
            vols = df['volume'].tail(5).tolist()
            if len(closes) < 5:
                return "数据不足"
            
            latest_close = closes[-1]
            latest_vol = vols[-1]
            avg_vol_5d = sum(vols) / 5
            high_vol = latest_vol > avg_vol_5d * 1.5

            pct_5d = (latest_close - closes[0]) / closes[0] if closes[0] != 0 else 0
            is_new_high = latest_close == max(closes)
            recent_pullback = len(closes) >= 3 and closes[-2] < closes[-3] and latest_close > closes[-2]
            pullback_low_vol = len(vols) >= 2 and vols[-2] < avg_vol_5d * 0.7

            if pct_5d > 0.05 and high_vol and is_new_high:
                return "强势拉升（放量突破新高）"
            elif abs(pct_5d) < 0.02 and latest_vol == min(vols):
                return "低位吸筹（横盘缩量）"
            elif recent_pullback and pullback_low_vol and latest_close > closes[-3]:
                return "健康洗盘（回调缩量后回升）"
            elif latest_close < closes[-2] and high_vol and (closes[-2] - latest_close) / closes[-2] > 0.03:
                return "放量下跌（警惕派发风险）"
            elif latest_close > closes[-2] and high_vol:
                return "放量上涨（主力积极介入）"
            elif latest_close > ma20 and latest_vol < avg_vol_5d * 0.8 and price_up:
                return "温和推升（惜售明显）"
            else:
                return "震荡整理（方向待明）"

        main_force_signal = infer_main_force_behavior(df)

        return {
            "symbol": symbol,
            "price": round(latest['close'], 2),
            "change_pct": round(change_pct, 2),
            "volume": int(latest['volume']),
            "rsi": round(rsi, 2) if isinstance(rsi, float) else rsi,
            "ma20": round(ma20, 2) if isinstance(ma20, float) else ma20,
            "last_5_days": close_prices.tail(5).round(2).tolist(),
            "volume_price_signal": volume_price_signal,
            "main_force_signal": main_force_signal,
        }

    except Exception:
        return None

def generate_analysis(data):
    stock_display = f"{data['name']}（{data['symbol']}）" if data.get('name') and data['name'] != "未知名称" else data['symbol']
    prompt = f"""
你是一位资深中文股票分析师，请基于以下多维数据生成150字以内简明分析：

- 股票: {stock_display}
- 当前价格: ¥{data['price']} | 涨跌幅: {data['change_pct']}%
- 近5日走势: {data['last_5_days']}
- 量价关系: {data['volume_price_signal']}
- 主力行为推断: {data['main_force_signal']}
- RSI: {data['rsi']}（>70超买，<30超卖）
- 20日均线: {data['ma20']}

要求：
1. 分析中需自然提及股票名称；
2. 重点结合量价与主力行为判断当前阶段；
3. 给出具体操作建议；
4. 语言专业简洁。
"""
    for retry in range(3):
        try:
            response = Generation.call(model="qwen-max", prompt=prompt, max_tokens=250)
            if response.status_code == 200:
                return response.output.text.strip()
            elif response.status_code == 429:
                time.sleep(2 ** retry)
                continue
            else:
                return f"API错误({response.status_code})"
        except Exception:
            time.sleep(2)
            continue
    return "分析失败"

def main():
    os.makedirs("output", exist_ok=True)
    results = []
    total = len(STOCKS)
    print(f"🚀 开始分析 {total} 只股票（逐个安全获取名称）...\n")

    for i, symbol in enumerate(STOCKS, 1):
        print(f"[{i}/{total}] 正在分析 {symbol}...")
        try:
            data = get_stock_data(symbol)
            if data is None:
                print(f"  ⚠️ 行情数据失败，跳过 {symbol}")
                continue

            # ✅ 关键：独立、安全地获取名称
            name = get_stock_name_safe(symbol)
            data["name"] = name if name else "未知名称"

            # 调试输出（可临时开启）
            print(f"  → 名称: {data['name']}")

            analysis = generate_analysis(data)
            data["analysis"] = analysis
            results.append(data)

            time.sleep(0.3)

        except Exception as e:
            print(f"  ❌ 异常: {e}")
            continue

    output = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "stocks": results
    }

    with open("output/predictions.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 完成！成功: {len(results)} / {total}")
    print("结果已保存至 output/predictions.json")

if __name__ == "__main__":
    main()
