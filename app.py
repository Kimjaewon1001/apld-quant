import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import time

# --- 1. 기본 화면 설정 (가로로 넓게) ---
st.set_page_config(page_title="AI 퀀트 트레이딩 터미널", layout="wide")

# --- 2. 사이드바 (한국 주식 지원 및 설정) ---
st.sidebar.header("⚙️ 분석 설정")

market = st.sidebar.radio("🌍 시장 선택", ["🇺🇸 미국 주식", "🇰🇷 한국 주식 (코스피)", "🇰🇷 한국 주식 (코스닥)"])

if market == "🇺🇸 미국 주식":
    raw_ticker = st.sidebar.text_input("🔍 종목 코드 (예: AAPL, TSLA)", "AAPL").upper()
    ticker_symbol = raw_ticker
    currency = "$"
else:
    raw_ticker = st.sidebar.text_input("🔍 종목 코드 (6자리 숫자, 예: 005930)", "005930")
    ticker_symbol = f"{raw_ticker}.KS" if "코스피" in market else f"{raw_ticker}.KQ"
    currency = "₩"

interval_options = {'1분봉': '1m', '5분봉': '5m', '15분봉': '15m', '일봉': '1d', '주봉': '1wk'}
selected_interval_label = st.sidebar.selectbox("⏱️ 타임프레임 선택", list(interval_options.keys()), index=3)
interval = interval_options[selected_interval_label]

st.sidebar.divider()
st.sidebar.markdown("### 🔄 실시간 업데이트")
auto_refresh = st.sidebar.checkbox("1분마다 자동 새로고침 켜기", value=False)

# --- 3. 타이틀 및 한국 시간 ---
st.title(f"🚀 {raw_ticker} 실시간 퀀트 & AI 분석 터미널")
kst_now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9)))
st.markdown(f"**현재 접속 시각 (정확한 KST):** `{kst_now.strftime('%Y-%m-%d %H:%M:%S')}`")

# --- 4. 데이터 불러오기 ---
@st.cache_data(ttl=60)
def load_data(ticker, interval):
    try:
        if interval == '1m': period = "7d"
        elif interval in ['5m', '15m']: period = "60d"
        else: period = "1y"
        
        t = yf.Ticker(ticker)
        df = t.history(period=period, interval=interval)
        return df
    except:
        return pd.DataFrame()

df = load_data(ticker_symbol, interval)

# --- 5. 데이터 분석 (지표 및 매수/매도 시그널 포착) ---
if df.empty:
    st.error(f"❌ '{raw_ticker}' 데이터를 불러올 수 없습니다. 코드를 다시 확인해주세요.")
else:
    # 퀀트 지표 계산
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['BB_std'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['SMA_20'] + (df['BB_std'] * 2)
    df['BB_Lower'] = df['SMA_20'] - (df['BB_std'] * 2)
    
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / loss)))

    df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['VWAP'] = (df['Typical_Price'] * df['Volume']).rolling(window=20).sum() / df['Volume'].rolling(window=20).sum()
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()

    # --- [NEW] 매수 / 매도 타점(시그널) 자동 포착 로직 ---
    # 1. VWAP 돌파 시그널 (현재 종가가 VWAP 위로 올라갈 때 매수, 내려갈 때 매도)
    df['VWAP_Buy'] = (df['Close'] > df['VWAP']) & (df['Close'].shift(1) <= df['VWAP'].shift(1))
    df['VWAP_Sell'] = (df['Close'] < df['VWAP']) & (df['Close'].shift(1) >= df['VWAP'].shift(1))
    
    # 2. MACD 크로스 시그널 (골든크로스 매수, 데드크로스 매도)
    df['MACD_Buy'] = (df['MACD'] > df['Signal_Line']) & (df['MACD'].shift(1) <= df['Signal_Line'].shift(1))
    df['MACD_Sell'] = (df['MACD'] < df['Signal_Line']) & (df['MACD'].shift(1) >= df['Signal_Line'].shift(1))

    # 최근 데이터 변수화
    last = df.iloc[-1]
    prev_last = df.iloc[-2] if len(df) > 1 else last
    
    current_price = float(last['Close'])
    vwap_val = float(last['VWAP']) if pd.notna(last['VWAP']) else current_price
    rsi_val = float(last['RSI']) if pd.notna(last['RSI']) else 50
    macd_val = float(last['MACD']) if pd.notna(last['MACD']) else 0
    signal_val = float(last['Signal_Line']) if pd.notna(last['Signal_Line']) else 0
    obv_trend = float(last['OBV']) - float(prev_last['OBV']) if pd.notna(last['OBV']) else 0
    
    # 지지선, 저항선, 진입가, 목표가 계산
    support_line = float(df['Low'].rolling(window=20).min().iloc[-1])
    resistance_line = float(df['High'].rolling(window=20).max().iloc[-1])
    if support_line == resistance_line: 
        support_line = current_price * 0.95
        resistance_line = current_price * 1.05

    ai_entry_price = current_price
    if macd_val > signal_val and current_price > vwap_val:
        ai_take_profit = resistance_line + (resistance_line - support_line) * 0.382 
        ai_stop_loss = vwap_val if vwap_val < current_price else support_line
        action_signal = "🟢 적극 매수 가능"
    else:
        ai_take_profit = resistance_line 
        ai_stop_loss = support_line - (current_price - support_line) * 0.1
        action_signal = "🟡 관망 또는 보수적 단타"

    # --- [NEW] 화면 탭(Tab) 분리로 깔끔한 가독성 확보 ---
    tab1, tab2 = st.tabs(["📊 1. 퀀트 차트 & 지표 가이드", "🤖 2. AI 예측 및 리딩 리포트"])

    # ========== 탭 1: 차트 화면 및 바로 아래 지표 설명 ==========
    with tab1:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("현재 주가", f"{currency}{current_price:,.2f}")
        col2.metric("현재 RSI", f"{rsi_val:.2f}", "과열" if rsi_val > 70 else "침체" if rsi_val < 30 else "정상")
        col3.metric("VWAP (세력단가)", f"{currency}{vwap_val:,.2f}")
        col4.metric("AI 종합 시그널", action_signal)

        # 가독성을 극대화한 4단 차트 그리기
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.06, row_heights=[0.5, 0.15, 0.15, 0.2])

        # 1. 메인 차트 (주가, 이평선, 볼밴, VWAP)
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='주가'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], mode='lines', line=dict(color='cyan', width=1.5), name='20일선'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['VWAP'], mode='lines', line=dict(color='magenta', width=2.5), name='VWAP'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], mode='lines', line=dict(color='rgba(255,255,255,0.4)', width=1, dash='dot'), name='볼린저 상단'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], mode='lines', line=dict(color='rgba(255,255,255,0.4)', width=1, dash='dot'), name='볼린저 하단', fill='tonexty', fillcolor='rgba(128,128,128,0.1)'), row=1, col=1)
        
        # [NEW] VWAP 매수/매도 시그널 차트에 화살표로 찍기
        vwap_buys = df[df['VWAP_Buy']]
        vwap_sells = df[df['VWAP_Sell']]
        fig.add_trace(go.Scatter(x=vwap_buys.index, y=vwap_buys['Close']*0.98, mode='markers', marker=dict(symbol='triangle-up', size=12, color='lime'), name='VWAP 돌파(매수)'), row=1, col=1)
        fig.add_trace(go.Scatter(x=vwap_sells.index, y=vwap_sells['Close']*1.02, mode='markers', marker=dict(symbol='triangle-down', size=12, color='red'), name='VWAP 이탈(매도)'), row=1, col=1)

        # 2. MACD 차트 및 시그널
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], mode='lines', line=dict(color='cyan', width=2), name='MACD'), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Signal_Line'], mode='lines', line=dict(color='orange', width=2), name='Signal'), row=2, col=1)
        macd_buys = df[df['MACD_Buy']]
        macd_sells = df[df['MACD_Sell']]
        fig.add_trace(go.Scatter(x=macd_buys.index, y=macd_buys['MACD'], mode='markers', marker=dict(symbol='triangle-up', size=10, color='lime'), name='MACD 골든크로스'), row=2, col=1)
        fig.add_trace(go.Scatter(x=macd_sells.index, y=macd_sells['MACD'], mode='markers', marker=dict(symbol='triangle-down', size=10, color='red'), name='MACD 데드크로스'), row=2, col=1)
        
        # 3. RSI 차트
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], mode='lines', line=dict(color='yellow', width=2), name='RSI'), row=3, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="green", row=3, col=1)

        # 4. OBV 차트
        fig.add_trace(go.Scatter(x=df.index, y=df['OBV'], mode='lines', line=dict(color='teal', width=2.5), name='OBV'), row=4, col=1)

        fig.update_layout(height=1100, template="plotly_dark", title_text=f"{raw_ticker} 실시간 퀀트 차트 (시그널 포착 중)", xaxis_rangeslider_visible=False, showlegend=True, margin=dict(l=30, r=30, t=50, b=30))
        st.plotly_chart(fig, use_container_width=True)

        # --- [NEW] 차트 바로 밑에 지표 설명 (요청 사항 완벽 반영) ---
        st.markdown("""
        ---
        ### 📊 그래프 바로 밑 지표 완벽 가이드
        * **🟢 빨간/초록 화살표 (차트 위):** 컴퓨터가 알고리즘으로 계산한 **자동 매수(초록)/매도(빨간) 신호**입니다.
        * **📈 VWAP (자홍색 선, 1번 칸):** '세력과 기관의 평균 단가'입니다. 주가가 이 선 위로 뚫고 올라갈 때(초록 화살표)가 강력한 매수 타이밍입니다.
        * **🌊 OBV (청록색 선, 4번 칸):** '매집 지표'입니다. 주가는 떨어지는데 이 선이 올라간다면 누군가 조용히 쓸어 담고 있다는 뜻입니다.
        * **📉 MACD & RSI:** MACD 파란선이 위로 뚫으면 매수, RSI가 70 위면 과열(위험), 30 아래면 침체(기회)입니다.
        """)

    # ========== 탭 2: AI 예측 리포트 (숫자 콕 집어주기) ==========
    with tab2:
        st.subheader(f"🎯 AI 퀀트 트레이딩 전략 (현재가: {currency}{current_price:,.2f})")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🟢 적정 진입가", f"{currency}{ai_entry_price:,.2f}")
        c2.metric("🚀 1차 익절가 (목표)", f"{currency}{ai_take_profit:,.2f}")
        c3.metric("🚨 손절가 (위험)", f"{currency}{ai_stop_loss:,.2f}")
        c4.metric("🧱 주요 저항 / 지지", f"{currency}{resistance_line:,.2f} / {currency}{support_line:,.2f}")

        st.info(f"""
        ### 🤖 {raw_ticker} AI 실시간 차트 리딩 리포트
        현재 차트의 **실제 데이터 값**을 읽어 AI가 해석한 결과입니다.
        
        * **세력 단가(VWAP):** 현재가({currency}{current_price:,.2f})가 VWAP({currency}{vwap_val:,.2f})보다 **{'높습니다. 매수세가 강해 상승 추세를 이끌고 있습니다.' if current_price > vwap_val else '낮습니다. 매물대 저항을 받아 억눌려 있습니다.'}**
        * **매집량(OBV):** 누군가 **{'조용히 사 모으고(매집) 있는 긍정적 흐름' if obv_trend > 0 else '점진적으로 물량을 넘기는(분산) 부정적 흐름' if obv_trend < 0 else '뚜렷한 움직임이 없는 횡보장'}**이 포착되었습니다.
        * **과열도(RSI):** 현재 `{rsi_val:.2f}`입니다. **{'70 이상 과매수 상태로, 단기 하락 조정이 올 수 있어 위험합니다.' if rsi_val > 70 else '30 이하 과매도 상태로, 반등을 노린 저가 매수 기회일 수 있습니다.' if rsi_val < 30 else '30~70 사이 안정적 구간입니다.'}**
        * **단기 방향성(MACD):** 단기선이 장기선을 **{'상향 돌파(골든크로스)해 상승 모멘텀이 켜졌습니다.' if macd_val > signal_val else '하향 이탈(데드크로스)해 하락 모멘텀이 강해졌습니다.'}**
        """)

# --- 6. 자동 업데이트 (가장 마지막에 실행) ---
if auto_refresh:
    time.sleep(60)
    st.rerun()