import os
import shutil
import certifi
import datetime

# 🚨 [핵심 패치] 한글 경로 SSL 인증서 인식 오류 자동 해결 로직
cert_path = certifi.where()
safe_cert_dir = r"C:\apld_certs"
safe_cert_path = os.path.join(safe_cert_dir, "cacert.pem")

if not os.path.exists(safe_cert_dir):
    os.makedirs(safe_cert_dir, exist_ok=True)
if not os.path.exists(safe_cert_path):
    shutil.copy2(cert_path, safe_cert_path)

os.environ['CURL_CA_BUNDLE'] = safe_cert_path
os.environ['REQUESTS_CA_BUNDLE'] = safe_cert_path
os.environ['SSL_CERT_FILE'] = safe_cert_path

# --- 여기서부터 본 대시보드 코드 시작 ---
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="궁극의 퀀트 분석 대시보드", layout="wide")

# --- 사이드바 설정 (검색 및 설정) ---
st.sidebar.header("⚙️ 분석 설정")
ticker_symbol = st.sidebar.text_input("🔍 종목 코드 검색 (예: APLD, AAPL, NVDA)", "APLD").upper()

interval_options = {'1분봉': '1m', '5분봉': '5m', '15분봉': '15m', '일봉': '1d', '주봉': '1wk'}
selected_interval_label = st.sidebar.selectbox("⏱️ 타임프레임 선택", list(interval_options.keys()), index=3)
interval = interval_options[selected_interval_label]

st.title(f"🚀 {ticker_symbol} 실시간 퀀트 분석 및 예측 대시보드")

# --- 데이터 로드 함수 ---
@st.cache_data(ttl=60) # 1분마다 캐시 갱신
def load_data(ticker, interval):
    # interval에 따라 가져올 수 있는 최대 기간 설정
    if interval == '1m': period = "7d"
    elif interval in ['5m', '15m']: period = "60d"
    else: period = "1y"
    
    t = yf.Ticker(ticker)
    df = t.history(period=period, interval=interval)
    
    info = t.info
    return df, info

df, info = load_data(ticker_symbol, interval)

if df.empty:
    st.error(f"❌ {ticker_symbol} 데이터를 불러올 수 없습니다. 종목 코드를 다시 확인해 주세요.")
else:
    # --- 1. 퀀트 보조지표 계산 (5가지 이상) ---
    
    # ① 이동평균선 (Moving Averages)
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_60'] = df['Close'].rolling(window=60).mean()
    
    # ② 볼린저 밴드 (Bollinger Bands)
    df['BB_std'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['SMA_20'] + (df['BB_std'] * 2)
    df['BB_Lower'] = df['SMA_20'] - (df['BB_std'] * 2)
    
    # ③ MACD (추세)
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # ④ RSI (과매수/과매도)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # ⑤ 스토캐스틱 (Stochastic Oscillator)
    low_14 = df['Low'].rolling(window=14).min()
    high_14 = df['High'].rolling(window=14).max()
    df['%K'] = (df['Close'] - low_14) * 100 / (high_14 - low_14)
    df['%D'] = df['%K'].rolling(window=3).mean()

    # --- 2. 퀀트 시그널 종합 평가 (방향 가늠) ---
    last = df.iloc[-1]
    signals = 0
    
    if last['Close'] > last['SMA_20']: signals += 1
    elif last['Close'] < last['SMA_20']: signals -= 1
        
    if last['MACD'] > last['Signal_Line']: signals += 1
    elif last['MACD'] < last['Signal_Line']: signals -= 1
        
    if last['RSI'] < 30: signals += 1 # 과매도(반등 예상)
    elif last['RSI'] > 70: signals -= 1 # 과매수(하락 예상)
        
    if last['%K'] > last['%D']: signals += 1
    elif last['%K'] < last['%D']: signals -= 1

    if signals >= 2: overall_signal = "🟢 강력 매수 (상승 추세)"
    elif signals <= -2: overall_signal = "🔴 강력 매도 (하락 추세)"
    else: overall_signal = "🟡 관망 (횡보/변동성 주의)"

    # --- UI 화면 구성 ---
    current_price = last['Close']
    last_time = df.index[-1].strftime("%Y-%m-%d %H:%M:%S")
    
    st.markdown(f"**기준 시각:** `{last_time}` (해당 거래소 현지 시간 기준)")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("현재 주가", f"${current_price:.2f}")
    col2.metric("AI 퀀트 시그널", overall_signal)
    col3.metric("RSI (과열도)", f"{last['RSI']:.2f}", "과매수" if last['RSI']>70 else "과매도" if last['RSI']<30 else "정상")
    col4.metric("MACD 상태", "골든 크로스" if last['MACD'] > last['Signal_Line'] else "데드 크로스")

    st.divider()

    # --- 3. 종합 차트 그리기 (Plotly Subplots) ---
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.6, 0.2, 0.2])

    # [Row 1] 캔들 차트 + 이동평균선 + 볼린저 밴드
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='주가'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], mode='lines', line=dict(color='blue', width=1.5), name='20일선'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], mode='lines', line=dict(color='gray', width=1, dash='dot'), name='볼린저 상단'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], mode='lines', line=dict(color='gray', width=1, dash='dot'), name='볼린저 하단', fill='tonexty', fillcolor='rgba(128,128,128,0.1)'), row=1, col=1)

    # [Row 2] MACD
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], mode='lines', line=dict(color='blue', width=1.5), name='MACD'), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Signal_Line'], mode='lines', line=dict(color='orange', width=1.5), name='Signal'), row=2, col=1)
    
    # [Row 3] RSI
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], mode='lines', line=dict(color='purple', width=1.5), name='RSI'), row=3, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="green", row=3, col=1)

    fig.update_layout(height=800, title_text=f"{ticker_symbol} {selected_interval_label} 종합 차트", xaxis_rangeslider_visible=False, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

    # 지표 설명
    with st.expander("📊 추가된 5가지 퀀트 지표 보는 법 (클릭해서 펼치기)"):
        st.markdown("""
        * **이동평균선 (20일/60일):** 주가의 평균적인 흐름입니다. 주가가 파란색 20일선 위에 있으면 상승 추세입니다.
        * **볼린저 밴드 (회색 영역):** 주가가 움직이는 '도로'입니다. 밴드가 좁아지면(응축) 곧 큰 변동(폭발)이 온다는 뜻이며, 주가가 상단 밖으로 튀어나가면 일시적 고점일 확률이 높습니다.
        * **MACD (중간 차트):** 파란 선이 주황 선을 뚫고 올라가면 매수(골든크로스), 뚫고 내려가면 매도(데드크로스) 시그널입니다.
        * **RSI (맨 아래 차트):** 70 이상이면 사람들이 너무 많이 사서(과열) 떨어질 확률이 높고, 30 이하이면 너무 많이 팔아서 반등할 확률이 높습니다.
        * **스토캐스틱:** 단기적인 고점/저점을 파악하는 데 쓰이며, AI 종합 시그널 계산에 포함되었습니다.
        """)