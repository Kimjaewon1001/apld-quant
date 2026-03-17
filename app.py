import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import time

# --- 1. 화면 기본 설정 ---
st.set_page_config(page_title="궁극의 퀀트 분석 대시보드", layout="wide")

# --- 2. 사이드바 (검색 및 자동 갱신 설정) ---
st.sidebar.header("⚙️ 분석 설정")
ticker_symbol = st.sidebar.text_input("🔍 종목 검색 (예: APLD, AAPL, NVDA)", "APLD").upper()

interval_options = {'1분봉': '1m', '5분봉': '5m', '15분봉': '15m', '일봉': '1d', '주봉': '1wk'}
selected_interval_label = st.sidebar.selectbox("⏱️ 타임프레임 선택", list(interval_options.keys()), index=3)
interval = interval_options[selected_interval_label]

st.sidebar.divider()
# [핵심 추가] 자동 갱신 토글 버튼
auto_refresh = st.sidebar.checkbox("🔄 1분마다 실시간 자동 갱신", value=False)

st.title(f"🚀 {ticker_symbol} 실시간 퀀트 분석 및 AI 예측 대시보드")

# --- [핵심 추가] 한국 기준 현재 시각 표시 ---
# 서버가 미국에 있어도 무조건 한국 시간(KST)으로 계산해서 보여줍니다.
kst_time = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9)))
st.markdown(f"**조회 시각 (한국 KST):** `{kst_time.strftime('%Y-%m-%d %H:%M:%S')}`")

# --- 3. 데이터 불러오기 ---
@st.cache_data(ttl=60) # 60초마다 새로운 데이터 가져오기 허용
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

# --- 4. 데이터 분석 및 AI 시그널 ---
if df.empty:
    st.error(f"❌ {ticker_symbol} 데이터를 불러올 수 없습니다. 종목 코드를 확인하거나 잠시 후 다시 시도해주세요.")
else:
    # 5가지 퀀트 보조지표 계산
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
    
    low_14 = df['Low'].rolling(window=14).min()
    high_14 = df['High'].rolling(window=14).max()
    df['%K'] = (df['Close'] - low_14) * 100 / (high_14 - low_14)
    df['%D'] = df['%K'].rolling(window=3).mean()

    # AI 퀀트 방향성 가늠 (점수화 시스템)
    last = df.iloc[-1]
    signals = 0
    
    if last['Close'] > last['SMA_20']: signals += 1
    elif last['Close'] < last['SMA_20']: signals -= 1
        
    if last['MACD'] > last['Signal_Line']: signals += 1
    elif last['MACD'] < last['Signal_Line']: signals -= 1
        
    if pd.notna(last['RSI']):
        if last['RSI'] < 30: signals += 1
        elif last['RSI'] > 70: signals -= 1
        
    if pd.notna(last['%K']) and pd.notna(last['%D']):
        if last['%K'] > last['%D']: signals += 1
        elif last['%K'] < last['%D']: signals -= 1

    if signals >= 2: overall_signal = "🟢 강력 매수 (상승 추세 기대)"
    elif signals <= -2: overall_signal = "🔴 강력 매도 (하락 추세 주의)"
    else: overall_signal = "🟡 관망 (방향성 탐색 중)"

    # 화면에 결과 보여주기
    current_price = float(last['Close'])
    last_trade_time = df.index[-1].strftime("%Y-%m-%d %H:%M:%S")
    
    st.caption(f"💡 마지막 거래 체결 시각 (현지 거래소 기준): {last_trade_time}")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("현재 주가", f"${current_price:.2f}")
    col2.metric("AI 퀀트 시그널", overall_signal)
    
    rsi_val = last['RSI']
    rsi_status = "과매수" if pd.notna(rsi_val) and rsi_val > 70 else "과매도" if pd.notna(rsi_val) and rsi_val < 30 else "정상"
    col3.metric("RSI (과열도)", f"{rsi_val:.2f}" if pd.notna(rsi_val) else "N/A", rsi_status)
    col4.metric("MACD 상태", "골든 크로스" if pd.notna(last['MACD']) and last['MACD'] > last['Signal_Line'] else "데드 크로스")

    st.divider()

    # 다중 지표 종합 차트 그리기
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.6, 0.2, 0.2])

    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='주가'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], mode='lines', line=dict(color='blue', width=1.5), name='20일선'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], mode='lines', line=dict(color='gray', width=1, dash='dot'), name='볼린저 상단'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], mode='lines', line=dict(color='gray', width=1, dash='dot'), name='볼린저 하단', fill='tonexty', fillcolor='rgba(128,128,128,0.1)'), row=1, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], mode='lines', line=dict(color='blue', width=1.5), name='MACD'), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Signal_Line'], mode='lines', line=dict(color='orange', width=1.5), name='Signal'), row=2, col=1)
    
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], mode='lines', line=dict(color='purple', width=1.5), name='RSI'), row=3, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="green", row=3, col=1)

    fig.update_layout(height=800, title_text=f"{ticker_symbol} {selected_interval_label} 종합 차트", xaxis_rangeslider_visible=False, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

# --- [핵심 추가] 1분 자동 갱신 로직 ---
if auto_refresh:
    time.sleep(60) # 60초 대기
    st.rerun()     # 화면 새로고침 (최신 데이터 다시 불러옴)