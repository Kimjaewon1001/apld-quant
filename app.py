import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import time

# --- 1. 기본 설정 ---
st.set_page_config(page_title="궁극의 퀀트 분석 대시보드", layout="wide")

# --- 2. 사이드바 (종목 검색 및 자동 갱신 스위치) ---
st.sidebar.header("⚙️ 분석 설정")
ticker_symbol = st.sidebar.text_input("🔍 종목 검색 (예: APLD, AAPL, NVDA)", "APLD").upper()

interval_options = {'1분봉': '1m', '5분봉': '5m', '15분봉': '15m', '일봉': '1d', '주봉': '1wk'}
selected_interval_label = st.sidebar.selectbox("⏱️ 타임프레임 선택", list(interval_options.keys()), index=3)
interval = interval_options[selected_interval_label]

st.sidebar.divider()
st.sidebar.markdown("### 🔄 실시간 업데이트")
# [수정 완료] 자동 갱신이 안 되던 문제를 해결하는 스위치!
auto_refresh = st.sidebar.checkbox("1분마다 화면 자동 새로고침 켜기", value=False)

st.title(f"🚀 {ticker_symbol} 실시간 퀀트 & AI 예측 대시보드")

# --- 3. [수정 완료] 정확한 한국 현재 시간 표시 ---
kst_now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9)))
st.markdown(f"**현재 접속 시각 (정확한 한국 KST):** `{kst_now.strftime('%Y-%m-%d %H:%M:%S')}`")

# --- 4. 데이터 불러오기 (에러 방지 & 최신화) ---
@st.cache_data(ttl=60) # 60초가 지나면 무조건 새로운 데이터를 가져오게 강제 설정
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

# --- 5. 데이터 분석 시작 ---
if df.empty:
    st.error(f"❌ {ticker_symbol} 데이터를 불러올 수 없습니다. 종목 코드를 확인해주세요.")
else:
    # --- [추가 완료] 모든 퀀트 보조지표 계산 ---
    
    # ① 이동평균선 및 볼린저 밴드
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['BB_std'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['SMA_20'] + (df['BB_std'] * 2)
    df['BB_Lower'] = df['SMA_20'] - (df['BB_std'] * 2)
    
    # ② MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # ③ RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / loss)))

    # ④ [신규 추가] VWAP (거래량 가중 평균 가격 - 세력 단가)
    df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['VWAP'] = (df['Typical_Price'] * df['Volume']).rolling(window=20).sum() / df['Volume'].rolling(window=20).sum()

    # ⑤ [신규 추가] OBV (온밸런스 볼륨 - 매집량 지표)
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()

    # --- [복구 완료] AI 퀀트 방향성 예측 ---
    last = df.iloc[-1]
    signals = 0
    
    if last['Close'] > last['SMA_20']: signals += 1
    elif last['Close'] < last['SMA_20']: signals -= 1
        
    if last['MACD'] > last['Signal_Line']: signals += 1
    elif last['MACD'] < last['Signal_Line']: signals -= 1
        
    if pd.notna(last['RSI']):
        if last['RSI'] < 30: signals += 1
        elif last['RSI'] > 70: signals -= 1

    # VWAP 기준 AI 시그널 추가 판단
    if pd.notna(last['VWAP']):
        if last['Close'] > last['VWAP']: signals += 1
        elif last['Close'] < last['VWAP']: signals -= 1

    if signals >= 2: overall_signal = "🟢 강력 매수 (상승 추세)"
    elif signals <= -2: overall_signal = "🔴 강력 매도 (하락 추세)"
    else: overall_signal = "🟡 관망 (방향성 탐색)"

    # --- 대시보드 상단 요약 ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("현재 주가", f"${last['Close']:.2f}")
    col2.metric("AI 퀀트 시그널", overall_signal)
    
    rsi_val = last['RSI']
    col3.metric("RSI (과열도)", f"{rsi_val:.2f}" if pd.notna(rsi_val) else "N/A", "과매수" if pd.notna(rsi_val) and rsi_val > 70 else "과매도" if pd.notna(rsi_val) and rsi_val < 30 else "정상")
    col4.metric("VWAP 상태", "단가 위 (강세)" if pd.notna(last['VWAP']) and last['Close'] > last['VWAP'] else "단가 아래 (약세)")

    st.divider()

    # --- 6. [추가 완료] 4단 종합 차트 그리기 (VWAP & OBV 포함) ---
    # 차트 칸을 3개에서 4개로 늘렸습니다.
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.45, 0.15, 0.15, 0.25])

    # [1번 칸] 주가 + 20일선 + 볼린저 밴드 + VWAP
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='주가'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], mode='lines', line=dict(color='blue', width=1), name='20일선'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['VWAP'], mode='lines', line=dict(color='magenta', width=2), name='VWAP (세력단가)'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], mode='lines', line=dict(color='gray', width=1, dash='dot'), name='볼린저 상단'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], mode='lines', line=dict(color='gray', width=1, dash='dot'), name='볼린저 하단', fill='tonexty', fillcolor='rgba(128,128,128,0.1)'), row=1, col=1)

    # [2번 칸] MACD
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], mode='lines', line=dict(color='blue', width=1.5), name='MACD'), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Signal_Line'], mode='lines', line=dict(color='orange', width=1.5), name='Signal'), row=2, col=1)
    
    # [3번 칸] RSI
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], mode='lines', line=dict(color='purple', width=1.5), name='RSI'), row=3, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="green", row=3, col=1)

    # [4번 칸] OBV (새로 추가된 매집량 지표)
    fig.add_trace(go.Scatter(x=df.index, y=df['OBV'], mode='lines', line=dict(color='teal', width=2), name='OBV (매집량)'), row=4, col=1)

    fig.update_layout(height=1000, title_text=f"{ticker_symbol} 실시간 분석 차트", xaxis_rangeslider_visible=False, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

    # --- 7. [추가 완료] 그래프 바로 밑에 친절한 지표 설명 넣기 ---
    st.markdown("""
    ### 📊 차트 지표 해석 가이드 (초보자용)
    
    * **📈 VWAP (자홍색 선, 1번 차트): '세력과 기관의 진짜 평균 단가'**
      * 거래량을 포함해서 계산한 진짜 평균 가격이야. **현재 주가가 VWAP 선 위에 있으면** 기관들이 수익 중이므로 상승 흐름이 강하다는 뜻이고, 반대로 주가가 VWAP 아래로 떨어지면 위험 신호야.
      
    * **🌊 OBV (청록색 선, 맨 아래 4번 차트): '누군가 몰래 사고 있는지 확인하는 매집 지표'**
      * 주가가 오를 때의 거래량은 더하고, 내릴 때의 거래량은 뺀 값이야. 만약 **주가는 횡보하거나 떨어지는데 OBV 선이 계속 위로 올라간다면?** 누군가 가격을 억누르며 조용히 주식을 쓸어 담고(매집) 있다는 강력한 매수 힌트가 돼!
      
    * **📉 MACD & RSI (2번, 3번 차트): '추세와 과열 상태'**
      * **MACD:** 파란 선이 주황 선을 뚫고 올라가면(골든크로스) 매수 타이밍이야.
      * **RSI:** 70 위로 가면 사람들이 너무 많이 사서 곧 떨어질 확률이 높고, 30 아래로 가면 너무 많이 팔아서 반등할 확률이 높아.
    """)

# --- 8. [복구 및 수정 완료] 1분 자동 업데이트 기능 ---
# 사용자가 사이드바에서 체크박스를 켜면, 이 코드가 작동해서 60초마다 화면을 새로고침(rerun) 합니다.
if auto_refresh:
    time.sleep(60)
    st.rerun()