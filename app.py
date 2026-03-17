import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 1. 화면 기본 설정
st.set_page_config(page_title="궁극의 퀀트 분석 대시보드", layout="wide")

# 2. 사이드바 (검색 및 설정 메뉴)
st.sidebar.header("⚙️ 분석 설정")
ticker_symbol = st.sidebar.text_input("🔍 종목 검색 (예: APLD, AAPL, NVDA)", "APLD").upper()

# 10분봉은 야후 무료 API에서 지원하지 않아 가장 유사한 15분봉으로 대체했습니다.
interval_options = {'1분봉': '1m', '5분봉': '5m', '15분봉': '15m', '일봉': '1d', '주봉': '1wk'}
selected_interval_label = st.sidebar.selectbox("⏱️ 타임프레임 선택", list(interval_options.keys()), index=3)
interval = interval_options[selected_interval_label]

st.title(f"🚀 {ticker_symbol} 실시간 퀀트 분석 및 AI 예측 대시보드")

# 3. 데이터 불러오기 (에러 방지를 위해 방어적으로 작성)
@st.cache_data(ttl=60) # 1분마다 새로고침
def load_data(ticker, interval):
    try:
        if interval == '1m': period = "7d"
        elif interval in ['5m', '15m']: period = "60d"
        else: period = "1y"
        
        t = yf.Ticker(ticker)
        df = t.history(period=period, interval=interval)
        return df
    except:
        return pd.DataFrame() # 에러 시 빈 데이터 반환

df = load_data(ticker_symbol, interval)

# 4. 데이터가 잘 들어왔을 때만 분석 시작
if df.empty:
    st.error(f"❌ {ticker_symbol} 데이터를 불러올 수 없습니다. 종목 코드를 확인하거나 잠시 후 다시 시도해주세요.")
else:
    # --- [기능 추가 1] 5가지 퀀트 보조지표 계산 ---
    
    # ① 이동평균선 (SMA: 주가의 평균 흐름)
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    
    # ② 볼린저 밴드 (주가가 움직이는 도로의 폭)
    df['BB_std'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['SMA_20'] + (df['BB_std'] * 2)
    df['BB_Lower'] = df['SMA_20'] - (df['BB_std'] * 2)
    
    # ③ MACD (단기 추세와 장기 추세의 차이)
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # ④ RSI (상대강도지수: 주가가 얼마나 과열되었는가?)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / loss)))
    
    # ⑤ 스토캐스틱 (현재 주가가 최근 변동폭 중 어디쯤 있나?)
    low_14 = df['Low'].rolling(window=14).min()
    high_14 = df['High'].rolling(window=14).max()
    df['%K'] = (df['Close'] - low_14) * 100 / (high_14 - low_14)
    df['%D'] = df['%K'].rolling(window=3).mean()

    # --- [기능 추가 2] AI 퀀트 방향성 가늠 (점수화 시스템) ---
    last = df.iloc[-1]
    signals = 0 # 0점에서 시작
    
    # 지표별 조건에 따라 점수 +1 (매수 신호) 또는 -1 (매도 신호) 부여
    if last['Close'] > last['SMA_20']: signals += 1 # 주가가 20일선 위에 있으면 긍정적
    elif last['Close'] < last['SMA_20']: signals -= 1
        
    if last['MACD'] > last['Signal_Line']: signals += 1 # 골든크로스
    elif last['MACD'] < last['Signal_Line']: signals -= 1 # 데드크로스
        
    if pd.notna(last['RSI']):
        if last['RSI'] < 30: signals += 1 # 너무 많이 빠져서 반등 예상 (과매도)
        elif last['RSI'] > 70: signals -= 1 # 너무 많이 올라서 하락 예상 (과매수)
        
    if pd.notna(last['%K']) and pd.notna(last['%D']):
        if last['%K'] > last['%D']: signals += 1
        elif last['%K'] < last['%D']: signals -= 1

    # 최종 점수로 AI 시그널 판별
    if signals >= 2: overall_signal = "🟢 강력 매수 (상승 추세 기대)"
    elif signals <= -2: overall_signal = "🔴 강력 매도 (하락 추세 주의)"
    else: overall_signal = "🟡 관망 (방향성 탐색 중)"

    # --- 화면에 결과 보여주기 ---
    current_price = float(last['Close'])
    last_time = df.index[-1].strftime("%Y-%m-%d %H:%M:%S")
    
    st.markdown(f"**기준 시각:** `{last_time}` (해당 거래소 현지 시간 기준)")
    
    # 상단 요약 대시보드
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("현재 주가", f"${current_price:.2f}")
    col2.metric("AI 퀀트 시그널", overall_signal)
    
    rsi_val = last['RSI']
    rsi_status = "과매수" if rsi_val > 70 else "과매도" if rsi_val < 30 else "정상"
    col3.metric("RSI (과열도)", f"{rsi_val:.2f}", rsi_status)
    col4.metric("MACD 상태", "골든 크로스" if last['MACD'] > last['Signal_Line'] else "데드 크로스")

    st.divider()

    # --- [기능 추가 3] 다중 지표 종합 차트 그리기 ---
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.6, 0.2, 0.2])

    # [1번 칸] 캔들차트 + 20일선 + 볼린저 밴드
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='주가'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], mode='lines', line=dict(color='blue', width=1.5), name='20일선'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], mode='lines', line=dict(color='gray', width=1, dash='dot'), name='볼린저 상단'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], mode='lines', line=dict(color='gray', width=1, dash='dot'), name='볼린저 하단', fill='tonexty', fillcolor='rgba(128,128,128,0.1)'), row=1, col=1)

    # [2번 칸] MACD 차트
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], mode='lines', line=dict(color='blue', width=1.5), name='MACD'), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Signal_Line'], mode='lines', line=dict(color='orange', width=1.5), name='Signal'), row=2, col=1)
    
    # [3번 칸] RSI 차트
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], mode='lines', line=dict(color='purple', width=1.5), name='RSI'), row=3, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="green", row=3, col=1)

    fig.update_layout(height=800, title_text=f"{ticker_symbol} {selected_interval_label} 종합 차트", xaxis_rangeslider_visible=False, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)