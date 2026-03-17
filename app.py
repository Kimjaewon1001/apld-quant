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

# --- 2. 사이드바 (종목 검색 및 자동 갱신) ---
st.sidebar.header("⚙️ 분석 설정")
ticker_symbol = st.sidebar.text_input("🔍 종목 검색 (예: APLD, AAPL, NVDA)", "APLD").upper()

interval_options = {'1분봉': '1m', '5분봉': '5m', '15분봉': '15m', '일봉': '1d', '주봉': '1wk'}
selected_interval_label = st.sidebar.selectbox("⏱️ 타임프레임 선택", list(interval_options.keys()), index=3)
interval = interval_options[selected_interval_label]

st.sidebar.divider()
st.sidebar.markdown("### 🔄 실시간 업데이트")
auto_refresh = st.sidebar.checkbox("1분마다 화면 자동 새로고침 켜기", value=False)

st.title(f"🚀 {ticker_symbol} 실시간 퀀트 & AI 예측 대시보드")

# --- 3. 정확한 한국 현재 시간 ---
kst_now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9)))
st.markdown(f"**현재 접속 시각 (정확한 한국 KST):** `{kst_now.strftime('%Y-%m-%d %H:%M:%S')}`")

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

# --- 5. 데이터 분석 시작 ---
if df.empty:
    st.error(f"❌ {ticker_symbol} 데이터를 불러올 수 없습니다.")
else:
    # 퀀트 보조지표 계산 (SMA, BB, MACD, RSI, VWAP, OBV)
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

    # 최근 데이터 추출 (변수화)
    last = df.iloc[-1]
    prev_last = df.iloc[-2] if len(df) > 1 else last
    
    current_price = float(last['Close'])
    vwap_val = float(last['VWAP']) if pd.notna(last['VWAP']) else current_price
    rsi_val = float(last['RSI']) if pd.notna(last['RSI']) else 50
    macd_val = float(last['MACD']) if pd.notna(last['MACD']) else 0
    signal_val = float(last['Signal_Line']) if pd.notna(last['Signal_Line']) else 0
    obv_trend = float(last['OBV']) - float(prev_last['OBV']) if pd.notna(last['OBV']) else 0

    # AI 퀀트 방향성 예측 점수
    signals = 0
    if current_price > last['SMA_20']: signals += 1
    elif current_price < last['SMA_20']: signals -= 1
    if macd_val > signal_val: signals += 1
    elif macd_val < signal_val: signals -= 1
    if rsi_val < 30: signals += 1
    elif rsi_val > 70: signals -= 1
    if current_price > vwap_val: signals += 1
    elif current_price < vwap_val: signals -= 1

    if signals >= 2: overall_signal = "🟢 강력 매수 (상승 추세 전환/유지)"
    elif signals <= -2: overall_signal = "🔴 강력 매도 (하락 압력 주의)"
    else: overall_signal = "🟡 관망 (횡보 및 방향성 탐색)"

    # --- 대시보드 상단 요약 ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("현재 주가", f"${current_price:.2f}")
    col2.metric("AI 종합 시그널", overall_signal)
    col3.metric("현재 RSI", f"{rsi_val:.2f}", "과열(위험)" if rsi_val > 70 else "침체(기회)" if rsi_val < 30 else "정상")
    col4.metric("세력 단가 (VWAP)", f"${vwap_val:.2f}", "돌파 강세" if current_price > vwap_val else "이탈 약세")

    # --- [NEW] 실제 값을 리딩하는 AI 해설 박스 ---
    st.info(f"""
    ### 🤖 {ticker_symbol} AI 실시간 차트 분석 리포트
    현재 차트의 실제 데이터를 바탕으로 AI가 분석한 결과입니다.
    
    * **세력 단가(VWAP) 분석:** 현재 주가(`${current_price:.2f}`)가 시장 참여자들의 평균 단가인 VWAP(`${vwap_val:.2f}`)보다 **{'높습니다. 매수세가 시장을 주도하고 있습니다.' if current_price > vwap_val else '낮습니다. 매도 압력이 강해 저항을 받고 있습니다.'}**
    * **매집량(OBV) 추세:** 최근 거래량을 동반한 매수/매도 흐름을 볼 때, 누군가 주식을 **{'조용히 사 모으고(매집) 있는 긍정적인 흐름입니다.' if obv_trend > 0 else '점진적으로 팔아치우고(분산) 있는 부정적인 흐름입니다.' if obv_trend < 0 else '뚜렷한 매집/분산 움직임이 없습니다.'}**
    * **RSI (과열도):** 현재 수치는 `{rsi_val:.2f}`입니다. **{'70을 초과한 과매수 상태로, 곧 차익 실현 매물이 쏟아져 하락할 위험이 큽니다.' if rsi_val > 70 else '30 미만인 과매도 상태로, 지나치게 많이 빠져있어 저가 매수를 노려볼 만한 반등 타점입니다.' if rsi_val < 30 else '30~70 사이의 안정적인 구간에서 힘을 모으고 있습니다.'}**
    * **MACD 추세:** 단기 추세선이 장기 추세선을 **{'상향 돌파(골든크로스)하여 상승 모멘텀이 강해지고 있습니다.' if macd_val > signal_val else '하향 이탈(데드크로스)하여 하락 모멘텀이 짙어지고 있습니다.'}**
    """)

    st.divider()

    # --- 6. 4단 종합 차트 그리기 ---
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.45, 0.15, 0.15, 0.25])

    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='주가'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], mode='lines', line=dict(color='blue', width=1), name='20일선'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['VWAP'], mode='lines', line=dict(color='magenta', width=2), name='VWAP (세력단가)'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], mode='lines', line=dict(color='gray', width=1, dash='dot'), name='볼린저 상단'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], mode='lines', line=dict(color='gray', width=1, dash='dot'), name='볼린저 하단', fill='tonexty', fillcolor='rgba(128,128,128,0.1)'), row=1, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], mode='lines', line=dict(color='blue', width=1.5), name='MACD'), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Signal_Line'], mode='lines', line=dict(color='orange', width=1.5), name='Signal'), row=2, col=1)
    
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], mode='lines', line=dict(color='purple', width=1.5), name='RSI'), row=3, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="green", row=3, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df['OBV'], mode='lines', line=dict(color='teal', width=2), name='OBV (매집량)'), row=4, col=1)

    fig.update_layout(height=1000, title_text=f"{ticker_symbol} 실시간 분석 차트", xaxis_rangeslider_visible=False, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

    # --- 7. 그래프 밑 지표 기본 설명 ---
    st.markdown("""
    ### 📊 차트 지표 기본 개념 (참고용)
    * **📈 VWAP (자홍색 선, 1번 차트):** '세력과 기관의 평균 단가'입니다. 주가가 이 선 위에 있어야 상승세가 유지됩니다.
    * **🌊 OBV (청록색 선, 4번 차트):** 거래량을 바탕으로 한 '매집 지표'입니다. 주가는 횡보하는데 OBV가 오르면 누군가 몰래 사고 있다는 뜻입니다.
    * **📉 MACD & RSI:** MACD는 추세의 방향(골든/데드크로스)을, RSI는 과열(70 이상)과 침체(30 이하)를 보여줍니다.
    """)

# --- 8. 1분 자동 업데이트 기능 ---
if auto_refresh:
    time.sleep(60)
    st.rerun()