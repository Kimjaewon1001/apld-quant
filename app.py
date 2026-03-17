import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import time

# --- 1. 기본 화면 설정 (넓고 시원하게) ---
st.set_page_config(page_title="궁극의 퀀트 분석 대시보드", layout="wide")

# --- 2. 사이드바 (종목 검색 및 1분 자동 갱신) ---
st.sidebar.header("⚙️ 분석 설정")
ticker_symbol = st.sidebar.text_input("🔍 종목 검색 (예: APLD, AAPL, NVDA)", "APLD").upper()

interval_options = {'1분봉': '1m', '5분봉': '5m', '15분봉': '15m', '일봉': '1d', '주봉': '1wk'}
selected_interval_label = st.sidebar.selectbox("⏱️ 타임프레임 선택", list(interval_options.keys()), index=3)
interval = interval_options[selected_interval_label]

st.sidebar.divider()
st.sidebar.markdown("### 🔄 실시간 업데이트")
# [기능 확인 완료] 이 스위치를 켜면 60초마다 화면이 자동으로 새로고침 됩니다.
auto_refresh = st.sidebar.checkbox("1분마다 화면 자동 새로고침 켜기", value=False)

st.title(f"🚀 {ticker_symbol} 실시간 퀀트 & AI 목표가 예측 대시보드")

# --- 3. [기능 확인 완료] 정확한 한국 현재 시간 ---
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

# --- 5. 데이터 분석 및 AI 예측 로직 ---
if df.empty:
    st.error(f"❌ {ticker_symbol} 데이터를 불러올 수 없습니다. 종목 코드를 확인해주세요.")
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

    # --- [핵심 기능] AI 주가 예측 (지지/저항/진입/손절/익절가 계산) ---
    last = df.iloc[-1]
    prev_last = df.iloc[-2] if len(df) > 1 else last
    
    current_price = float(last['Close'])
    vwap_val = float(last['VWAP']) if pd.notna(last['VWAP']) else current_price
    rsi_val = float(last['RSI']) if pd.notna(last['RSI']) else 50
    macd_val = float(last['MACD']) if pd.notna(last['MACD']) else 0
    signal_val = float(last['Signal_Line']) if pd.notna(last['Signal_Line']) else 0
    obv_trend = float(last['OBV']) - float(prev_last['OBV']) if pd.notna(last['OBV']) else 0
    
    # 20일 기준 지지선(최저가) 및 저항선(최고가) 도출
    support_line = float(df['Low'].rolling(window=20).min().iloc[-1])
    resistance_line = float(df['High'].rolling(window=20).max().iloc[-1])
    
    if support_line == resistance_line: # 데이터가 부족할 경우 예외 처리
        support_line = current_price * 0.95
        resistance_line = current_price * 1.05

    # AI 트레이딩 타점 계산 (피보나치 및 변동성 기반)
    ai_entry_price = current_price # 현재가를 기본 진입가로 설정
    if macd_val > signal_val and current_price > vwap_val:
        trend_status = "상승(강세) 추세"
        # 강세장: 저항선을 뚫고 올라갈 확률이 높으므로 목표가를 저항선 위로 설정
        ai_take_profit = resistance_line + (resistance_line - support_line) * 0.382 
        ai_stop_loss = vwap_val if vwap_val < current_price else support_line
        action_signal = "🟢 적극 매수 가능"
    else:
        trend_status = "하락/조정(약세) 추세"
        # 약세장: 지지선 부근에서 짧게 먹고 빠지거나, 더 아래를 손절가로 설정
        ai_take_profit = resistance_line # 목표가를 이전 고점(저항선)으로 보수적 설정
        ai_stop_loss = support_line - (current_price - support_line) * 0.1
        action_signal = "🟡 관망 또는 보수적 접근"

    # --- 대시보드 상단 요약 (명시적 가격 안내) ---
    st.subheader(f"🎯 AI 퀀트 트레이딩 전략 (현재가: ${current_price:.2f})")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("💡 AI 추천 액션", action_signal)
    col2.metric("🟢 적정 진입가", f"${ai_entry_price:.2f}")
    col3.metric("🚀 1차 익절가 (목표)", f"${ai_take_profit:.2f}")
    col4.metric("🚨 손절가 (위험)", f"${ai_stop_loss:.2f}")
    col5.metric("🧱 주요 저항 / 지지", f"${resistance_line:.2f} / ${support_line:.2f}")

    # --- [기능 확인 완료] 실제 값을 읽고 해석해주는 AI 리포트 ---
    st.info(f"""
    ### 🤖 {ticker_symbol} AI 실시간 차트 분석 및 리딩 리포트
    현재 차트에서 추출한 **실제 데이터 값**을 바탕으로 AI가 분석한 결과입니다.
    
    * **세력 단가(VWAP) 분석:** 현재 주가(`${current_price:.2f}`)가 시장 참여자들의 평균 단가인 VWAP(`${vwap_val:.2f}`)보다 **{'높습니다. 세력과 기관이 수익권에 있어 매수세가 시장을 주도하고 있습니다.' if current_price > vwap_val else '낮습니다. 매물대 저항을 받아 주가가 짓눌려 있는 상태입니다.'}**
    * **매집량(OBV) 추세:** 거래량을 분석한 결과, 누군가 주식을 **{'조용히 사 모으고(매집) 있는 긍정적인 흐름이 포착되었습니다.' if obv_trend > 0 else '점진적으로 팔아치우고(분산) 있는 부정적인 흐름이 포착되었습니다.' if obv_trend < 0 else '뚜렷한 매집/분산 움직임이 멈춘 횡보 상태입니다.'}**
    * **과열도(RSI):** 현재 RSI 수치는 `{rsi_val:.2f}`입니다. **{'70을 초과한 과매수 상태입니다. 단기적으로 차익 실현 매물이 쏟아질 수 있으니 신규 매수는 위험합니다.' if rsi_val > 70 else '30 미만인 과매도 상태입니다. 낙폭 과대로 인한 단기 기술적 반등이 나올 수 있는 기회입니다.' if rsi_val < 30 else '30~70 사이의 안정적인 구간에서 힘을 모으고 있습니다.'}**
    * **단기 방향성(MACD):** 단기 추세선이 장기 추세선을 **{'상향 돌파(골든크로스)하여 상승 모멘텀이 발생했습니다.' if macd_val > signal_val else '하향 이탈(데드크로스)하여 하락 모멘텀이 짙어지고 있습니다.'}**
    """)

    st.divider()

    # --- 6. [기능 확인 완료] 가독성 극대화 4단 종합 차트 ---
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.5, 0.15, 0.15, 0.2])

    # 1번 차트: 캔들, 이동평균선, VWAP, 볼린저밴드
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='주가'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], mode='lines', line=dict(color='cyan', width=1.5), name='20일선'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['VWAP'], mode='lines', line=dict(color='magenta', width=2.5), name='VWAP (세력단가)'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], mode='lines', line=dict(color='rgba(255, 255, 255, 0.4)', width=1, dash='dot'), name='볼린저 상단'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], mode='lines', line=dict(color='rgba(255, 255, 255, 0.4)', width=1, dash='dot'), name='볼린저 하단', fill='tonexty', fillcolor='rgba(128,128,128,0.1)'), row=1, col=1)
    
    # 지지/저항선 시각적 표시
    fig.add_hline(y=resistance_line, line_dash="dash", line_color="red", annotation_text="단기 저항선", row=1, col=1)
    fig.add_hline(y=support_line, line_dash="dash", line_color="green", annotation_text="단기 지지선", row=1, col=1)

    # 2번 차트: MACD
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], mode='lines', line=dict(color='cyan', width=2), name='MACD'), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Signal_Line'], mode='lines', line=dict(color='orange', width=2), name='Signal'), row=2, col=1)
    
    # 3번 차트: RSI
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], mode='lines', line=dict(color='yellow', width=2), name='RSI'), row=3, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="green", row=3, col=1)

    # 4번 차트: OBV
    fig.add_trace(go.Scatter(x=df.index, y=df['OBV'], mode='lines', line=dict(color='teal', width=2.5), name='OBV (매집량)'), row=4, col=1)

    # 어두운 배경(다크모드)과 차트 높이(1200) 설정으로 가독성 완벽 개선
    fig.update_layout(
        height=1200, 
        template="plotly_dark", 
        title_text=f"{ticker_symbol} 실시간 퀀트 차트 (가독성 최적화 완료)", 
        xaxis_rangeslider_visible=False, 
        showlegend=True,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- 7. [기능 확인 완료] 그래프 바로 밑 친절한 지표 설명 ---
    st.markdown("""
    ---
    ### 📊 차트 지표 완벽 가이드 (차트 순서대로 설명)
    
    **[1번 차트: 가격 및 주요 선]**
    * **📈 VWAP (자홍색 굵은 선):** 거래량을 포함한 '세력과 기관의 진짜 평균 단가'입니다. 주가가 이 선 위에 머물러야 튼튼한 상승세입니다.
    * **☁️ 볼린저 밴드 (회색 영역):** 주가가 움직이는 도로입니다. 주가가 위쪽 점선(상단)을 뚫으면 단기 고점일 확률이, 아래 점선(하단)을 뚫으면 단기 저점일 확률이 높습니다.
    * **🔴 저항선 / 🟢 지지선:** 주가가 올라가다 부딪히는 천장(저항)과, 떨어지다 튕겨 오르는 바닥(지지)을 가리킵니다.

    **[2번 차트: 추세 방향성]**
    * **📉 MACD:** 파란 선이 주황 선을 뚫고 올라가면 매수 타이밍(골든크로스), 뚫고 내려가면 매도 타이밍(데드크로스)입니다.

    **[3번 차트: 과열도]**
    * **🔥 RSI:** 주가가 얼마나 뜨거운지 보여줍니다. 70 위로 가면 너무 많이 사서 위험한 상태, 30 아래로 가면 사람들이 너무 많이 팔아서 싸진 상태를 의미합니다.

    **[4번 차트: 세력 매집]**
    * **🌊 OBV (청록색 굵은 선):** 주가가 제자리걸음을 하거나 떨어지는데, 이 선이 위로 올라간다면? 누군가 가격을 누르면서 조용히 물량을 쓸어 담고 있다는 강력한 힌트(매집)입니다.
    """)

# --- 8. [기능 확인 완료] 1분 자동 업데이트 기능 ---
if auto_refresh:
    time.sleep(60) # 60초를 세고 난 뒤
    st.rerun()     # 화면을 스스로 새로고침합니다.