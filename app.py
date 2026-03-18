import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import datetime
import time
import ssl

# --- [필수] 윈도우 SSL 인증서 에러 방지 ---
ssl._create_default_https_context = ssl._create_unverified_context

# --- 1. 환경 설정 (전체화면) ---
st.set_page_config(page_title="재원 전용 퀀트 AI 터미널", layout="wide")

# --- 2. 사이드바 (시장 및 종목 선택) ---
st.sidebar.header("⚙️ 시스템 설정")
market = st.sidebar.radio("🌍 시장 선택", ["🇺🇸 미국 주식", "🇰🇷 한국 주식 (코스피)", "🇰🇷 한국 주식 (코스닥)"])

if market == "🇺🇸 미국 주식":
    raw_ticker = st.sidebar.text_input("🔍 종목 코드 (예: AAPL, TSLA)", "AAPL").upper()
    ticker_symbol = raw_ticker
    currency = "$"
else:
    raw_ticker = st.sidebar.text_input("🔍 종목 번호 (6자리, 예: 005930)", "005930")
    ticker_symbol = f"{raw_ticker}.KS" if "코스피" in market else f"{raw_ticker}.KQ"
    currency = "₩"

interval_dict = {'1분봉': '1m', '5분봉': '5m', '15분봉': '15m', '일봉': '1d', '주봉': '1wk'}
interval = interval_dict[st.sidebar.selectbox("⏱️ 타임프레임", list(interval_dict.keys()), index=3)]

st.sidebar.divider()
auto_refresh = st.sidebar.checkbox("🔄 1분 자동 업데이트 켜기", value=True)

# 숫자 포맷팅 함수 (한국 주식은 소수점 없이, 미국 주식은 소수점 2자리)
def fmt(val):
    return f"₩{val:,.0f}" if currency == "₩" else f"${val:,.2f}"

# --- 3. 헤더 및 정확한 KST 시간 ---
st.title(f"🚀 {raw_ticker} 실시간 퀀트 시그널 & AI 예측 터미널")
kst_now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9)))
st.markdown(f"**⏰ 현재 분석 시각 (정확한 KST):** `{kst_now.strftime('%Y-%m-%d %H:%M:%S')}`")

# --- 4. 데이터 로드 ---
@st.cache_data(ttl=60)
def load_data(ticker, inv):
    try:
        p = "7d" if inv == '1m' else "60d" if inv in ['5m', '15m'] else "2y"
        return yf.download(ticker, period=p, interval=inv, progress=False)
    except:
        return pd.DataFrame()

df = load_data(ticker_symbol, interval)

if df.empty or len(df) < 30:
    st.error("❌ 데이터를 불러올 수 없습니다. 종목 코드가 맞는지 확인해주세요.")
else:
    # --- 5. 퀀트 지표 계산 ---
    close = df['Close']
    df['SMA20'] = close.rolling(20).mean()
    df['BB_Upper'] = df['SMA20'] + (close.rolling(20).std() * 2)
    df['BB_Lower'] = df['SMA20'] - (close.rolling(20).std() * 2)
    
    # VWAP 계산
    df['VWAP'] = ((df['High']+df['Low']+df['Close'])/3 * df['Volume']).rolling(20).sum() / df['Volume'].rolling(20).sum()
    
    # MACD 계산
    exp1, exp2 = close.ewm(span=12).mean(), close.ewm(span=26).mean()
    df['MACD'] = exp1 - exp2
    df['Signal'] = df['MACD'].ewm(span=9).mean()
    
    # RSI 계산
    delta = close.diff()
    up, down = delta.clip(lower=0).rolling(14).mean(), -delta.clip(upper=0).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + up/down))
    
    # OBV 계산
    df['OBV'] = (np.sign(close.diff()) * df['Volume']).fillna(0).cumsum()

    # --- 6. [핵심] 자동 매수/매도 시그널 로직 ---
    # ① VWAP 시그널 (돌파시 매수, 이탈시 매도)
    df['VWAP_Buy'] = (close > df['VWAP']) & (close.shift(1) <= df['VWAP'].shift(1))
    df['VWAP_Sell'] = (close < df['VWAP']) & (close.shift(1) >= df['VWAP'].shift(1))
    
    # ② MACD 시그널 (골든크로스 매수, 데드크로스 매도)
    df['MACD_Buy'] = (df['MACD'] > df['Signal']) & (df['MACD'].shift(1) <= df['Signal'].shift(1))
    df['MACD_Sell'] = (df['MACD'] < df['Signal']) & (df['MACD'].shift(1) >= df['Signal'].shift(1))
    
    # ③ RSI 시그널 (30 상향돌파 매수, 70 하향이탈 매도)
    df['RSI_Buy'] = (df['RSI'] > 30) & (df['RSI'].shift(1) <= 30)
    df['RSI_Sell'] = (df['RSI'] < 70) & (df['RSI'].shift(1) >= 70)

    # --- 7. 현재 값 및 AI 예측 로직 (변동성 시뮬레이션) ---
    curr_p = float(close.iloc[-1])
    vwap_p = float(df['VWAP'].iloc[-1])
    rsi_p = float(df['RSI'].iloc[-1])
    macd_p = float(df['MACD'].iloc[-1])
    sig_p = float(df['Signal'].iloc[-1])
    
    # 과거 20일 기준 지지/저항선
    supp = float(df['Low'].tail(20).min())
    resi = float(df['High'].tail(20).max())
    if supp == resi: supp, resi = curr_p * 0.95, curr_p * 1.05

    # 변동성 기반 미래 주가 예측 (Drift + Volatility Model)
    ret = df['Close'].pct_change().fillna(0)
    mu, vol = ret.mean(), ret.std()
    if pd.isna(vol) or vol == 0: vol = 0.02
    
    pred_1d = curr_p * (1 + mu*1 + vol*np.sqrt(1))
    pred_1w = curr_p * (1 + mu*5 + vol*np.sqrt(5))
    pred_1m = curr_p * (1 + mu*20 + vol*np.sqrt(20))

    # --- 8. 화면 구성 (탭 분리) ---
    tab1, tab2 = st.tabs(["📊 1. 분리형 퀀트 차트 & 지표 가이드", "🤖 2. AI 미래 주가 예측 & 전략 리포트"])

    # ==========================================
    # 탭 1: 완전 분리형 차트 및 개별 설명
    # ==========================================
    with tab1:
        st.subheader("💡 상단 요약 데이터")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("현재 주가", fmt(curr_p))
        c2.metric("세력 평균단가 (VWAP)", fmt(vwap_p))
        c3.metric("현재 RSI (과열도)", f"{rsi_p:.2f}")
        c4.metric("현재 추세 (MACD)", "상승(골든)" if macd_p > sig_p else "하락(데드)")

        st.divider()

        # --- [차트 1] 주가 & VWAP ---
        st.markdown("### 1️⃣ 주가 흐름 및 세력선(VWAP) 분석")
        fig1 = go.Figure()
        fig1.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="주가"))
        fig1.add_trace(go.Scatter(x=df.index, y=df['VWAP'], line=dict(color='magenta', width=2.5), name="VWAP(세력선)"))
        fig1.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], line=dict(color='rgba(255,255,255,0.3)', dash='dot'), name="볼린저 상단"))
        fig1.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], line=dict(color='rgba(255,255,255,0.3)', dash='dot'), name="볼린저 하단"))
        
        # VWAP 시그널 매수/매도 화살표
        fig1.add_trace(go.Scatter(x=df[df['VWAP_Buy']].index, y=df[df['VWAP_Buy']]['Close']*0.98, mode='markers', marker=dict(symbol='triangle-up', size=14, color='lime'), name='VWAP 매수'))
        fig1.add_trace(go.Scatter(x=df[df['VWAP_Sell']].index, y=df[df['VWAP_Sell']]['Close']*1.02, mode='markers', marker=dict(symbol='triangle-down', size=14, color='red'), name='VWAP 매도'))
        
        fig1.update_layout(height=450, template="plotly_dark", xaxis_rangeslider_visible=False, margin=dict(t=30, b=30))
        st.plotly_chart(fig1, use_container_width=True)
        st.info("💡 **[해석 가이드]** 자홍색 선은 기관/세력의 진짜 평균 단가(VWAP)입니다. 주가가 이 선을 뚫고 올라가며 **🔺초록 화살표**가 뜰 때가 가장 확실한 **매수 타이밍**입니다. 반대로 깨고 내려가면 도망쳐야 합니다.")

        # --- [차트 2] MACD ---
        st.markdown("### 2️⃣ 방향성(MACD) 분석")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df.index, y=df['MACD'], line=dict(color='cyan', width=2), name="MACD"))
        fig2.add_trace(go.Scatter(x=df.index, y=df['Signal'], line=dict(color='orange', width=2), name="Signal"))
        
        # MACD 시그널 매수/매도 화살표
        fig2.add_trace(go.Scatter(x=df[df['MACD_Buy']].index, y=df[df['MACD_Buy']]['MACD'], mode='markers', marker=dict(symbol='triangle-up', size=12, color='lime'), name='골든크로스(매수)'))
        fig2.add_trace(go.Scatter(x=df[df['MACD_Sell']].index, y=df[df['MACD_Sell']]['MACD'], mode='markers', marker=dict(symbol='triangle-down', size=12, color='red'), name='데드크로스(매도)'))
        
        fig2.update_layout(height=300, template="plotly_dark", margin=dict(t=30, b=30))
        st.plotly_chart(fig2, use_container_width=True)
        st.info("💡 **[해석 가이드]** 파란선(MACD)이 주황선(Signal)을 위로 뚫을 때(골든크로스, 🔺초록 화살표)는 상승 추세가 시작된다는 뜻입니다.")

        # --- [차트 3] RSI ---
        st.markdown("### 3️⃣ 과열도(RSI) 분석")
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='yellow', width=2), name="RSI"))
        fig3.add_hline(y=70, line_dash="dot", line_color="red")
        fig3.add_hline(y=30, line_dash="dot", line_color="green")
        
        # RSI 시그널 화살표
        fig3.add_trace(go.Scatter(x=df[df['RSI_Buy']].index, y=df[df['RSI_Buy']]['RSI'], mode='markers', marker=dict(symbol='triangle-up', size=12, color='lime'), name='과매도 탈출(매수)'))
        fig3.add_trace(go.Scatter(x=df[df['RSI_Sell']].index, y=df[df['RSI_Sell']]['RSI'], mode='markers', marker=dict(symbol='triangle-down', size=12, color='red'), name='과매수 이탈(매도)'))

        fig3.update_layout(height=300, template="plotly_dark", margin=dict(t=30, b=30))
        st.plotly_chart(fig3, use_container_width=True)
        st.info("💡 **[해석 가이드]** 70 위는 사람들이 너무 많이 산 상태(위험), 30 아래는 너무 많이 판 상태(기회)입니다. 30을 찍고 반등할 때(🔺초록 화살표)가 저점 매수 타점입니다.")

        # --- [차트 4] OBV ---
        st.markdown("### 4️⃣ 세력 매집량(OBV) 분석")
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=df.index, y=df['OBV'], line=dict(color='teal', width=2.5), name="OBV"))
        fig4.update_layout(height=300, template="plotly_dark", margin=dict(t=30, b=30))
        st.plotly_chart(fig4, use_container_width=True)
        st.info("💡 **[해석 가이드]** 주가는 횡보하거나 떨어지는데 OBV 선만 혼자 우상향하고 있다면? 누군가 가격을 누르면서 몰래 주식을 쓸어 담고 있다는 강력한 힌트입니다.")


    # ==========================================
    # 탭 2: AI 미래 주가 예측 및 리포트 (숫자 명시)
    # ==========================================
    with tab2:
        st.header(f"🎯 AI 시뮬레이션: {raw_ticker} 미래 주가 예측")
        st.write("과거 데이터의 평균 수익률과 일일 변동성(Volatility)을 수학적으로 계산하여 도출한 예측값입니다.")
        
        # 1일, 1주, 1달 예측치 (가장 크게 명시)
        p1, p2, p3 = st.columns(3)
        p1.metric("📅 1일 뒤 예상 주가", fmt(pred_1d))
        p2.metric("📅 1주 뒤 예상 주가", fmt(pred_1w))
        p3.metric("📅 1달 뒤 예상 주가", fmt(pred_1m))
        
        st.divider()
        
        st.header("🧱 전략적 타점 계산 (진입 / 익절 / 손절)")
        t1, t2, t3, t4 = st.columns(4)
        t1.metric("🟢 권장 진입가", fmt(curr_p))
        t2.metric("🚀 1차 목표가 (익절)", fmt(resi))
        t3.metric("🚨 최후 방어선 (손절)", fmt(supp))
        t4.metric("📊 단기 저항대", fmt(resi * 1.02))

        st.divider()

        # 실제 차트 값을 파이썬이 읽어서 해석하는 AI 리포트
        st.success(f"""
        ### 🤖 실시간 AI 퀀트 리딩 리포트
        
        현재 차트의 **실제 수치**를 AI가 분석한 전략 요약입니다.
        
        * **1. 세력 단가(VWAP) 비교:** 현재 주가({fmt(curr_p)})가 시장 평균 단가인 VWAP({fmt(vwap_p)})보다 **{'위에 있습니다. 매수세가 강해 상승을 주도하고 있으며 보유를 권장합니다.' if curr_p > vwap_p else '아래에 있습니다. 매물대 저항을 받아 약세 흐름이므로 섣부른 매수는 위험합니다.'}**
        * **2. 단기 방향성(MACD):** MACD 수치({macd_p:.2f})가 시그널 선({sig_p:.2f})을 **{'상향 돌파하여 단기 상승 모멘텀이 터졌습니다.' if macd_p > sig_p else '하향 이탈하여 하락 압력이 커지고 있습니다.'}**
        * **3. 과열도(RSI):** 현재 RSI는 `{rsi_p:.2f}`입니다. **{'70을 넘어 매우 과열되었습니다. 분할 매도로 수익을 챙기세요.' if rsi_p > 70 else '30 미만으로 바닥 구간입니다. 기술적 반등을 노린 저점 매수가 유효합니다.' if rsi_p < 30 else '30~70 사이의 안정적인 박스권 흐름을 보이고 있습니다.'}**
        * **4. 종합 행동 지침:** 과거 20일 변동성을 고려할 때 하방으로는 **{fmt(supp)}**가 뚫리면 전량 손절해야 하며, 상방으로는 **{fmt(resi)}**를 뚫어낼 시 1달 내에 **{fmt(pred_1m)}**까지의 우상향 랠리를 기대해 볼 수 있습니다.
        """)

# --- 9. 자동 갱신 ---
if auto_refresh:
    time.sleep(60)
    st.rerun()