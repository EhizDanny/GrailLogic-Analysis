import streamlit as st 
import plotly.graph_objects as go
import yfinance as yf
import datetime as dt
from datetime import datetime
from plotly.subplots import make_subplots
import pandas as pd

def getData():
    data = st.session_state['candleStickData']
    return data
data = getData()
ticker = st.session_state['tickers'][st.session_state['selectedData']]
year = data['Year'].iloc[0]
dataSelected = st.session_state['selectedData']

st.header(f"Chart For The Year Ended {data['Year'].iloc[0]}")
time_options=['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '4h', '1d', '5d', '1wk', '1mo', '3mo']
months = [
    "January", "February", "March", "April", "May",
    "June", "July", "August", "September",
    "October", "November", "December"
]
month_to_number = {
    "January": 1, "February": 2, "March": 3, "April": 4, "May": 5,
    "June": 6, "July": 7, "August": 8, "September": 9, "October": 10,
    "November": 11, "December": 12
}

bar1, bar2, bar3, bar4 = st.columns([1,1,1,1], gap='large')
bar11, bar12 = st.columns([3,1])
bar11.info('CAVEAT: Data of less than 60minutes can only be viewed for the last 2months worth of data')

current_month = datetime.now().month 
timeUnit = bar1.selectbox('Choose Time Unit', options=time_options, index=2)
month_sel = bar2.selectbox('Choose Month', options= months, index=current_month-1 )
showData = bar3.select_slider('Show Financial Data', options=['Show', 'Dont Show'], value='Dont Show')
filterForData = pd.to_datetime(f'{year}-{month_sel}-01')



mapTimeUnit = {
    'Minute': 'm',
    'Hour': 'h',
    'Day': 'd',
    'Week': 'wk',
    'Month': 'mo'
}

ticker_to_yfinance = {
    # --- Majors
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "USDJPY=X",
    "USDCHF": "USDCHF=X",
    "USDCAD": "USDCAD=X",
    "AUDUSD": "AUDUSD=X",
    "NZDUSD": "NZDUSD=X",

    # --- Popular crosses
    "EURGBP": "EURGBP=X", "EURJPY": "EURJPY=X", "EURCHF": "EURCHF=X", "EURAUD": "EURAUD=X", 
    "EURNZD": "EURNZD=X", "EURCAD": "EURCAD=X", "GBPJPY": "GBPJPY=X", "GBPCHF": "GBPCHF=X", "GBPAUD": "GBPAUD=X",
    "GBPNZD": "GBPNZD=X", "GBPCAD": "GBPCAD=X", "AUDJPY": "AUDJPY=X", "AUDNZD": "AUDNZD=X", "AUDCAD": "AUDCAD=X",
    "AUDCHF": "AUDCHF=X",  "NZDJPY": "NZDJPY=X",  "NZDCAD": "NZDCAD=X",  "NZDCHF": "NZDCHF=X",  "CADJPY": "CADJPY=X",
    "CADCHF": "CADCHF=X", "CHFJPY": "CHFJPY=X",

    # --- Metals / energy (common TV symbols)
    "XAUUSD": "XAUUSD=X", "XAGUSD": "XAGUSD=X","XTIUSD": "CL=F",  "XBRUSD": "BZ=F",    

    # --- Crypto (USD quotes)
    "BTCUSD": "BTC-USD", "ETHUSD": "ETH-USD", "SOLUSD": "SOL-USD", "XRPUSD": "XRP-USD", "BNBUSD": "BNB-USD",
    "ADAUSD": "ADA-USD", "DOGEUSD": "DOGE-USD", "LTCUSD": "LTC-USD", "BCHUSD": "BCH-USD", "DOTUSD": "DOT-USD",
    "AVAXUSD": "AVAX-USD", "MATICUSD": "MATIC-USD", "SHIBUSD": "SHIB-USD", "LINKUSD": "LINK-USD", "TRXUSD": "TRX-USD",
    "TONUSD": "TON-USD", "ATOMUSD": "ATOM-USD", "XLMUSD": "XLM-USD", "ETCUSD": "ETC-USD",
    }

def get_year_data(ticker: str, year: int, month: int, interval: str = "1d", auto_adjust: bool = True):
    start = dt.datetime(year, month, 1)
    if month == 12:
        stopMonth = 1
        stopYear = year + 1
    else:
        stopMonth = month + 1
        stopYear = year
    end   = dt.datetime(stopYear, stopMonth, 1)  # end is exclusive
    df = yf.download(
        tickers=ticker,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=auto_adjust,
        progress=False,
        threads=True, ignore_tz=True, multi_level_index=False
    )
    # Drop timezone on index if present (makes CSV/printing cleaner) 
    if df is None:
        return
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_convert(None) #  type: ignore
    return df

def plotData():
    df = get_year_data(ticker_to_yfinance[ticker], int(year), month_to_number[month_sel], timeUnit)
    if df is None:
        return
    originalDataSelected = st.session_state['upF'][dataSelected]
    originalDataSelected = pd.DataFrame(originalDataSelected)
    for cols in originalDataSelected.columns:
        if 'price' in cols.lower():
            originalDataSelected.rename(columns = {cols: 'price'}, inplace=True)
        if 'Net P&L' in cols and cols!= 'Net P&L %':
            originalDataSelected.rename(columns = {cols: 'Net P&L'}, inplace=True)
    # df = df.reset_index()
    # x = df["Datetime"] if "Datetime" in df.columns else df.index

    # # --- Trades data (trades) ---
    # # trades must have at least: "Date/Time", "Type" (Entry/Exit + Long/Short), "price"
    # tr = originalDataSelected.rename(columns={"Date/Time": "dt"}).copy()
    # tr["dt"] = pd.to_datetime(tr["dt"])
    # tr["price"] = pd.to_numeric(tr["price"], errors="coerce")
    # tr["is_entry"] = tr["Type"].str.contains(r"\bEntry\b", case=False, na=False)
    # tr["is_exit"]  = tr["Type"].str.contains(r"\bExit\b",  case=False, na=False)

    # ENTRY_COLOR = "rgba(46,204,113,0.95)"  # green
    # EXIT_COLOR  = "rgba(239,83,80,0.95)"   # red

    # # ---------- FIGURE (declared here) ----------
    # fig = make_subplots(specs=[[{"secondary_y": True}]])

    # # Candlestick on secondary y (price axis)
    # fig.add_trace(
    #     go.Candlestick(
    #         x=x,
    #         open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
    #         name="Price",
    #         increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
    #         increasing_fillcolor="#26a69a", decreasing_fillcolor="#ef5350",
    #     ),
    #     secondary_y=True
    # )

    # # Entry markers
    # entries = tr[tr["is_entry"]]
    # fig.add_trace(
    #     go.Scatter(
    #         x=entries["dt"], y=entries["price"],
    #         mode="markers",
    #         marker=dict(color=ENTRY_COLOR, size=9, symbol="triangle-up"),
    #         name="Entry"
    #     ),
    #     secondary_y=True
    # )

    # # Exit markers
    # exits = tr[tr["is_exit"]]
    # fig.add_trace(
    #     go.Scatter(
    #         x=exits["dt"], y=exits["price"],
    #         mode="markers",
    #         marker=dict(color=EXIT_COLOR, size=9, symbol="triangle-down"),
    #         name="Exit"
    #     ),
    #     secondary_y=True
    # )

    # # (Optional) draw entry/exit horizontal lines per trade in time order (simple pairing)
    # # Comment this block out if you don't want lines.
    # for side, sd in tr.sort_values("dt").groupby(tr["Type"].str.extract(r"(Long|Short)", expand=False)):
    #     stack = []
    #     for _, r in sd.iterrows():
    #         if r["is_entry"]:
    #             stack.append(r)
    #         elif r["is_exit"] and stack:
    #             e = stack.pop(0)  # pair first entry with next exit
    #             fig.add_shape(type="line", xref="x", yref="y2",
    #                         x0=e["dt"], x1=r["dt"], y0=e["price"], y1=e["price"],
    #                         line=dict(color=ENTRY_COLOR, width=2))
    #             fig.add_shape(type="line", xref="x", yref="y2",
    #                         x0=e["dt"], x1=r["dt"], y0=r["price"], y1=r["price"],
    #                         line=dict(color=EXIT_COLOR, width=2))

    # # Last price guide line
    # last = float(df["Close"].iloc[-1])
    # fig.add_hline(y=last, line_dash="dot", line_color="#8ab4f8",
    #             annotation_text=f"{last:.2f}", annotation_position="right",
    #             secondary_y=True)

    # fig.update_layout(
    #     template="plotly_dark",
    #     height=850,
    #     margin=dict(l=10, r=10, t=40, b=10),
    #     hovermode="x unified",
    #     xaxis=dict(
    #         showgrid=False,
    #         rangeslider=dict(visible=False),
    #         rangeselector=dict(buttons=[
    #             dict(count=5, label="5D", step="day", stepmode="backward"),
    #             dict(count=1, label="1M", step="month", stepmode="backward"),
    #             dict(count=3, label="3M", step="month", stepmode="backward"),
    #             dict(count=6, label="6M", step="month", stepmode="backward"),
    #             dict(count=1, label="1Y", step="year", stepmode="backward"),
    #             dict(step="all", label="All"),
    #         ]),
    #         # remove this for crypto/24x7
    #         rangebreaks=[dict(bounds=["sat","mon"])],
    #     ),
    #     yaxis=dict(title="Volume", showgrid=False),
    #     yaxis2=dict(title="Price", showgrid=True, gridcolor="rgba(255,255,255,0.08)")
    # )

    # # Streamlit
    # st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()

    # Volume bar colors: green if close >= open else red (with transparency)
    import numpy as np
    vol_color = np.where(df["Close"] >= df["Open"],
                        "rgba(38,166,154,0.6)",  # teal-ish
                        "rgba(239,83,80,0.6)")   # soft red

    # --- Figure with secondary y-axis for volume ---
    
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    if df.empty:
        return None
    df = df.reset_index()
    if 'Datetime' in df.columns:
        x = df["Datetime"]
    elif 'Date' in df.columns:
        x = df['Date']
    else:
        return False
    # Candles
    fig = go.Figure(go.Candlestick(
        x=x, #type:ignore
        open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name="Price",
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
        increasing_fillcolor="#26a69a", decreasing_fillcolor="#ef5350",

    ))

    # Last price guide line
    last = float(df["Close"].iloc[-1])
    fig.add_hline(y=last, line_dash="dot", line_color="#8ab4f8",
                annotation_text=f"{last:.2f}", annotation_position="right")

    fig.update_layout(
        template="plotly_dark",
        height=850,
        margin=dict(l=10, r=10, t=40, b=10),
        hovermode="x unified",
        xaxis=dict(
            showgrid=False,
            rangeslider=dict(visible=False),
            rangeselector=dict(buttons=[
                dict(count=5, label="5D", step="day", stepmode="backward"),
                dict(count=1, label="1M", step="month", stepmode="backward"),
                dict(count=3, label="3M", step="month", stepmode="backward"),
                dict(count=6, label="6M", step="month", stepmode="backward"),
                dict(count=1, label="1Y", step="year", stepmode="backward"),
                dict(step="all", label="All"),
            ]),
            # Hide weekends (nice for stocks/FX daily data)
            rangebreaks=[dict(bounds=["sat", "mon"])],
        ),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.08)")
    )

    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
    if showData == 'Show':
        st.dataframe(df)




if plotData() is None:
    st.warning('Readjust your selected time unit to atleast 60m. If that doesnt work, try higher timeframes')
elif plotData() is False:
    st.warning("Due to YFinance column names variations, the date and time column in this data is not the same that was expected. Click on 'Show Financial Data' to have a view of the data and confirm. Pls selected another year from the main page")
else:
    plotData()
data = None
del data