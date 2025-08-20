''' ------------------------------------------------------------------------ 
delta_hedge_sim.py

Interactive Delta Hedging Simulation with Streamlit

This application simulates the delta hedging of options positions over time,
demonstrating how traders manage risk by continuously adjusting their hedge
ratios. The simulator uses Black-Scholes pricing to calculate option values
and deltas, while allowing users to specify different implied vs realized
volatilities to observe the impact on hedging performance.

Features:
  • Real-time option pricing using Black-Scholes model
  • Daily delta hedging adjustments with transaction tracking
  • Interactive visualization of stock price paths and P/L evolution
  • Simulation summary with performance metrics
  • Detailed results table showing daily hedging activity
  • Support for both call and put options in long/short positions

Educational Purpose: Demonstrates the practical challenges of delta hedging
including the impact of volatility differences, transaction costs (implicitly),
and the discrete nature of hedging adjustments.

Author: Alexis D. Plascencia
Date: July 21, 2025
License: MIT
------------------------------------------------------------------------ '''

import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt

st.set_page_config(page_title='Delta Hedging Simulation', layout='centered')

# Custom CSS for black background theme
st.markdown("""
<style>
    .stApp {
        background-color: #000000;
        color: #FFFFFF;
    }
    
    .stApp > header {
        background-color: transparent;
    }
    
    .stApp > div[data-testid="stToolbar"] {
        background-color: #000000;
    }
    
    /* Main content area */
    .main .block-container {
        background-color: #000000;
        color: #FFFFFF;
    }
    
    /* Input widgets */
    .stNumberInput > div > div > input {
        background-color: #1a1a1a;
        color: #FFFFFF;
        border: 1px solid #444444;
    }
    
    .stNumberInput label {
        color: #FFFFFF !important;
    }
    
    .stSelectbox > div > div > div {
        background-color: #1a1a1a;
        color: #FFFFFF;
    }
    
    .stSelectbox label {
        color: #FFFFFF !important;
    }
    
    /* All input labels */
    label {
        color: #FFFFFF !important;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #333333;
        color: #FFFFFF;
        border: 1px solid #555555;
    }
    
    .stButton > button:hover {
        background-color: #555555;
        border: 1px solid #777777;
    }
    
    /* Success/Info messages */
    .stSuccess {
        background-color: #1a4a1a;
        color: #FFFFFF;
    }
    
    .stInfo {
        background-color: #1a1a4a;
        color: #FFFFFF;
    }
    
    /* Dataframes */
    .stDataFrame {
        background-color: #1a1a1a;
        color: #FFFFFF;
    }
    
    /* Subheaders */
    .stSubheader {
        color: #FFFFFF;
    }
    
    /* Caption */
    .stCaption {
        color: #CCCCCC;
    }
</style>
""", unsafe_allow_html=True)

# ---------- Black–Scholes helpers ------------------------------------------------
def _d1(S, K, T, r, sigma):
    return (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T) + 1e-12)

def _d2(S, K, T, r, sigma):
    return _d1(S, K, T, r, sigma) - sigma * np.sqrt(T)

def bs_price(S, K, T, r, sigma, option_type='call'):
    if T <= 0:
        return max(S - K, 0.0) if option_type == 'call' else max(K - S, 0.0)
    d1 = _d1(S, K, T, r, sigma)
    d2 = _d2(S, K, T, r, sigma)
    if option_type == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def bs_delta(S, K, T, r, sigma, option_type='call'):
    if T <= 0:
        return 1.0 if (option_type == 'call' and S > K) else -1.0 if (option_type == 'put' and S < K) else 0.0
    d1 = _d1(S, K, T, r, sigma)
    return norm.cdf(d1) if option_type == 'call' else norm.cdf(d1) - 1

# ---------- UI -------------------------------------------------------------------
st.title('Delta Hedging Simulation ⚡📈')

# Define the input parameters for the simulation
col1, col2, col3, col4 = st.columns(4)
with col1:
    S0 = st.number_input('Initial Stock Price ($)', min_value=1.0, value=100.0, step=1.0)
with col2:
    K = st.number_input('Strike Price ($)', min_value=1.0, value=100.0, step=1.0)
with col3:
    days_to_expiry = st.number_input('Days to Expiry', min_value=1, value=5, step=1)
with col4:
    implied_vol_pct = st.number_input('Implied Volatility (%)', min_value=1.0, value=20.0, step=1.0)

col5, col6, col7 = st.columns(3)
with col5:
    realized_vol_pct = st.number_input('Realized Volatility (%)', min_value=1.0, value=30.0, step=1.0)
with col6:
    option_type = st.selectbox('Option Type', options=['Call', 'Put'], index=0)
with col7:
    option_position = st.selectbox('Option Position', options=['Long', 'Short'], index=0)

# Buttons for simulation control
init_btn, next_btn, reset_btn = st.columns(3)
initialize = init_btn.button('Initialize Simulation')
next_day   = next_btn.button('Next Day')
reset_sim  = reset_btn.button('Reset Simulation')

# ---------- Stateful data --------------------------------------------------------
def reset_state():
    st.session_state.day = 0
    st.session_state.S_path = [S0]
    st.session_state.deltas = []
    st.session_state.hedge_shares = []
    st.session_state.cash = 0.0
    st.session_state.pnl_cum = [0.0]
    st.session_state.pnl_daily = []
    st.session_state.pnl_stock_daily = []
    st.session_state.pnl_option_daily = []
    st.session_state.option_prices = []     # unsigned (always positive)
    st.session_state.trades = []            # each: day, shares, price

if 'day' not in st.session_state:
    reset_state()

# --- constants ---
r = 0.0
dt = 1.0 / 252.0
sigma_realized = realized_vol_pct / 100.0
sigma_implied  = implied_vol_pct / 100.0
option_type_lower = option_type.lower()
pos_sign = 1.0 if option_position.lower() == 'long' else -1.0

# ---------- Initialise -----------------------------------------------------------
if initialize:
    reset_state()

    T = days_to_expiry / 252.0
    option_price0 = bs_price(S0, K, T, r, sigma_implied, option_type_lower)
    delta0 = bs_delta(S0, K, T, r, sigma_implied, option_type_lower)

    shares_to_trade = -pos_sign * delta0
    st.session_state.cash = -shares_to_trade * S0
    st.session_state.trades.append({'day': 0, 'shares': shares_to_trade, 'price': S0})

    # store paths
    st.session_state.option_prices.append(option_price0)
    st.session_state.deltas.append(delta0)
    st.session_state.hedge_shares.append(shares_to_trade)

    st.success('Simulation initialised. Press **Next Day** to step forward.')

# ---------- Next day -------------------------------------------------------------
if next_day and st.session_state.day < days_to_expiry:
    S_prev = st.session_state.S_path[-1]
    shares_prev = st.session_state.hedge_shares[-1]
    option_price_prev = st.session_state.option_prices[-1]

    # Generate new price
    Z = np.random.normal()
    S_new = S_prev * np.exp((r - 0.5 * sigma_realized ** 2) * dt + sigma_realized * np.sqrt(dt) * Z)
    st.session_state.S_path.append(S_new)

    st.session_state.day += 1
    T_new = max(days_to_expiry - st.session_state.day, 0) / 252.0

    option_price_new = bs_price(S_new, K, T_new, r, sigma_implied, option_type_lower)
    delta_new = bs_delta(S_new, K, T_new, r, sigma_implied, option_type_lower)

    # hedge adjustment
    target_shares = -pos_sign * delta_new
    trade = target_shares - shares_prev
    if abs(trade) > 1e-10:
        st.session_state.cash -= trade * S_new
        st.session_state.trades.append({'day': st.session_state.day, 'shares': trade, 'price': S_new})
    st.session_state.hedge_shares.append(target_shares)

    # daily P/L components
    daily_stock_pl  = shares_prev * (S_new - S_prev)
    daily_option_pl = pos_sign * (option_price_new - option_price_prev)
    daily_total_pl  = daily_stock_pl + daily_option_pl

    # update state
    st.session_state.option_prices.append(option_price_new)
    st.session_state.deltas.append(delta_new)
    st.session_state.pnl_stock_daily.append(daily_stock_pl)
    st.session_state.pnl_option_daily.append(daily_option_pl)
    st.session_state.pnl_daily.append(daily_total_pl)
    st.session_state.pnl_cum.append(st.session_state.pnl_cum[-1] + daily_total_pl)

# ---------- Reset ---------------------------------------------------------------
if reset_sim:
    reset_state()
    st.info('Simulation state cleared.')

# ---------- Charts --------------------------------------------------------------
if len(st.session_state.S_path) > 1:
    st.subheader('Simulation Charts')

    # Stock path
    fig1, ax1 = plt.subplots()
    ax1.plot(range(len(st.session_state.S_path)), st.session_state.S_path, label='Stock Price', linewidth=2)
    for tr in st.session_state.trades:
        m = '^' if tr['shares'] > 0 else 'v'
        c = 'g' if tr['shares'] > 0 else 'r'
        ax1.scatter(tr['day'], st.session_state.S_path[tr['day']], marker=m, color=c, s=80)
    ax1.set_xlabel('Day')
    ax1.set_ylabel('Price ($)')
    ax1.set_title('Stock Price Over Time with Hedging Adjustments')
    ax1.legend()
    st.pyplot(fig1)

    # Running P/L
    fig2, ax2 = plt.subplots()
    ax2.plot(range(len(st.session_state.pnl_cum)), st.session_state.pnl_cum, linewidth=2)
    ax2.set_xlabel('Day')
    ax2.set_ylabel('P/L ($)')
    ax2.set_title('Running Total P/L Over Time')
    st.pyplot(fig2)

    # Daily P/L bar
    if st.session_state.pnl_daily:
        fig3, ax3 = plt.subplots()
        ax3.bar(range(1, len(st.session_state.pnl_cum)), st.session_state.pnl_daily)
        ax3.set_xlabel('Day')
        ax3.set_ylabel('P/L ($)')
        ax3.set_title('Daily P/L')
        st.pyplot(fig3)

# ---------- Summary + Results tables --------------------------------------------
if st.session_state.option_prices:
    st.subheader('Simulation Summary')

    # realised vol from path
    if len(st.session_state.S_path) > 1:
        log_rets = np.diff(np.log(st.session_state.S_path))
        realised_vol_ann = np.std(log_rets) * np.sqrt(252)
    else:
        realised_vol_ann = np.nan

    total_option_pl = pos_sign * (st.session_state.option_prices[-1] - st.session_state.option_prices[0])
    total_stock_pl  = sum(st.session_state.pnl_stock_daily)
    final_total_pl  = st.session_state.pnl_cum[-1]

    # share statistics
    buys = [tr for tr in st.session_state.trades if tr['shares'] > 0]
    sells = [tr for tr in st.session_state.trades if tr['shares'] < 0]
    total_bought = sum(tr['shares'] for tr in buys)
    total_sold   = -sum(tr['shares'] for tr in sells)
    avg_buy_px   = (sum(tr['shares'] * tr['price'] for tr in buys) / total_bought) if total_bought else np.nan
    avg_sell_px  = (sum(-tr['shares'] * tr['price'] for tr in sells) / total_sold) if total_sold else np.nan

    summary_dict = {
        'Final Total P/L ($)': final_total_pl,
        'Total Option P/L ($)': total_option_pl,
        'Total Stock P/L (Hedging) ($)': total_stock_pl,
        'Realized Volatility (Annualized %)': realised_vol_ann * 100,
        'Total Shares Bought': total_bought,
        'Average Buy Price ($)': avg_buy_px,
        'Total Shares Sold': total_sold,
        'Average Sell Price ($)': avg_sell_px
    }
    summary_df = pd.DataFrame(summary_dict, index=['Value']).T
    def color_pl(val):
        if isinstance(val, (int, float)):
            if val > 0:  return 'color: green;'
            if val < 0:  return 'color: red;'
        return ''
    st.dataframe(summary_df.style.applymap(color_pl).format('{:.2f}'))

    # -------- Results table per day ----------
    st.subheader('Simulation Results Table')

    df_results = pd.DataFrame({
        'Day': range(len(st.session_state.S_path)),
        'Stock Price': st.session_state.S_path,
        f'{option_type} Price': st.session_state.option_prices,
        'Option Delta': st.session_state.deltas + ([np.nan] if len(st.session_state.deltas) < len(st.session_state.S_path) else []),
        'Shares Held': st.session_state.hedge_shares + ([np.nan] if len(st.session_state.hedge_shares) < len(st.session_state.S_path) else []),
        'Daily Stock P/L': [0.0] + st.session_state.pnl_stock_daily,
        'Daily Option P/L': [0.0] + st.session_state.pnl_option_daily,
        'Total Daily P/L': [0.0] + st.session_state.pnl_daily,
        'Running Total P/L': st.session_state.pnl_cum
    })

    st.dataframe(df_results.style.applymap(color_pl, subset=['Daily Stock P/L',
                                                             'Daily Option P/L',
                                                             'Total Daily P/L',
                                                             'Running Total P/L'])
                                 .format({'Stock Price': '{:.2f}',
                                          f'{option_type} Price': '{:.2f}',
                                          'Option Delta': '{:.4f}',
                                          'Shares Held': '{:.4f}',
                                          'Daily Stock P/L': '{:.2f}',
                                          'Daily Option P/L': '{:.2f}',
                                          'Total Daily P/L': '{:.2f}',
                                          'Running Total P/L': '{:.2f}'}))

