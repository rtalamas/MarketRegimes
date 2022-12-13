from datetime import datetime

import streamlit as st
import yfinance as yf

import quandl
from fredapi import Fred

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

from joblib import load

quandl.ApiConfig.api_key = 'n-JBokomw9UHXTsb1s-j'

fred = Fred(api_key='6a118a0ce0c76a5a1d1ad052a65162d6')

# ***** Config *****
st.set_page_config(
    page_icon="🏦",
    layout="wide"
)
st.title('Market Regime Dashboard')
st.caption("""
Dashboard to track the results from [An Analysis of Market Regimes Through The Inflation Lens](https://docs.google.com/document/d/12FLpksaxR8S8ANWjv09GWell4ZY6jlxP9Mvt2gldIf8/edit?usp=sharing)\n
built by [@RobertoTalamas](https://twitter.com/RobertoTalamas)
""")

st.markdown("""## Market Regime Descriptions

### Market Regime 1: Equilibrium
The Equilibrium market regime has been the prevailing market condition since the 1970s. During this period, the S&P 500 
has seen steady state upward trends and daily average returns of 1%. Low volatility across asset classes have made 
investment more reliable and consistent compared to other market regimes. The S&P P/E during this period has been 18x 
and CPI 2.7%. This period of healthy market conditions has provided a more stable environment for investors to plan 
their financial goals. 

### Market Regime 2: '08 Recession
The ‘08 Recession was a period of extreme volatility and uncertainty in the global markets. Equity valuations reached 
levels that had not been seen since the tech bubble burst, with some stocks reaching as high as 120x their value. 
Despite this, it is important to remember that this was an outlier compared to other market regimes, making it 
difficult to generalize these conditions to future states. 
 
### Market Regime 3: Slippery Slope
The Slippery Slope regime has been seen before in instances such as the dot-com bubble, 2008 recession and the COVID-19 
recession. During this market regime equities continue to rally, but valuations begin to overextend getting ahead of 
fundamentals. This regime can also be referred to as a "bubble in the making". 
The average S&P P/E during this regime is 29x and CPI 3.7%.  

### Market Regime 4: Puffed Up
The Puffed Up market regime is characterized by high and incredibly volatile inflation. This regime was prevalent 
during the 1950s and 1970s, and is the only market regime where equity valuations are in the single digits. Those 
investing in stocks during this time would have seen an average S&P 500 P/E of 9x, while CPI was at 5.6%. 
This is by all means a scenario we want to avoid when inflation gets out of hand. 
""")


def line_format_year(label):
    """
    Convert time label to the format of pandas line plot
    """
    year = label.year
    return year


def plot_in_sample_hidden_states(df, series, log_y=False):
    """
    Plot the adjusted closing prices masked by
    the in-sample hidden states as a mechanism
    to understand the market regimes.
    """
    # Predict the hidden states array
    labels = df['labels']
    # Create the correctly formatted plot
    fig, ax = plt.subplots(figsize=(12, 5))

    colours = ["lawngreen", "darkorange", "darkred", "yellow"]
    market_regime_labels = ['Equilibrium', '08 Recession', 'Slippery Slope', 'Puffed Up']

    for i, (colour) in enumerate(colours):
        mask = labels == i
        ax.plot_date(
            df.index[mask],
            df[series][mask],
            '.', linestyle='none',
            c=colour,
            label=market_regime_labels[i]
        )

    ax.set_title(f"{series} By Market Regime")
    ax.legend(loc='upper center', ncol=4)
    if log_y:
        ax.set_yscale('log')
    return ax


# Generate Datasets
# Equity valuations
snp_pe = quandl.get("MULTPL/SP500_PE_RATIO_MONTH")
shiller_pe = quandl.get("MULTPL/SHILLER_PE_RATIO_MONTH")
snp_pe.columns = ["S&P P/E (TTM)"]
shiller_pe.columns = ['Shiller P/E']
pe = snp_pe.join(shiller_pe, how='outer')

# Inflation
head_cpi = fred.get_series('CPIAUCSL', observation_start=snp_pe.index[0]).pct_change(periods=12).to_frame(
    name='CPI') * 100

# Create pandas DataFrame
df = pe.join(head_cpi, how='outer').ffill().dropna()
df = df.join(yf.download('^GSPC', start='1970-01-01', end=datetime.today().strftime('%Y-%m-%d'))['Close'].to_frame(
    name='S&P 500'), how='left')
df['S&P 500 returns'] = df['S&P 500'].pct_change()

# Load GMM model
gmm = load('gmm_market_regimes.joblib')

# Predict Market Regimes
labels = gmm.predict(df[['S&P P/E (TTM)', 'CPI']])
df['labels'] = labels

################################################################################
# P/E vs CPI correlation
################################################################################
corr_ax = df['S&P P/E (TTM)'].rolling(12 * 10).corr(df['CPI']).plot(figsize=(9, 5), color='black')
corr_ax.set_title('10-year Rolling Correlation: S&P P/E (TTM) vs CPI', fontsize=13)
corr_ax.set_ylabel('Correlation')

pos = corr_ax.axhspan(0.0005, .5, color='green', alpha=.3, label='Positive Correlation')
neg = corr_ax.axhspan(-0.0005, -1, color='red', alpha=.3, label='Negative Correlation')

corr_ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))

corr_ax.legend(handles=[pos, neg], loc='upper center', ncol=2)
plt.xticks(rotation=0, ha='center')
corr_fig = plt.gcf()

################################################################################
# P/E vs CPI correlation
################################################################################
corr_ax_spike = df['S&P P/E (TTM)'].rolling(12 * 10).corr(df['CPI']).plot(figsize=(9, 5), color='black')
peak_correlation = round(df['S&P P/E (TTM)'].rolling(12 * 10).corr(df['CPI']).max(), 2)

pos = corr_ax_spike.axvspan('2018-11-01', '2021-09-01', color='gold', alpha=.3, label='Correlation Spike: 2020-21')

peak_corr_label = corr_ax_spike.axhline(peak_correlation, color='green', linestyle='--', label='Peak Correlation')
corr_ax_spike.axhline(0, color='red', linestyle='--')

corr_ax_spike.yaxis.set_major_formatter(mtick.PercentFormatter(1))

corr_ax_spike.set_title(
    f'10-year Rolling Correlation: S&P P/E (TTM) vs CPI\n Peak Correlation {peak_correlation * 100}%', fontsize=13)
corr_ax_spike.set_ylabel('Correlation')

corr_ax_spike.legend(handles=[pos, peak_corr_label], loc='lower center', ncol=3)
plt.xticks(rotation=0, ha='center')
corr_spike_fig = plt.gcf()

################################################################################
# S&P 500 vs CPI scatter plot
################################################################################
# Predictions
beta_2, beta, alpha = np.polyfit(df['CPI'], df['S&P P/E (TTM)'], 2)
preds = beta_2 * df['CPI'] ** 2 + beta * df['CPI'] + alpha

# Plotting
ax = df.plot(kind='scatter', x='CPI', y='S&P P/E (TTM)', figsize=(9, 9))
ax.plot(df['CPI'], preds, linestyle='--', color='black', label='Historical expected PE given CPI')

ax.set_title('S&P P/E (TTM) vs Headline CPI', fontsize=14)

ax.axhline(df['S&P P/E (TTM)'].iloc[-1], linestyle='--', color='r')
ax.scatter(y=df['S&P P/E (TTM)'].iloc[-1], x=df['CPI'].iloc[-1],
           label='Current P/E (TTM)', color='r', marker=".", s=300)

ax.axhline(preds[-1], linestyle='--', color='black')
ax.scatter(y=preds[-1], x=df['CPI'].iloc[-1],
           label='Predicted', color='black', marker=".", s=300)

ax.annotate(f"Current S&P PE: {df['S&P P/E (TTM)'].iloc[-1]}",
            xy=(8.8, 23))
ax.annotate(f"Predicted S&P PE: {round(preds[-1], 2)}",
            xy=(8.8, 13))

ax.xaxis.set_major_formatter(mtick.PercentFormatter())
plt.legend(fontsize=12)
snp_fig = plt.gcf()

################################################################################
# S&P 500 vs CPI scatter plot Highlighting 2021 and 2022
################################################################################
# Plotting
pe_cpi_2021_2022_ax = df.plot(kind='scatter', x='CPI', y='S&P P/E (TTM)', figsize=(9, 9), alpha=0.2)
pe_cpi_2021_2022_ax.plot(df['CPI'], preds, linestyle='--', color='black', label='Historical expected PE given CPI')

pe_cpi_2021_2022_ax.set_title('S&P P/E vs Headline CPI', fontsize=14)

pe_cpi_2021_2022_ax.scatter(y=df[df.index.year.isin([2021, 2022])]['S&P P/E (TTM)'],
                            x=df[df.index.year.isin([2021, 2022])]['CPI'],
                            label="2021 & 2022 Values", color='purple', marker="D", s=20)

pe_cpi_2021_2022_ax.axhline(df['S&P P/E (TTM)'].iloc[-1], linestyle='--', color='r')
pe_cpi_2021_2022_ax.scatter(y=df['S&P P/E (TTM)'].iloc[-1], x=df['CPI'].iloc[-1],
                            label='Current PE (TTM)', color='r', marker=".", s=300)

pe_cpi_2021_2022_ax.xaxis.set_major_formatter(mtick.PercentFormatter())
plt.legend(fontsize=12)
snp_fig_outlier = plt.gcf()

################################################################################
# S&P 500 vs CPI scatter plot heatmap
################################################################################
# Plotting
joint_prob_ax_kde = sns.jointplot(data=df, x="CPI", y='S&P P/E (TTM)', height=8, kind="hex", joint_kws={'gridsize': 30})
historical_values = joint_prob_ax_kde.ax_joint.plot(df['CPI'], preds, linestyle='--', color='black',
                                                    label='Historical expected PE given CPI')

outliers = joint_prob_ax_kde.ax_joint.scatter(y=df[df.index.year.isin([2021, 2022])]['S&P P/E (TTM)'],
                                              x=df[df.index.year.isin([2021, 2022])]['CPI'],
                                              label="2021 & 2022 Values", color='purple', marker="D", s=20)

current = joint_prob_ax_kde.ax_joint.scatter(y=df['S&P P/E (TTM)'].iloc[-1], x=df['CPI'].iloc[-1],
                                             label='Current PE (TTM)', color='r', marker=".", s=300)

joint_prob_ax_kde.ax_joint.axhline(df['S&P P/E (TTM)'].iloc[-1], linestyle='--', color='r')

cbar_ax = joint_prob_ax_kde.fig.add_axes([.91, .25, .05, .6])  # x, y, width, height
plt.colorbar(cax=cbar_ax)

joint_prob_ax_kde.ax_joint.set_xlim([-4, 15.5])
joint_prob_ax_kde.ax_joint.xaxis.set_major_formatter(mtick.PercentFormatter())

handles = [historical_values[0], outliers, current]
joint_prob_ax_kde.ax_joint.legend(handles=handles)
joint_prob_ax_kde_fig = plt.gcf()

################################################################################
# S&P 500 vs CPI scatter plot by market regime
################################################################################
# Plotting
colours = ["lawngreen", "darkorange", "darkred", "yellow"]
market_regime_labels = ['Equilibrium', '08 Recession', 'Slippery Slope', 'Puffed Up']

joint_prob_ax = sns.jointplot(data=df, x="CPI", y='S&P P/E (TTM)', height=8, hue='labels', joint_kws={"s": 1})
joint_prob_ax.ax_joint.xaxis.set_major_formatter(mtick.PercentFormatter())
joint_prob_ax.ax_joint.axhline(df['S&P P/E (TTM)'].iloc[-1], linestyle='--', color='r')

handles = []
for i, (colour) in enumerate(colours):
    mask = labels == i
    regime_line = joint_prob_ax.ax_joint.scatter(
        x=df['CPI'][mask],
        y=df['S&P P/E (TTM)'][mask],
        c=colour,
        label=market_regime_labels[i],
        s=20
    )
    handles.append(regime_line)

predictions = joint_prob_ax.ax_joint.plot(df['CPI'], preds, linestyle='solid', color='black',
                                          label='Historical expected PE given CPI')
current = joint_prob_ax.ax_joint.scatter(y=df['S&P P/E (TTM)'].iloc[-1], x=df['CPI'].iloc[-1],
                                         label='Current P/E (TTM)', color='red', marker=".", s=250)

joint_prob_ax.ax_joint.set_ylim([0, 130])
joint_prob_ax.ax_joint.set_xlim([-3.5, 15])

handles = handles + [predictions[0], current]
joint_prob_ax.ax_joint.legend(handles=handles)
snp_fig_market_regime = plt.gcf()

################################################################################
# S&P 500 by Regime
####################################################################
# ############
fmt = '$%.0f'
tick = mtick.FormatStrFormatter(fmt)
snp_ax = plot_in_sample_hidden_states(df, 'S&P 500')
snp_ax.yaxis.set_major_formatter(tick)
snp_by_regime_fig = plt.gcf()

################################################################################
# CPI  by Regime
################################################################################
cpi_ax = plot_in_sample_hidden_states(df, 'CPI')
cpi_ax.yaxis.set_major_formatter(mtick.PercentFormatter())
cpi_fig = plt.gcf()

################################################################################
# S&P 500 P/E by Regime
################################################################################
pe_ax = plot_in_sample_hidden_states(df, 'S&P P/E (TTM)')
pe_fig = plt.gcf()

################################################################################
# Market Regime Probability
################################################################################
regime_probability_df = pd.DataFrame(gmm.predict_proba(df[['S&P P/E (TTM)', 'CPI']]), index=df.index)
regime_probability_df.columns = market_regime_labels

regime_probability_ax = regime_probability_df.plot(kind='area', color=colours, figsize=(12, 5))
regime_probability_ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
regime_probability_ax.legend(ncol=4, loc='lower center')
regime_probability_ax.set_title('Highest Probability Market Condition Since The 1950s', fontsize=13)
regime_probability_ax.margins(x=0, y=0)
regime_probability_ax.set_ylabel('Market Condition Probability ')
plt.xticks(ha='center', rotation=0)

regime_prob_fig = plt.gcf()


################################################################################
# Streamlit App
################################################################################
def get_start_end_dates(series):
    series_no_na = series.dropna()
    start = series_no_na.dropna().index[0].strftime('%Y-%m-%d')
    end = series_no_na.dropna().index[-1].strftime('%Y-%m-%d')
    return start, end


tab_linecharts, tab_scatterplots = st.tabs(['Market Regimes Through Time', 'Market Regime Clusters'])

################################################################################
# Market Regime Through Time By Regime
################################################################################
with tab_linecharts:
    line_chart_col1, line_chart_col2 = st.columns([1, 1])
    line_chart_col3, line_chart_col4 = st.columns([1, 1])

    # Market Regime Probability
    with line_chart_col1:
        start, end = get_start_end_dates(regime_probability_df['Equilibrium'])
        st.header('Market Regime Probability')
        st.caption(f'Dataset includes observations from {start} to {end}')
        st.pyplot(regime_prob_fig)
        st.caption(
            '[Source Code](https://nbviewer.org/github/rtalamas/MarketRegimes/blob/main/MarketRegimes.ipynb#Market-Regime-Probability)')

    # S&P 500 by Regime
    with line_chart_col2:
        start, end = get_start_end_dates(df['S&P 500'])
        st.header('S&P 500 By Market Regime')
        st.caption(f'Dataset includes observations from {start} to {end}')
        st.pyplot(snp_by_regime_fig)
        st.caption('[Source Code](https://nbviewer.org/github/rtalamas/MarketRegimes/blob/main/MarketRegimes.ipynb#S&P-500-By-Market-Regime)')

    # CPI by Regime
    with line_chart_col3:
        start, end = get_start_end_dates(df['CPI'])
        st.header('CPI By Market Regime')
        st.caption(f'Dataset includes observations from {start} to {end}')
        st.pyplot(cpi_fig)
        st.caption('[Source Code](https://nbviewer.org/github/rtalamas/MarketRegimes/blob/main/MarketRegimes.ipynb#CPI-By-Market-Regime)')

    # P/E by Regime
    with line_chart_col4:
        start, end = get_start_end_dates(df['S&P P/E (TTM)'])
        st.header('S&P 500 P/E By Market Regime')
        st.caption(f'Dataset includes observations from {start} to {end}')
        st.pyplot(pe_fig)
        st.caption('[Source Code](https://nbviewer.org/github/rtalamas/MarketRegimes/blob/main/MarketRegimes.ipynb#S&P-500-P/E-By-Market-Regime)')

################################################################################
# Scatterplot Tab
################################################################################
with tab_scatterplots:
    col1, col2 = st.columns([1, 1])
    col3, col4 = st.columns([1, 1])
    start, end = get_start_end_dates(df['S&P P/E (TTM)'])

    with col1:
        st.header('S&P 500 P/E vs CPI By Market Regime')
        st.caption(f'Dataset includes observations from {start} to {end}')
        st.pyplot(snp_fig_market_regime)
        st.caption('[Source Code](https://nbviewer.org/github/rtalamas/MarketRegimes/blob/main/MarketRegimes.ipynb#S&P-500-P/E-vs-CPI-By-Market-Regime)')

    with col2:
        st.header('S&P 500 P/E vs CPI Heatmap')
        st.caption(f'Dataset includes observations from {start} to {end}')
        st.pyplot(joint_prob_ax_kde_fig)
        st.caption('[Source Code](https://nbviewer.org/github/rtalamas/MarketRegimes/blob/main/MarketRegimes.ipynb#S&P-500-P/E-vs-CPI-Heatmap)')

    with col3:
        st.header('S&P 500 P/E vs CPI 2021-2022 Highlight')
        st.caption(f'Dataset includes observations from {start} to {end}')
        st.pyplot(snp_fig_outlier)
        st.caption('[Source Code](https://nbviewer.org/github/rtalamas/MarketRegimes/blob/main/MarketRegimes.ipynb#S&P-500-P/E-vs-CPI)')

    with col4:
        st.header('S&P 500 P/E vs CPI')
        st.caption(f'Dataset includes observations from {start} to {end}')
        st.pyplot(snp_fig)
        st.caption('[Source Code](https://nbviewer.org/github/rtalamas/MarketRegimes/blob/main/MarketRegimes.ipynb#S&P-500-P/E-vs-CPI)')
