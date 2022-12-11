from datetime import datetime

import streamlit as st
import yfinance as yf

import quandl
from fredapi import Fred

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.dates import YearLocator, MonthLocator
import matplotlib.ticker as mtick
import seaborn as sns

from joblib import dump, load

quandl.ApiConfig.api_key = 'n-JBokomw9UHXTsb1s-j'

fred = Fred(api_key='6a118a0ce0c76a5a1d1ad052a65162d6')

# ***** Config *****
st.set_page_config(
    page_icon="🏦",
    layout="wide"
)
st.title('Market Regime Dashboard')
st.caption(
    'built by [@RobertoTalamas](https://twitter.com/RobertoTalamas)')


def line_format_year(label):
    """
    Convert time label to the format of pandas line plot
    """
    year = label.year
    return year


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

# P/E vs CPI correlation
corr_ax = df['S&P P/E (TTM)'].rolling(12 * 10).corr(df['CPI']).plot(figsize=(9, 5), color='black')
corr_ax.set_title('10-year Rolling Correlation: S&P P/E (TTM) vs CPI', fontsize=13)
corr_ax.set_ylabel('Correlation')

pos = corr_ax.axhspan(0.0005, .5, color='green', alpha=.3, label='Positive Correlation')
neg = corr_ax.axhspan(-0.0005, -1, color='red', alpha=.3, label='Negative Correlation')

corr_ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))

corr_ax.legend(handles=[pos, neg], loc='upper center', ncol=2)
plt.xticks(rotation=0, ha='center')
corr_fig = plt.gcf()

# P/E vs CPI Correlation Spike
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

start_date = df.index[0].strftime('%Y-%m-%d')
end = df.index[-1].strftime('%Y-%m-%d')

# Streamlit App
col1, col2 = st.columns([1, 1])
col3, col4 = st.columns([1, 1])

with col1:
    st.header('S&P 500 P/E vs CPI By Market Regime')
    st.caption(f'Dataset includes observations from {start_date} to {end}')
    st.pyplot(snp_fig_market_regime)

with col2:
    st.header('S&P 500 P/E vs CPI Heatmap')
    st.caption(f'Dataset includes observations from {start_date} to {end}')
    st.pyplot(joint_prob_ax_kde_fig)

with col3:
    st.header('S&P 500 P/E vs CPI 2021-2022 Highlight')
    st.caption(f'Dataset includes observations from {start_date} to {end}')
    st.pyplot(snp_fig_outlier)

with col4:
    st.header('S&P 500 P/E vs CPI')
    st.caption(f'Dataset includes observations from {start_date} to {end}')
    st.pyplot(snp_fig)
