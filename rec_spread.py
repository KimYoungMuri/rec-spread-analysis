import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from arch import arch_model
from datetime import datetime
from scipy.stats import linregress
import os

plt.style.use('default')
sns.set_theme(style="darkgrid")

# ensure output directory exists
OUTPUT_DIR = "output_graphs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def LoadData(file_path): 
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    return df

def CalcSpread(df, rec1, rec2, blockSize = None): 
    spread = df[rec1] - df[rec2] 
    ret = (spread/spread.shift(1)).pct_change()
    ret = ret.replace([np.inf, -np.inf], np.nan).dropna()
    metrics = {
        'mean' : ret.mean(),
        'std_dev' : ret.std(), 
        'skew' : stats.skew(ret.dropna()), 
        #high skew = long tail on right side of distribution (mean < median)
        #low skew = long tail on left side of distribution (mean > median)
        'kurtosis' : stats.kurtosis(ret.dropna()), 
        #high kurtosis = peaked distribution with thicker tails
        #low kurtosis = flat distribution with thinner tails
        'min' : ret.min(), 
        'max' : ret.max(), 
        'median' : ret.median(), 
        'q1' : ret.quantile(0.25), 
        'q3' : ret.quantile(0.75), 
        'var' : ret.var(), 
    }

    metrics['var95'] = np.percentile(ret.dropna(), 5) #VaR analysis at 95% confidence interval
    metrics['es95'] = ret[ret <= metrics['var95']].mean() #Expected loss when loss does happen (spread < VaR)

    if blockSize: 
        blockSpread = spread * blockSize 
        metrics['blockMean'] = blockSpread.mean()
        metrics['blockStdDev'] = blockSpread.std()
        metrics['blockVar95'] = np.percentile(blockSpread.dropna(), 5)
        metrics['blockEs95'] = blockSpread[blockSpread <= metrics['blockVar95']].mean()

    return metrics

def plotSpreadDistribution(df, rec1, rec2, metrics): 
    spread = df[rec1] - df[rec2]
    spread.index = df['Date'] #Spread series indexed by date

    #2 subplots, 1 column, ax1 (top), ax2 (bottom)
    fig, (ax1, ax2) = plt.subplots(2,1, figsize=(10,6))

    sns.histplot(data=spread, kde=True, ax=ax1)
    ax1.set_title(f'Distribution of Spread: {rec1} vs. {rec2}')
    ax1.set_xlabel('Spread')
    ax1.set_ylabel('Frequency')
    ax1.axvline(metrics['mean'], color='r', linestyle='--', label='Mean')
    ax1.axvline(metrics['var95'], color='b', linestyle='--', label='95% VaR')
    ax1.legend()

    ax2.plot(spread.index, spread.values, label='Spread', color='blue', linewidth=0.8)
    ax2.set_title(f'Time Series of Spread {rec1} vs. {rec2}')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Spread')
    ax2.grid(True)

    spread_df = pd.DataFrame({
        'Date': spread.index,
        'Spread': spread.values,
    })
    """
    - the next 7 trading days, we want to create a range of possible returns
    - print out table REC_SPREAD
    use where spread is currently, and what the last 7 day std is, 
    apply std to 95% z-score through 7 days to create that range

    generate same range so that log/normal/pct_change is adjusted for monte carlo


    """


    spread_df['RollingMean'] = spread_df['Spread'].rolling(window=3).mean()
    spread_df['RollingStd'] = spread_df['Spread'].rolling(window=3).std()

    rolling_mean = spread.rolling(window=3).mean()
    rolling_std = spread.rolling(window=3).std()

    ax2.plot(spread.index, rolling_mean, label='Rolling Mean', color='orange')
    ax2.plot(spread.index, rolling_std, label='Rolling Std', color='green')
    ax2.legend()
    
    plt.tight_layout()

    # Save the figure to disk
    filename_safe = f"spread_{rec1.replace(' ', '_').replace('[','').replace(']','')}_vs_{rec2.replace(' ', '_').replace('[','').replace(']','')}.png"
    filepath = os.path.join(OUTPUT_DIR, filename_safe)
    
    spread.to_excel("REC_SPREAD.xlsx")
    print(f"Spread distribution plot saved to {filepath}")

    plt.show()

def garch_var(returns, forecast_period):
    # Fit GARCH(1,1) with constant mean
    model = arch_model(returns, mean='Zero', vol='Garch', p=1, q=1)
    res = model.fit(disp='off')
    forecasts = res.forecast(horizon=forecast_period)

    forecasted_var = forecasts.variance.values[-1,:]
    forecasted_vol = np.sqrt(forecasted_var)
    return forecasted_vol


def MonteCarlo(spreadSeries, numSim, forecastPeriod):  # number of simulations, forecast days
    spreadSeries = pd.Series(spreadSeries).dropna()

    if (spreadSeries <= 0).any():
        offset = abs(min(spreadSeries.min(), 0)) + 1e-6
        spreadSeries = spreadSeries + offset

    # Log returns
    returns = (spreadSeries - spreadSeries.shift(1)).pct_change()
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
    returns = returns.rolling(window=10).mean().dropna()
    plt.plot(returns)
    mu = returns.mean()

    forecasted_vol = garch_var(returns, forecastPeriod)
    print("This is forecasted_vol")
    print(forecasted_vol)
    if np.isnan(mu) or np.any(np.isnan(forecasted_vol)):
        raise ValueError("Invalid parameters estimated for Monte Carlo simulation.")

    dt = 1  # Daily steps
    initial_val = float(spreadSeries.iloc[-1])

    paths = np.zeros((numSim, forecastPeriod))
    paths[:, 0] = initial_val

    for t in range(1, forecastPeriod):
        Z = np.random.normal(size=numSim)
        sigma_t = forecasted_vol[t - 1] # conditional std dev for period t
        print("This is daily vol: ")
        print(sigma_t)
        paths[:, t] = paths[:, t - 1] * np.exp((mu - 0.5 * forecasted_vol[t - 1]) * dt + sigma_t * np.sqrt(dt) * Z)

    
    return paths

def plotMonteCarlo(paths, rec1, rec2, S_0): 
    plt.figure(figsize = (12,8))
    dates = pd.date_range(start=S_0, periods=paths.shape[1], freq='D')
    
    #for i in range(len(paths)):  #plot all, but optionally change setting
    #    plt.plot(dates, paths[i], alpha=0.05, color='m', linewidth=0.75)
    for i in range (len(paths)): # <-- customize to len(paths)
           plt.plot(dates, paths[i], alpha=0.8, color='blue', linewidth=0.75)
           
    meanPath = np.mean(paths, axis=0)
    percentile5 = np.percentile(paths, 5, axis=0)
    percentile95 = np.percentile(paths, 95, axis=0)

    plt.plot(dates, meanPath, color='r', linewidth=1.5, label='(Expected) Mean Path')
    plt.fill_between(dates, percentile5, percentile95, 
                     color='green', alpha=0.3, label='90% Confidence Interval') #check, is this right?
    
    plt.title('Monte Carlo Simulation of REC Spread')
    plt.xlabel('Date')
    plt.ylabel('Spread')
    plt.legend()
    plt.grid(True, alpha=0.25)

    stats_text = f'Initial Spread: {paths[0,0]:.2f}\n'
    stats_text += f'Mean Final Spread: {np.mean(paths[:,-1]):.2f}\n'
    stats_text += f'95th Percentile: {np.percentile(paths[:,-1], 95):.2f}\n'
    stats_text += f'5th Percentile: {np.percentile(paths[:,-1], 5):.2f}'
    
    plt.text(0.02, 0.98, stats_text,
             transform=plt.gca().transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    #plt.tight_layout()
    plt.show()


def johansen_cointegration_test(df, cols, det_order=0, k_ar_diff=1, significance=0.05, verbose=True): #k_arrdiff lags, deterministic order = 0 --> intercept only

    data = df[cols].dropna()
    result = coint_johansen(data, det_order, k_ar_diff)
    # this rewrites that VAR in Vector Error-Correction Model (VECM) form, which can be decomposed into the cointegration vector
    # two statistics produced for r = 0~k-1 = 0~1
        # trace statistic
        # max-eigenvalue statistic
        # Λ_trace(i) > 95% critical value --> yes cointegration
        # H0: rank <= r vs. H1: rank > r
        # H0 ("null"): what we assume is true, H1("alternative") what we accept if the data allows us to reject H0
        # wtf
        
    if verbose:
        trace_stats = result.lr1
        cv_idx = {0.10: 0, 0.05: 1, 0.01: 2}.get(significance, 1)
        crit_vals = result.cvt[:, cv_idx]
        print(f"Johansen Trace Test (significance = {int(significance*100)}%)")
        for r, (stat, cv) in enumerate(zip(trace_stats, crit_vals)):
            decision = "REJECT" if stat > cv else "fail to reject"
            print(f"  r <= {r}: Λ_trace = {stat:.4f}  |  CV = {cv:.4f}  → {decision}")
        print()

    return result


def is_spread_stationary(df, rec1, rec2, det_order=0, k_ar_diff=1, significance=0.05):
    result = johansen_cointegration_test(df, [rec1, rec2], det_order, k_ar_diff, significance, verbose=False)
    trace_stat = result.lr1[0]
    cv_idx = {0.10: 0, 0.05: 1, 0.01: 2}.get(significance, 1)
    critical_value = result.cvt[0, cv_idx]
    return trace_stat > critical_value

def main():
    df = LoadData('UREC_BGC.csv')
    rec1 = 'PJM Tri-Qual RY25 [BGC]'
    rec2 = 'PJM VA RY24 [BGC]'

    # Johansen test at 5% significance
    johansen_cointegration_test(df, [rec1, rec2], det_order=0, k_ar_diff=1, significance=0.05)
    if is_spread_stationary(df, rec1, rec2, significance=0.05):
        print('\nJohansen test (5%): The spread is STATIONARY (cointegrated).')
    else:
        print('\nJohansen test (5%): The spread is NOT stationary.')

    metrics = CalcSpread(df, rec1, rec2, blockSize=1)  # can change block size to 25000
    paths = MonteCarlo(df[rec1] - df[rec2], 10000, 30)  # 10k simulations, 1-month horizon
    plotSpreadDistribution(df, rec1, rec2, metrics)
    #plotMonteCarlo(paths, rec1, rec2, df['Date'].iloc[-1])

    for item, value in metrics.items():
        print(f"{item}: {value:.4f}")


if __name__ == "__main__":
    main() 

'''

1) cointegration test (what is, what they are)
    - since prices don't have stable mean and variance, they are I(1) --> need to use linear combination of them to be stationary
    - tells us cointegration rank r, the number of cointegrating relations (from Vector Autoregression)
2) dip, z-score (z-score analysis)
3) hedge ratio

fix monte carlo
--> backtesting
--> AIR BIC testing
--> interpretive conclusion

use 3 multivariate garch models

compare w/ simple regression, ARIMA/SARIMA, SARIMA-GARCH, 

'''

