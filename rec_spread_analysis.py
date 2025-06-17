import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import statistics
import matplotlib.pyplot as plt
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from arch import arch_model
from datetime import datetime
from sklearn.linear_model import LinearRegression
from scipy.stats import laplace
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
import os

plt.style.use('default')
sns.set_theme(style="darkgrid")
scaler = StandardScaler()

class RecSpreadAnalyzer: 
    def __init__(
        self,
        df: pd.DataFrame,
        rec1: str,
        rec2: str,
        block_size: int = 1,
        output_dir: str = "output_graphs"
    ) -> None:

        self.df = df
        self.rec1 = rec1
        self.rec2 = rec2
        self.block_size = block_size
        self.output_dir = output_dir
        self.spread_df = None
        self.spread_metrics = None

        os.makedirs(self.output_dir, exist_ok=True)

    def calc_spread(self) -> tuple[pd.DataFrame, dict]: 
        df = self.df.copy()
        rec1 = self.rec1
        rec2 = self.rec2

        df['Spread'] = df[rec1] - df[rec2] - 0.00000000001
        df['Date'] = pd.to_datetime(df['Date'])
        df.sort_values('Date', inplace=True)
        
        ret_df = df[['Date', 'Spread', rec1, rec2]].copy()
        ret_df = ret_df.replace([np.inf, -np.inf], np.nan).dropna()
        spread = ret_df['Spread']
        
        metrics = {
            'mean' : spread.mean(), 
            'std' : spread.std(), 
            'skew' : stats.skew(spread.dropna()), 
            'kurtosis' : stats.kurtosis(spread.dropna()), 
            'variance' : statistics.variance(spread.dropna()), 
            'min' : spread.min(), 
            'max' : spread.max(), 
            'median' : spread.median(), 
            'q1' : spread.quantile(0.25), 
            'q3' : spread.quantile(0.75), 
        }
        metrics['var95'] = np.percentile(spread.dropna(), 5) #VaR analysis at 95% confidence interval
        metrics['es95'] = spread[spread <= metrics['var95']].mean() #Expected loss when loss does happen (spread < VaR)

        self.spread_df = ret_df
        self.spread_metrics = metrics
    

    def plot_spread(self): 
        df = self.spread_df
        metrics = self.spread_metrics

        fig, (ax1, ax2) = plt.subplots(2,1, figsize=(10,6))

        spread = df['Spread']
        rec1 = self.rec1
        rec2 = self.rec2

        sns.histplot(data=df['Spread'], kde=True, ax=ax1)
        ax1.set_title(f'Distribution of Spread: {rec1} vs. {rec2}')
        ax1.set_xlabel('Spread')
        ax1.set_ylabel('Frequency')
        ax1.axvline(metrics['mean'], color='r', linestyle='--', label='Mean')
        ax1.axvline(metrics['var95'], color='b', linestyle='--', label='95% VaR')
        ax1.legend()

        ax2.plot(df['Date'], df['Spread'], label='Spread', color='blue', linewidth=0.8)
        ax2.set_title(f'Time Series of Spread {rec1} vs. {rec2}')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Spread')
        ax2.grid(True)

        df['RollingMean'] = df['Spread'].rolling(window=10).mean()
        df['RollingStd'] = df['Spread'].rolling(window=10).std()
        
        self.spread_df = df

        #ax2.plot(df['Spread'].index, df['RollingMean'], label='Rolling Mean', color='orange')
        #ax2.plot(df['Spread'].index, df['RollingStd'], label='Rolling Std', color='green')
        ax2.legend()

        plt.tight_layout()
        plt.show()
    
    def calc_returns(self): 
        # 1st Difference of Spread (in dollar amount)
        self.spread_df['SpreadDiff'] = self.spread_df['Spread'].diff()
        mu_dollar = self.spread_df['SpreadDiff'].mean()

        # Log Returns
        #self.spread_df['SpreadReturns'] = self.spread_df['Spread'].pct_change()
        prev = self.spread_df['Spread'].shift(1)
        curr = self.spread_df['Spread']
        self.spread_df['SpreadReturns'] = (curr - prev) / prev.abs()
        self.spread_df.loc[self.spread_df['SpreadReturns'] > 2, 'SpreadReturns'] = 1
        self.spread_df.loc[self.spread_df['SpreadReturns'] < -2, 'SpreadReturns'] = -1
        self.spread_df['LogReturns'] = np.log(self.spread_df['SpreadReturns'] + 1)
        filename_safe = f"spread_{self.rec1.replace(' ', '_').replace('[','').replace(']','')}_vs_{self.rec2.replace(' ', '_').replace('[','').replace(']','')}.png"
        filepath = os.path.join(self.output_dir, filename_safe)
        self.spread_df.to_excel("REC_SPREAD.xlsx")
        mu_log = self.spread_df['LogReturns'].mean()
        #using log return, find historical volatility
        #using dollar, find how much historical volatility
        print(self.spread_df)

        x = self.spread_df['Date']
        y_log = self.spread_df['LogReturns']
        y_diff = self.spread_df['SpreadDiff']

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        # Plot Log Returns
        ax1.plot(x, y_log, label='Log Returns', color='blue')
        ax1.set_title('Log Returns of Spread')
        ax1.set_ylabel('Log Returns')
        ax1.grid(True)
        ax1.legend(loc = 'lower right', frameon=False)
        ax1.set_xlabel('Date')

        # Plot 1st Difference
        ax2.plot(x, y_diff, label='1st Difference', color='orange')
        ax2.set_title('1st Difference of Spread')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Spread Difference')
        ax2.grid(True)
        ax2.legend(loc = 'lower right', frameon=False)

        plt.tight_layout()
        plt.show()

    def garch(self): 
        # reformat and clean up log returns
        log_returns = self.spread_df['LogReturns'].replace([np.inf, -np.inf], np.nan).dropna()
        dates = self.spread_df.loc[log_returns.index, 'Date']
        
        model_log = arch_model(log_returns, mean='constant', vol='Garch', p=1, q=1)
        res_log = model_log.fit(disp='off')
        res_log.summary()

        forecasts_log = res_log.forecast(horizon=30) #30 day forecast
        forecasted_var_log = forecasts_log.variance[-1:]  #change these
        forecasted_vol_log = np.sqrt(forecasted_var_log)
        print("This is the forecasted var for 30 day horizon", forecasted_var_log)
        mu_log = log_returns.mean()

        fig, (ax1, ax2, ax3) = plt.subplots(3,1,figsize=(15,12))

        ax1.plot(dates, log_returns, label='Log Returns', color='blue')
        ax1.set_title(f'Log Returns of Spread: {self.rec1} vs. {self.rec2}')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Log Returns')
        ax1.grid(True)
        ax1.legend(loc = 'lower right', frameon=False)

        skewt_gm = arch_model(log_returns, mean='constant', p=1, q=1, vol='GARCH', dist='skewt')
        skewt_result = skewt_gm.fit()
        #skewt_result.summary()
        normal_volatility = res_log.conditional_volatility
        skewt_volatility = skewt_result.conditional_volatility

        ax2.plot(dates, skewt_volatility, color = 'gold', label = 'Skewed-t Volatility')
        ax2.plot(dates, normal_volatility, color = 'turquoise', label = 'Normal Volatility')
        ax2.plot(dates, log_returns, color='grey', label='Log Returns', alpha = 0.4)
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Volatility')
        ax2.legend(loc = 'lower right', frameon=False)
        ax2.grid(True)

        armean_gm = arch_model(log_returns, p=1, q=1, mean='AR', lags=1, vol='GARCH', dist='skewt') #autoregressive mean
        armean_result = armean_gm.fit()
        #armean_result.summary()
        armean_volatility = armean_result.conditional_volatility.dropna()
        skewt_volatility = skewt_volatility.iloc[1:]

        ax3.plot(dates[1:], skewt_volatility, color='gold', label='Constant Mean Volatility')
        ax3.plot(dates[1:], armean_volatility, color='turquoise', label='AR Mean Volatility')
        ax3.plot(dates[1:], log_returns[1:], color='grey', label='Log Returns', alpha=0.4)
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Volatility')
        ax3.legend(loc = 'lower right', frameon=False)
        ax3.grid(True)
        '''----------------------------------------------------------------------------------------------'''





        '''----------------------------------------------------------------------------------------------'''
        
        # Akaike Information Criterion (AIC) and Bayesian Information Criterion (BIC)
        aic = res_log.aic
        bic = res_log.bic
        aic2 = skewt_result.aic
        bic2 = skewt_result.bic
        aic3 = armean_result.aic
        bic3 = armean_result.bic
        print("For normal, constant GARCH", aic, bic)
        print("For skewt student distribution, constant GARCH", aic2, bic2)
        print("For skewt distribution with AR mean", aic3, bic3)

        
        # Backtesting
        residuals = self.spread_df['LogReturns'] - res_log.conditional_volatility
        res_t = residuals / res_log.conditional_volatility
        backtest = (res_t ** 2).sum() # sum of residual squared


        # Out-of-sample Testing
        data_length = len(self.spread_df['Spread'])
        train_size = int(0.8 * data_length) #using 80% of past data, but we can reconfigure this
        train_data = self.spread_df['Spread'][:train_size]
        test_data = self.spread_df['Spread'][train_size:]

        res_oos = model_log.fit(last_obs=train_data.index[-1], disp='off')
        forecast = res_oos.forecast(start=train_data.index[-1], horizon=len(test_data))
        oos_forecast_vol = forecast.residual_variance.iloc[-1, :]
        error = (test_data - oos_forecast_vol).dropna()

        #print("THESE ARE AIC/BIC/BACKTEST/ERROR FOR NORMAL GARCH (constant mean):", aic, bic, backtest, error)
       
    def jarque_bera(self): #check against chi square table for normal distribution
        data = self.spread_df['LogReturns'].replace([np.inf, -np.inf], np.nan).dropna()
        JB = stats.jarque_bera(data)
        print("From library, JB Test: ", stats.jarque_bera(data), stats.jarque_bera(data).pvalue)
        print("JB p-value (scientific notation): {:.20e}".format(JB.pvalue))
        if (JB.pvalue < 0.01): # should follow chi-squared distribution with two degrees of freedom, but set as 0.01 temp
            print("The p value is :", JB.pvalue)
            print("The distribution is not normal")

    """
    def cointegration_test(self): (johansen test)

    """
    def monte_carlo(self):
        num_sim = 5000
        forecast_period = 7
        log_returns = self.spread_df['LogReturns'].replace([np.inf, -np.inf], np.nan).dropna()
        drift = log_returns.mean()
        volatility = log_returns.std()
        dt = 1
        #initial_val = 0
        initial_val = float(self.spread_df['Spread'].iloc[-1])
        #print("Initial value is: ", initial_val)
        simulated_paths = np.zeros((num_sim, forecast_period))
        simulated_paths[:, 0] = initial_val #last known price
        
        reg_spread = self.spread_df['Spread']
        X_t = reg_spread[:-1].values.reshape(-1,1)
        X_t1 = reg_spread[1:].values
        Y = X_t1 - X_t.ravel()
        reg = LinearRegression().fit(X_t, Y)
        slope = reg.coef_[0]
        theta = -slope / dt
        
        rand_shock = None

        for t in range(1, forecast_period):
            #rand_shock = np.random.normal(loc=0, scale=1, size=num_sim)
            rand_shock = np.random.laplace(loc=0, scale=1, size=num_sim)
            simulated_price = simulated_paths[:, t-1] + theta * (drift - simulated_paths[:, t-1]) * dt + volatility * rand_shock * np.sqrt(dt)
            #simulated_price = simulated_paths[:, t - 1] + np.exp((drift - 0.5 * volatility ** 2) * dt + volatility * rand_shock * np.sqrt(dt))
            #simulated_price = simulated_paths[:, t - 1] * (1 + drift + volatility * rand_shock)

            simulated_price[simulated_price == 0] = 1e-6
            simulated_paths[:, t] = simulated_price
        #check laplace dist., see if that's better than skew 
        # 
        # Calculate returns and handle NaN values
        simulated_returns = np.log(simulated_paths[:, 1:] / simulated_paths[:, :-1])
        simulated_returns[np.isinf(simulated_returns)] = np.nan #replace +/- inf with nan
        implied_volatility = np.nanstd(simulated_returns) * np.sqrt(252)
        print(f"Implied vol using Monte Carlo Simulation (Log Returns): {implied_volatility:.4f}")
        
        first_date = self.spread_df['Date'].iloc[-1]
        date_range = pd.date_range(start=first_date, periods=simulated_paths.shape[1], freq='D')
        plt.figure(figsize=(12,8))

        for i in range (simulated_paths.shape[0]): 
            plt.plot(date_range, simulated_paths[i], alpha=0.25, color='blue', linewidth=0.75)
            #print("Simulated path is: ", simulated_paths[i])
        max_spread_per_sim = np.max(simulated_paths, axis=1)
        min_spread_per_sim = np.min(simulated_paths, axis=1)
        print("95% of max spread in 7 days:", np.percentile(max_spread_per_sim, 95))
        print("5% of min spread in 7 days:", np.percentile(min_spread_per_sim, 5))

        meanPath = np.mean(simulated_paths, axis=0)
        percentile5 = np.percentile(simulated_paths, 5, axis=0)
        percentile95 = np.percentile(simulated_paths, 95, axis=0)

        plt.plot(date_range, meanPath, color='r', linewidth=1.5, label='(Expected) Mean Path')
        plt.fill_between(date_range, percentile5, percentile95, color='green', alpha=0.3, label='90% Confidence Interval')

        plt.title('Monte Carlo Simulation of REC Spread')
        plt.xlabel('Date')
        plt.ylabel('Spread')
        plt.legend()
        plt.grid(True, alpha=0.25)

        stats_text = f'Initial Spread: {simulated_paths[0,0]:.2f}\n'
        stats_text += f'Mean Final Spread: {np.mean(simulated_paths[:,-1]):.2f}\n'
        stats_text += f'95th Percentile: {np.percentile(simulated_paths[:,-1], 95):.2f}\n'
        stats_text += f'5th Percentile: {np.percentile(simulated_paths[:,-1], 5):.2f}'
    
        plt.text(0.02, 0.98, stats_text,
                transform=plt.gca().transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()
        plt.show()
        """
        for t in range (1, forecast_period):
            Z = np.random.normal(size = num_sim)
            sigma_t = forecasted_vol_log[t-1] # use dollar for spreadDiff
            print("This is daily vol: ", sigma_t)
            paths[:, t] = paths[:, t-1] * np.exp((mu_log - 0.5 * forecasted_vol_log[t-1]) * dt + sigma_t * np.sqrt(dt) * Z)
        """

    def plot_hist(self):
        data = self.spread_df['LogReturns'].replace([np.inf, -np.inf], np.nan).dropna()
        plt.figure(figsize=(10,6))
        sns.histplot(data, bins=50, kde=True, color='skyblue', edgecolor='black', stat='density', label='Log Returns')
        mu, std = data.mean(), data.std()
        loc, scale = stats.laplace.fit(data)
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 500)
        p_norm = stats.norm.pdf(x, mu, std)
        p_laplace = stats.laplace.pdf(x, loc, scale)
        plt.plot(x, p_norm, 'r-', lw=2, label='Normal PDF')
        plt.plot(x, p_laplace, 'g--', lw=2, label='Laplace PDF')
        plt.title('Histogram of Log Returns with Normal and Laplace Fit')
        plt.xlabel('Log Returns')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        # Optionally, print a quick summary
        print(data.describe())
        print("Number of unique log return values:", data.nunique())

def main(): 
    data = pd.read_csv('UREC_BGC.csv')
    rec1 = 'PJM Tri-Qual RY25 [BGC]'
    rec2 = 'PJM VA RY24 [BGC]'

    df = RecSpreadAnalyzer(data, rec1, rec2)
    df.calc_spread()
    #df.plot_spread()
    df.calc_returns()
    #df.garch()
    df.monte_carlo()
    #df.jarque_bera()
    #df.plot_hist()

if __name__ == "__main__":
    main()


# get 95th and 5th percentile, get jarque bera test for adnan --> check hypothesis on laplace
# look at what the arrays individually store

