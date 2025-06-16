import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import statistics
import matplotlib.pyplot as plt
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from arch import arch_model
from datetime import datetime
from scipy.stats import linregress
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

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

        # Plot Log Returns
        ax1.plot(x, y_log, label='Log Returns', color='blue')
        ax1.set_title('Log Returns of Spread')
        ax1.set_ylabel('Log Returns')
        ax1.grid(True)
        ax1.legend()

        # Plot 1st Difference
        ax2.plot(x, y_diff, label='1st Difference', color='orange')
        ax2.set_title('1st Difference of Spread')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Spread Difference')
        ax2.grid(True)
        ax2.legend()

        plt.tight_layout()
        plt.show()

    def garch(self): 
        """
        forecasted_vol_dollar
        print("Spread Difference Vol ($): ", forecasted_vol_dollar)
        spread_diff = self.spread_df['SpreadDiff']
        spread_diff = spread_diff.replace([np.inf, -np.inf], np.nan).dropna()
        model_dollar = arch_model(spread_diff, mean='Zero', vol='Garch', p=1, q=1)
        res_dollar = model_dollar.fit(disp='off')
        forecasts_dollar = res_dollar.forecast(horizon=30)
        forecasted_var_dollar = forecasts_dollar.variance.values[-1,:]
        forecasted_vol_dollar = np.sqrt(forecasted_var_dollar)
        mu_dollar = spread_diff.mean()
        """

        #forecasted_vol_log 
        log_returns = self.spread_df['LogReturns'].replace([np.inf, -np.inf], np.nan).dropna()
        log_returns = log_returns.replace([np.inf, -np.inf], np.nan).dropna()
        model_log = arch_model(log_returns, mean='Zero', vol='Garch', p=1, q=1)
        res_log = model_log.fit(disp='off')
        forecasts_log = res_log.forecast(horizon=30) #30 day forecast
        forecasted_var_log = forecasts_log.variance.values[-1,:] 
        forecasted_vol_log = np.sqrt(forecasted_var_log)
        mu_log = log_returns.mean()

        for item in forecasted_vol_log: 
            print(item)
        cond_vol = res_log.conditional_volatility
        forecast_horizon = 30
        forecasts = res_log.forecast(horizon = forecast_horizon)
        forecasted_vol = np.sqrt(forecasts.variance.values[-1,:])

        plt.figure(figsize=(14, 6))
        # Plot in-sample volatility
        plt.plot(cond_vol.index, cond_vol, label='In-sample Volatility (GARCH)', color='blue')
        # Plot forecasted volatility
        last_date = cond_vol.index[-1]
        forecast_dates = pd.date_range(start=last_date, periods=forecast_horizon+1, freq='D')[1:]
        plt.plot(forecast_dates, forecasted_vol, label='Forecasted Volatility (GARCH)', color='red', linestyle='--', marker='o')

        plt.title('GARCH(1,1) Conditional and Forecasted Volatility')
        plt.xlabel('Date')
        plt.ylabel('Volatility')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def monte_carlo(self):
        num_sim = 1000
        forecast_period = 30
        log_returns = self.spread_df['LogReturns'].replace([np.inf, -np.inf], np.nan).dropna()
        drift = log_returns.mean()
        volatility = log_returns.std()
        dt = 1
        #initial_val = 0
        initial_val = float(self.spread_df['Spread'].iloc[-1])
        print("Initial value is: ", initial_val)
        simulated_paths = np.zeros((num_sim, forecast_period))
        simulated_paths[:, 0] = initial_val #last known price
        
        rand_shock = None

        for t in range(1, forecast_period):
            prev_val = simulated_paths[0, t - 1]
            rand_shock = np.random.normal(size=num_sim)
            #exp_term = np.exp((drift - 0.5 * volatility ** 2) + volatility * rand_shock)
            #simulated_paths[:, t] = simulated_paths[:, t - 1] * exp_term
            simulated_price = simulated_paths[:, t - 1] * (1 + drift + volatility * rand_shock)
            #print("SIMULATED PRICE IS", simulated_price)
            #print("PREV PRICE: ", simulated_paths[:, t - 1])
            simulated_price[simulated_price == 0] = 1e-6
            simulated_paths[:, t] = simulated_price

            #simulated_paths[:, t] = simulated_paths[:, t - 1] + (simulated_paths[:, t - 1] * drift + volatility * simulated_paths[:, t-1] * rand_shock)
            # Print for the first simulation path
            """
            print(f"Step {t}:")
            print(f"  prev_val: {prev_val}")
            print(f"  drift: {drift}")
            print(f"  volatility: {volatility}")
            print(f"  rand_shock[0]: {rand_shock[0]}")
            print(f"  exp_term[0]: {exp_term[0]}")
            print(f"  new_val[0]: {simulated_paths[0, t]}")
            print(f"  Any NaN in simulated_paths[:, t]? {np.isnan(simulated_paths[:, t]).any()}")
            print(f"  min: {simulated_paths[:, t].min()}, max: {simulated_paths[:, t].max()}, mean: {simulated_paths[:, t].mean()}")
            """
        """
        print("here was random shock: ", rand_shock)
        #print("here is exp term: ", exp_term)
        print("and here is volatility: ", volatility)
        print("and lastly here is drift", drift)
        """

        """
        for t in range (1, forecast_period): 
            simulated_paths[:, t] = simulated_paths[:, t - 1] * np.exp((drift - 0.5 * volatility ** 2) + volatility * np.random.normal(size=num_sim))
            print(f"Step {t}, min: {simulated_paths[:, t].min()}, max: {simulated_paths[:, t].max()}, mean: {simulated_paths[:, t].mean()}")
        
        """
        
        simulated_returns = np.log(simulated_paths[:, 1:] / simulated_paths[:, :-1])
        implied_volatility = simulated_returns.std() * np.sqrt(252)
        print(f"Implied vol using Monte Carlo Simulation (Log Returns): {implied_volatility:.4f}")
        
        first_date = self.spread_df['Date'].iloc[-1]
        date_range = pd.date_range(start=first_date, periods=simulated_paths.shape[1], freq='D')
        plt.figure(figsize=(12,8))

        for i in range (simulated_paths.shape[0]): 
            plt.plot(date_range, simulated_paths[i], alpha=0.5, color='blue', linewidth=0.75)
            #print("Simulated path is: ", simulated_paths[i])

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


def main(): 
    data = pd.read_csv('UREC_BGC.csv')
    rec1 = 'PJM Tri-Qual RY25 [BGC]'
    rec2 = 'PJM VA RY24 [BGC]'

    df = RecSpreadAnalyzer(data, rec1, rec2)
    df.calc_spread()
    df.plot_spread()
    df.calc_returns()
    df.garch()
    #df.monte_carlo()

    for item, value in df.spread_metrics.items(): 
        print(item,": ", value)

    spread_df = df.spread_df
    print(spread_df)

if __name__ == "__main__":
    main()