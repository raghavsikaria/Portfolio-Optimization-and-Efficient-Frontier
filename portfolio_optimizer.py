##########################################################
# Author: Raghav Sikaria
# LinkedIn: https://www.linkedin.com/in/raghavsikaria/
# Github: https://github.com/raghavsikaria
# Last Update: 30-5-2020
# Project: Portfolio Optimization & Efficient Frontier
# generator
##########################################################

# Python libraries imports
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
plt.style.use('ggplot')

from bokeh.io import export_png, export_svgs
from bokeh.plotting import figure, output_file, show, save
from bokeh.layouts import grid, column, gridplot, layout, row
from bokeh.models import ColumnDataSource, Title, Legend, Circle, Line, ColorBar, LinearColorMapper, BasicTicker, PrintfTickFormatter, LabelSet, Label, Text, HoverTool
from bokeh.transform import linear_cmap
from bokeh.palettes import inferno, magma, viridis, gray, cividis, turbo

# Project imports
from correlation_matrix_generator import generate_correlation_graph

class PortfolioGenerator:
    def __init__(self):
        self.df_log_daily_return_mean = None
        self.df_log_daily_return_cov = None
    
    def read_csv_and_generate_df(self, path: 'str', index_col: 'str', columns: 'list' = None) -> 'stock df, cumulative returns df, log normalized daily returns df':
        """Reads the stock data csv, prepares stock df + lognormalized daily returns df + cumulative returns df and initializes returns mean & covariance to object variables."""

        df = pd.read_csv(path,index_col = index_col,parse_dates=True)
        if columns: df = df[columns]

        df_normalised_return = pd.DataFrame(index=df.index)
        df_normalised_return.index.name = 'Date'
        for column in df.columns.values:
            df_normalised_return[f'{column} Normed Return'] = df[column]/df.iloc[0][column]

        df_log_daily_return = pd.DataFrame(index=df.index)
        df_log_daily_return.index.name = 'Date'
        for column in df.columns.values:
            df_log_daily_return[f'{column} Log Daily Return'] = np.log(df[column]/df[column].shift(1))
        
        self.df_log_daily_return_mean = df_log_daily_return.mean()
        self.df_log_daily_return_cov = df_log_daily_return.cov()
        
        return df, df_normalised_return, df_log_daily_return
    
    @staticmethod
    def generate_df_histograms(df, path: 'str', title: 'str', bins: 'int'=100, figsize: 'tuple'=(20,20)) -> 'None':
        """Generates and saves DF plot as histogram for all its columns"""

        df.hist(bins=bins, alpha=0.8, figsize=figsize)
        plt.suptitle(title, x=0.5, y=0.95, ha='center', fontsize=15)
        plt.savefig(path)
    
    @staticmethod
    def generate_df_columns_plot(df, path: 'str', title: 'str', figsize: 'tuple'=(20,16)) -> 'None':
        """Generates and saves DF plot for all its columns"""

        df.plot(figsize=figsize, title = title)
        plt.savefig(path)
    
    @staticmethod
    def generate_and_save_correlation_matrix(df: 'Stocks Data Frame', path_for_correlation_df: 'str', path_for_correlation_plot: 'str') -> 'None':
        """Generates correlation matrix, saves it and then uses Correlation Matrix generator project to create interactive plot!"""

        corr_m = df.corr()
        corr_m.to_csv(path_for_correlation_df)
        generate_correlation_graph(path_for_correlation_df, path_for_correlation_plot, plot_height=800, plot_width=1400)

    def minimize_volatility(self, weights: 'list') -> 'volatility np array from get_ret_vol_sr function call':
        """Minimization function for the scipy optimize function. Taken from Jose's abovementioned course!"""

        return  self.get_ret_vol_sr(weights)[1]
    
    def get_ret_vol_sr(self, weights: 'list') -> 'np array of returns, volatility & Sharpe ratio':
        """Takes in weights, returns array or return,volatility, sharpe ratio. Taken from Jose's abovementioned course!"""

        weights = np.array(weights)
        ret = np.sum(self.df_log_daily_return_mean * weights) * 252
        vol = np.sqrt(np.dot(weights.T, np.dot(self.df_log_daily_return_cov * 252, weights)))
        sr = ret/vol
        return np.array([ret,vol,sr])
  
    @staticmethod
    def check_sum(weights: 'list'):
        """Returns 0 if sum of weights is 1.0. Taken from Jose's abovementioned course!"""

        return np.sum(weights) - 1

    def generate_efficient_frontier(self, number_of_stocks: 'int', frontier_min_return: 'float', frontier_max_return: 'float', number_of_frontier_portfolios: 'int' = 100) -> 'Frontier Volatility & Returns, lists':
        """Generates efficient frontier for the given range of desirable portfolio returns. Taken from Jose's abovementioned course!"""

        frontier_volatility = []

        # We generate 'number_of_frontier_portfolios' number of returns between our return ranges
        # so that we can generate optimal portfolios amongst them!
        frontier_returns = np.linspace(frontier_min_return, frontier_max_return, number_of_frontier_portfolios)

        # Initial weights that we choose for our portfolio
        init_guess = [1/number_of_stocks]*number_of_stocks

        # Contraining the bounds for each weight between 0 & 1
        bounds = tuple([(0,1) for _ in range(number_of_stocks)])

        # Iterating over different returns and churning out
        # portfolios with minimum risk
        for possible_return in tqdm(frontier_returns):
            cons = ({'type':'eq','fun': PortfolioGenerator.check_sum},
                    {'type':'eq','fun': lambda w: self.get_ret_vol_sr(w)[0] - possible_return}
                )
            
            # GROUND ZERO - the real magic of SCIPY-optimize happens here
            result = minimize(self.minimize_volatility,init_guess,method='SLSQP',bounds=bounds,constraints=cons)

            # Storing the volatility of the generated portfolio
            frontier_volatility.append(result['fun'])
        return frontier_volatility, frontier_returns

    def generate_all_portfolios(self, df: 'Data Frame of all stocks', number_of_portfolios: 'int' = 5000) -> 'dict':
        """Runs Monte Carlo simulation and creates desired number of portfolios for universe. Taken from Jose's abovementioned course!"""

        # Seeting seed for future reproducibility
        np.random.seed(55)
        log_ret_mean = self.df_log_daily_return_mean
        log_ret_cov = self.df_log_daily_return_cov
        number_of_stocks = len(df.columns)

        # Initializing portfolio particulars
        all_weights = np.zeros((number_of_portfolios,number_of_stocks))
        ret_arr = np.zeros(number_of_portfolios)
        vol_arr = np.zeros(number_of_portfolios)
        sharpe_arr = np.zeros(number_of_portfolios)

        # Beginning our iterations and generating all the portfolios
        for index in tqdm(range(number_of_portfolios)):

            # Here we generate the random weights for our stocks
            weights = np.array(np.random.random(number_of_stocks))

            # We make sure that all weights lie between 0 and 1
            weights = weights / np.sum(weights)
            all_weights[index,:] = weights
            
            # We calculate annualised returns for our created portfolio
            # assuming 252 trading days in an year
            ret_arr[index] = np.sum(log_ret_mean * weights *252)

            # We calculate the annualised volatility (standard deviation)
            # for our portfolio by exploiting some wicked Linear Algebra
            vol_arr[index] = np.sqrt(np.dot(weights.T, np.dot(log_ret_cov * 252, weights)))

            # Finally we find out the Sharpe Ratio for our portfolio
            # assuming 0 (Zero) Risk Free rate
            sharpe_arr[index] = ret_arr[index]/vol_arr[index]
        
        return {
            'max_sharpe_ratio': sharpe_arr.max(),
            'max_sharpe_ratio_index': sharpe_arr.argmax(),
            'max_sharpe_ratio_return': ret_arr[sharpe_arr.argmax()],
            'max_sharpe_ratio_volatility': vol_arr[sharpe_arr.argmax()],
            'all_portfolio_weights': all_weights,
            'all_portfolio_returns': ret_arr,
            'all_portfolio_volatility': vol_arr,
            'all_portfolio_sharpe_ratio': sharpe_arr
        }
    
    @staticmethod
    def plot_efficient_frontier(title: 'str', plot_save_path: 'str', plot_image_save_path: 'str', all_portfolio_volatility: 'np array', all_portfolio_returns: 'np array', all_portfolio_sharpe_ratio: 'np array', max_sharpe_ratio_location: '[x,y] Coordinates', max_sharpe_ratio: 'float', efficient_frontier_returns: 'list', efficient_frontier_volatility: 'list', plot_width: 'int' = 1000, plot_height: 'int' = 800, axes_labels: '[X-axis label, Y-axis label]' = ['Portfolio Volatility', 'Portfolio Returns']) -> 'None':
        """Generates plot for the entire portfolio universe and efficient frontier curve."""

        # Creating a color mapper so that we have a smooth gradient type color 
        # effect for portfolios ranging from lowest to highest Sharpe Ratio
        mapper = LinearColorMapper(palette=inferno(256), low=all_portfolio_sharpe_ratio.min(), high=all_portfolio_sharpe_ratio.max())

        # Preparing main ColumnDataSource which will be fed to the Bokeh Plot
        source = ColumnDataSource(data=dict(all_portfolio_volatility = all_portfolio_volatility, all_portfolio_returns = all_portfolio_returns, all_portfolio_sharpe_ratio = all_portfolio_sharpe_ratio))

        # Initializing our Bokeh Plot
        p = figure(plot_width=plot_width, plot_height=plot_height, toolbar_location='right', x_axis_label=axes_labels[0], y_axis_label=axes_labels[1], tooltips=[('Sharpe Ratio', '@all_portfolio_sharpe_ratio'),('Portfolio Return', '@all_portfolio_returns'),('Portfolio Volatility', '@all_portfolio_volatility')])

        # Adding circular rings for Sharpe Ratios of each portfolio
        p.circle(x='all_portfolio_volatility', y='all_portfolio_returns', source=source, fill_color={'field': 'all_portfolio_sharpe_ratio', 'transform': mapper}, size=8)

        # Adding circular ring for Maximum Sharpe Ratio portfolio
        p.circle(x=max_sharpe_ratio_location[0], y=max_sharpe_ratio_location[1], fill_color='white', line_width=5, size=20)

        # Adding Text annotation for Maximum Sharpe Ratio portfolio
        maximum_sharpe_ratio_portfolio = Label(x=max_sharpe_ratio_location[0]+0.005, y=max_sharpe_ratio_location[1]+0.02, text=f'Maximum SR ~ {round(max_sharpe_ratio, 4)}', render_mode='css', border_line_color='black', border_line_alpha=1.0, background_fill_color='white', background_fill_alpha=0.8)
        p.add_layout(maximum_sharpe_ratio_portfolio)

        # Adding the Efficient Frontier curve to our plot
        p.line(x=efficient_frontier_volatility, y=efficient_frontier_returns, line_width=2, line_color='white', legend='Efficient Frontier', line_dash='dashed')

        # Adding a color bar to indicate about varying Sharpe Ratio
        color_bar = ColorBar(color_mapper=mapper, major_label_text_font_size="5pt",ticker=BasicTicker(desired_num_ticks=10),formatter=PrintfTickFormatter(format="%.1f"),label_standoff=6, border_line_color=None, location=(0, 0))
        p.add_layout(color_bar, 'right')

        # Adding title to our plot
        p.add_layout(Title(text=title, text_font_size="16pt"), 'above')
        p.title.text_font_size = '20pt'
        
        # Fixing up Plot properties
        p.background_fill_color = "black"
        p.background_fill_alpha = 0.9
        p.xgrid.grid_line_color = None
        p.ygrid.grid_line_color = None

        # Fixing up Plot legend
        p.legend.background_fill_alpha = 0.5
        p.legend.background_fill_color = 'black'
        p.legend.label_text_color = 'white'
        p.legend.click_policy = 'mute'

        # Saving the plot
        output_file(plot_save_path)
        save(p)
        export_png(p, filename = plot_image_save_path)

if __name__ == "__main__": 
    
    # For all NIFTYBANK Index Stocks
    nb_allindex = PortfolioGenerator()
    
    df, df_normalised_return, df_log_daily_return = nb_allindex.read_csv_and_generate_df('data/nifty_bank_df_2018-4-2_to_2020-5-22.csv', 'Date')
    
    nb_allindex.generate_df_histograms(df_log_daily_return, 'assets/TEMPNIFTYBANK_df_log_returns_histogram.png', 'NIFTYBANK Stocks Daily Log Returns')
    
    nb_allindex.generate_df_columns_plot(df_normalised_return, 'assets/TEMPNIFTYBANK_df_norm_returns.png', 'NIFTYBANK Cumulative Return')
    
    nb_allindex.generate_and_save_correlation_matrix(df, 'assets/TEMPNIFTYBANK_df_correlation.csv', 'assets/TEMPNIFTYBANK_df_correlation_matrix.html')
    
    all_portfolio_data = nb_allindex.generate_all_portfolios(df, 25000)

    frontier_volatility, frontier_returns = nb_allindex.generate_efficient_frontier(number_of_stocks = len(df.columns), frontier_min_return = -0.5 , frontier_max_return = 0.2)

    nb_allindex.plot_efficient_frontier(
        title = 'NIFTYBANK Index Stock Efficient Portfolio Frontier',
        plot_save_path = 'assets/TEMPNIFTYBANK_portfolio_efficient_frontier.html',
        plot_image_save_path = 'assets/TEMPNIFTYBANK_portfolio_efficient_frontier.png',
        all_portfolio_volatility=all_portfolio_data['all_portfolio_volatility'], 
        all_portfolio_returns=all_portfolio_data['all_portfolio_returns'], 
        all_portfolio_sharpe_ratio=all_portfolio_data['all_portfolio_sharpe_ratio'], 
        max_sharpe_ratio_location=[all_portfolio_data['max_sharpe_ratio_volatility'],all_portfolio_data['max_sharpe_ratio_return']], 
        max_sharpe_ratio=all_portfolio_data['max_sharpe_ratio'],
        efficient_frontier_returns=frontier_returns,
        efficient_frontier_volatility=frontier_volatility
    )

    # For select few NIFTYBANK Stocks
    nb_select_stocks = PortfolioGenerator()

    df, df_normalised_return, df_log_daily_return = nb_select_stocks.read_csv_and_generate_df('data/nifty_bank_df_2018-4-2_to_2020-5-22.csv', 'Date', ['HDFCBANK Adjusted Close','ICICIBANK Adjusted Close','INDUSINDBK Adjusted Close','KOTAKBANK Adjusted Close','SBIN Adjusted Close'])
    
    all_portfolio_data = nb_select_stocks.generate_all_portfolios(df, 50000)

    nb_select_stocks.plot_efficient_frontier(
        title = 'NIFTYBANK [HDFC,ICICI,INDUSIND,KOTAK,SBI] Efficient Portfolio Frontier',
        plot_save_path = 'assets/TEMPNIFTYBANK_HIIKS_portfolio_efficient_frontier.html',
        plot_image_save_path = 'assets/TEMPNIFTYBANK_HIIKS_portfolio_efficient_frontier.png',
        all_portfolio_volatility=all_portfolio_data['all_portfolio_volatility'], 
        all_portfolio_returns=all_portfolio_data['all_portfolio_returns'], 
        all_portfolio_sharpe_ratio=all_portfolio_data['all_portfolio_sharpe_ratio'], 
        max_sharpe_ratio_location=[all_portfolio_data['max_sharpe_ratio_volatility'],all_portfolio_data['max_sharpe_ratio_return']], 
        max_sharpe_ratio=all_portfolio_data['max_sharpe_ratio'],
        efficient_frontier_returns=frontier_returns,
        efficient_frontier_volatility=frontier_volatility
    )