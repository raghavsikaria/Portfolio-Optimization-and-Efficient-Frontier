##########################################################
# Author: Raghav Sikaria
# LinkedIn: https://www.linkedin.com/in/raghavsikaria/
# Github: https://github.com/raghavsikaria
# Last Update: 30-5-2020
# Project: Portfolio Optimization & Efficient Frontier
# generator
# Much of the code has been taken from Jose Portilla's
# course on Udemy - [SOURCE] - 
# https://www.udemy.com/course/python-for-finance-and-trading-algorithms/
# I intend to apply my learning for the course and
# generate efficient frontier for NIFTYBANK Index Stocks
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
    
    def read_csv_and_generate_df(self, path, index_col, columns = None):
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
    def generate_df_histograms(df, path, title, bins=100, figsize=(20,20)):        
        df.hist(bins=bins, alpha=0.8, figsize=figsize)
        plt.suptitle(title, x=0.5, y=0.95, ha='center', fontsize=15)
        plt.savefig(path)
    
    @staticmethod
    def generate_df_columns_plot(df, path, title, figsize=(20,16)):
        df.plot(figsize=figsize, title = title)
        plt.savefig(path)
    
    @staticmethod
    def generate_and_save_correlation_matrix(df, path_for_correlation_df, path_for_correlation_plot):
        corr_m = df.corr()
        corr_m.to_csv(path_for_correlation_df)
        generate_correlation_graph(path_for_correlation_df, path_for_correlation_plot, plot_height=800, plot_width=1400)

    def minimize_volatility(self, weights):
        return  self.get_ret_vol_sr(weights)[1]
    
    def get_ret_vol_sr(self, weights):
        """
        Takes in weights, returns array or return,volatility, sharpe ratio
        """
        weights = np.array(weights)
        ret = np.sum(self.df_log_daily_return_mean * weights) * 252
        vol = np.sqrt(np.dot(weights.T, np.dot(self.df_log_daily_return_cov * 252, weights)))
        sr = ret/vol
        return np.array([ret,vol,sr])
  
    @staticmethod
    def check_sum(weights):
        '''
        Returns 0 if sum of weights is 1.0
        '''
        return np.sum(weights) - 1

    def generate_efficient_frontier(self, number_of_stocks, frontier_min_return, frontier_max_return, number_of_frontier_portfolios = 100):
        frontier_volatility = []
        frontier_returns = np.linspace(frontier_min_return, frontier_max_return, number_of_frontier_portfolios)
        init_guess = [1/number_of_stocks]*number_of_stocks
        bounds = tuple([(0,1) for _ in range(number_of_stocks)])

        for possible_return in tqdm(frontier_returns):
            cons = ({'type':'eq','fun': PortfolioGenerator.check_sum},
                    {'type':'eq','fun': lambda w: self.get_ret_vol_sr(w)[0] - possible_return}
                )
            
            result = minimize(self.minimize_volatility,init_guess,method='SLSQP',bounds=bounds,constraints=cons)
            frontier_volatility.append(result['fun'])
        return frontier_volatility, frontier_returns

    def generate_all_portfolios(self, df, number_of_portfolios = 5000):
        np.random.seed(55)
        log_ret_mean = self.df_log_daily_return_mean
        log_ret_cov = self.df_log_daily_return_cov
        number_of_stocks = len(df.columns)

        all_weights = np.zeros((number_of_portfolios,number_of_stocks))
        ret_arr = np.zeros(number_of_portfolios)
        vol_arr = np.zeros(number_of_portfolios)
        sharpe_arr = np.zeros(number_of_portfolios)

        for index in tqdm(range(number_of_portfolios)):
            weights = np.array(np.random.random(number_of_stocks))
            weights = weights / np.sum(weights)
            all_weights[index,:] = weights

            ret_arr[index] = np.sum(log_ret_mean * weights *252)
            vol_arr[index] = np.sqrt(np.dot(weights.T, np.dot(log_ret_cov * 252, weights)))
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
    def plot_efficient_frontier(title, plot_save_path, plot_image_save_path, all_portfolio_volatility, all_portfolio_returns, all_portfolio_sharpe_ratio, max_sharpe_ratio_location, max_sharpe_ratio, efficient_frontier_returns, efficient_frontier_volatility, plot_width = 1000, plot_height = 800, axes_labels = ['Portfolio Volatility', 'Portfolio Returns']):

        mapper = LinearColorMapper(palette=inferno(256), low=all_portfolio_sharpe_ratio.min(), high=all_portfolio_sharpe_ratio.max())

        source = ColumnDataSource(data=dict(all_portfolio_volatility = all_portfolio_volatility, all_portfolio_returns = all_portfolio_returns, all_portfolio_sharpe_ratio = all_portfolio_sharpe_ratio))

        p = figure(plot_width=plot_width, plot_height=plot_height, toolbar_location='right', x_axis_label=axes_labels[0], y_axis_label=axes_labels[1], tooltips=[('Sharpe Ratio', '@all_portfolio_sharpe_ratio'),('Portfolio Return', '@all_portfolio_returns'),('Portfolio Volatility', '@all_portfolio_volatility')])

        p.circle(x='all_portfolio_volatility', y='all_portfolio_returns', source=source, fill_color={'field': 'all_portfolio_sharpe_ratio', 'transform': mapper}, size=8)

        p.circle(x=max_sharpe_ratio_location[0], y=max_sharpe_ratio_location[1], fill_color='white', line_width=5, size=20)
        maximum_sharpe_ratio_portfolio = Label(x=max_sharpe_ratio_location[0]+0.005, y=max_sharpe_ratio_location[1]+0.02, text=f'Maximum SR ~ {round(max_sharpe_ratio, 4)}', render_mode='css', border_line_color='black', border_line_alpha=1.0, background_fill_color='white', background_fill_alpha=0.8)
        p.add_layout(maximum_sharpe_ratio_portfolio)

        p.line(x=efficient_frontier_volatility, y=efficient_frontier_returns, line_width=2, line_color='white', legend='Efficient Frontier', line_dash='dashed')

        color_bar = ColorBar(color_mapper=mapper, major_label_text_font_size="5pt",ticker=BasicTicker(desired_num_ticks=10),formatter=PrintfTickFormatter(format="%.1f"),label_standoff=6, border_line_color=None, location=(0, 0))
        p.add_layout(color_bar, 'right')

        p.add_layout(Title(text=title, text_font_size="16pt"), 'above')
        p.title.text_font_size = '20pt'
        p.background_fill_color = "black"
        p.background_fill_alpha = 0.9
        p.xgrid.grid_line_color = None
        p.ygrid.grid_line_color = None

        p.legend.background_fill_alpha = 0.5
        p.legend.background_fill_color = 'black'
        p.legend.label_text_color = 'white'
        p.legend.click_policy = 'mute'

        output_file(plot_save_path)
        save(p)
        export_png(p, filename = plot_image_save_path)

if __name__ == "__main__": 
    
    # For all NIFTYBANK Index Stocks
    nb_allindex = PortfolioGenerator()
    
    df, df_normalised_return, df_log_daily_return = nb_allindex.read_csv_and_generate_df('data/nifty_bank_df_2018-4-2_to_2020-5-22.csv', 'Date')
    
    nb_allindex.generate_df_histograms(df_log_daily_return, 'assets/NIFTYBANK_df_log_returns_histogram.png', 'NIFTYBANK Stocks Daily Log Returns')
    
    nb_allindex.generate_df_columns_plot(df_normalised_return, 'assets/NIFTYBANK_df_norm_returns.png', 'NIFTYBANK Cumulative Return')
    
    nb_allindex.generate_and_save_correlation_matrix(df, 'assets/NIFTYBANK_df_correlation.csv', 'assets/NIFTYBANK_df_correlation_matrix.html')
    
    all_portfolio_data = nb_allindex.generate_all_portfolios(df, 25000)

    frontier_volatility, frontier_returns = nb_allindex.generate_efficient_frontier(number_of_stocks = len(df.columns), frontier_min_return = -0.5 , frontier_max_return = 0.2)

    nb_allindex.plot_efficient_frontier(
        title = 'NIFTYBANK Index Stock Efficient Portfolio Frontier',
        plot_save_path = 'assets/NIFTYBANK_portfolio_efficient_frontier.html',
        plot_image_save_path = 'assets/NIFTYBANK_portfolio_efficient_frontier.png',
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
        plot_save_path = 'assets/NIFTYBANK_HIIKS_portfolio_efficient_frontier.html',
        plot_image_save_path = 'assets/NIFTYBANK_HIIKS_portfolio_efficient_frontier.png',
        all_portfolio_volatility=all_portfolio_data['all_portfolio_volatility'], 
        all_portfolio_returns=all_portfolio_data['all_portfolio_returns'], 
        all_portfolio_sharpe_ratio=all_portfolio_data['all_portfolio_sharpe_ratio'], 
        max_sharpe_ratio_location=[all_portfolio_data['max_sharpe_ratio_volatility'],all_portfolio_data['max_sharpe_ratio_return']], 
        max_sharpe_ratio=all_portfolio_data['max_sharpe_ratio'],
        efficient_frontier_returns=frontier_returns,
        efficient_frontier_volatility=frontier_volatility
    )