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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from tqdm import tqdm
plt.style.use('ggplot')

from bokeh.io import export_png, export_svgs
from bokeh.plotting import figure, output_file, show, save
from bokeh.layouts import grid, column, gridplot, layout, row
from bokeh.models import ColumnDataSource, Title, Legend, Circle, Line, ColorBar, LinearColorMapper, BasicTicker, PrintfTickFormatter, LabelSet, Label, Text
from bokeh.transform import linear_cmap
from bokeh.palettes import inferno, magma, viridis, gray, cividis, turbo

from correlation_matrix_generator import generate_correlation_graph

class PortfolioGenerator:
    def __init__(self):
        pass
    
    @staticmethod
    def read_csv_and_generate_df(path, index_col, columns = None):
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

    @staticmethod
    def generate_efficient_frontier(df, returns_mean, returns_covariance, number_of_portfolios = 5000):
        np.random.seed(55)
        log_ret_mean = returns_mean
        log_ret_cov = returns_covariance
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
    def plot_efficient_frontier(all_portfolio_volatility, all_portfolio_returns, all_portfolio_sharpe_ratio, max_sharpe_ratio_location, max_sharpe_ratio):
        
        title = 'NIFTYBANK Index Stock Efficient Portfolio Frontier'
        plot_width = 1000
        plot_height = 800
        axes_labels = ['Portfolio Volatility', 'Portfolio Returns']

        mapper = LinearColorMapper(palette=inferno(256), low=all_portfolio_sharpe_ratio.min(), high=all_portfolio_sharpe_ratio.max())

        source = ColumnDataSource(data=dict(all_portfolio_volatility = all_portfolio_volatility, all_portfolio_returns = all_portfolio_returns, all_portfolio_sharpe_ratio = all_portfolio_sharpe_ratio))

        p = figure(plot_width=plot_width, plot_height=plot_height, toolbar_location='right', x_axis_label=axes_labels[0], y_axis_label=axes_labels[1], tooltips=[('Sharpe Ratio', '@all_portfolio_sharpe_ratio'),('Portfolio Return', '@all_portfolio_returns'),('Portfolio Volatility', '@all_portfolio_volatility')])

        p.circle(x='all_portfolio_volatility', y='all_portfolio_returns', source=source, fill_color={'field': 'all_portfolio_sharpe_ratio', 'transform': mapper}, size=8)

        p.circle(x=max_sharpe_ratio_location[0], y=max_sharpe_ratio_location[1], fill_color='white', line_width=5, size=20)
        maximum_sharpe_ratio_portfolio = Label(x=max_sharpe_ratio_location[0], y=max_sharpe_ratio_location[1], text=f'Maximum SR: {max_sharpe_ratio}', render_mode='css',
                        border_line_color='black', border_line_alpha=1.0, background_fill_color='white', background_fill_alpha=0.8)
        p.add_layout(maximum_sharpe_ratio_portfolio)

        color_bar = ColorBar(color_mapper=mapper, major_label_text_font_size="5pt",ticker=BasicTicker(desired_num_ticks=10),formatter=PrintfTickFormatter(format="%.1f"),label_standoff=6, border_line_color=None, location=(0, 0))
        p.add_layout(color_bar, 'right')

        p.add_layout(Title(text=title, text_font_size="16pt"), 'above')
        p.title.text_font_size = '20pt'
        p.background_fill_color = "black"
        p.background_fill_alpha = 0.9
        p.xgrid.grid_line_color = None
        p.ygrid.grid_line_color = None

        output_file("bokeh_portfolio_optimizer.html")
        save(p)

if __name__ == "__main__": 
    po = PortfolioGenerator()
    df, df_normalised_return, df_log_daily_return = po.read_csv_and_generate_df('nifty_bank_df_2018-4-2_to_2020-5-22.csv', 'Date')
    po.generate_df_histograms(df_log_daily_return, 'nifty_bank_df_log_returns_histogram.png', 'NIFTYBANK Stocks Daily Log Returns')
    po.generate_df_columns_plot(df_normalised_return, 'nifty_bank_df_norm_returns.png', 'NIFTYBANK Cumulative Return')
    po.generate_and_save_correlation_matrix(df, 'nifty_bank_df_correlation.csv', 'nifty_bank_df_correlation_matrix.html')
    efficient_frontier_data = po.generate_efficient_frontier(df, df_log_daily_return.mean(), df_log_daily_return.cov() number_of_portfolios = 1000)
    po.plot_efficient_frontier(efficient_frontier_data['all_portfolio_volatility'], efficient_frontier_data['all_portfolio_returns'], efficient_frontier_data['all_portfolio_sharpe_ratio'], [efficient_frontier_data['max_sharpe_ratio_volatility'],efficient_frontier_data['max_sharpe_ratio_return']], efficient_frontier_data['max_sharpe_ratio'])