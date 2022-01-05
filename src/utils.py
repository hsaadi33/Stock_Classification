import os
import os.path
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

import numpy as np                            
import pandas as pd
import yfinance as yf
from pandas_datareader import data as pdr
from hurst import compute_Hc
from tsfresh import extract_features
from tsfresh.feature_extraction import extract_features, EfficientFCParameters
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from sklearn.calibration import calibration_curve


def download_stocks(directory_data,stocks,start_year,end_year):
    """Downloads stocks from yahoo finance, and returns a list of missing stocks that cannot be downloaded.
    Args:
        directory_data: Data directory. (str)
        stocks: List of S&P 500 stocks symbols. (str list)
        start_year: Start year to get data from. (str)
        end_year: Last year to not get data from. (str)
    Returns:
        missing_stocks: List of missing stocks that cannot be downloaded. (str list)
    """
    missing_stocks = []
    for stock in stocks:
        if os.path.exists(os.path.join(directory_data,f'{stock}.pkl')):
            continue
        try:
            df = pdr.get_data_yahoo(stock,start_year,end_year)
            df.to_pickle(os.path.join(directory_data,f"{stock}.pkl"))
        except:
            missing_stocks.append(stock)
    
    return missing_stocks

def load_preprocess(directory_data,directory_inter,df_stocks_filename,tickers_sector):
    """Loads the data, add columns HL, OC, id, and time, and pickles and returns the resulting dataframe.
    Args:
        directory_data: Data directory. (str)
        directory_inter: Directory to save the resulting dataframe. (str)
        df_stocks_filename: Filename of df_stocks. (str)
        tickers_sector: Dataframe containing a stock ticker and its corresponding GICS Sector. (dataframe)
    Returns:
        df_stocks: Dataframe that includes HL, OC, id, and time columns, and in a format compatible with tsfresh. (dataframe)
    """
    if os.path.exists(os.path.join(directory_inter,df_stocks_filename)):
        df_stocks = pd.read_pickle(os.path.join(directory_inter,df_stocks_filename))
        return df_stocks
    
    stocks_files = sorted([filename.split('.')[0] for filename in os.listdir(directory_data) if filename.endswith('.pkl')])
    df_stocks = pd.DataFrame(columns=['stock','sector','df'])
    for stock_file in stocks_files:
        df = pd.read_pickle(os.path.join(directory_data,f'{stock_file}.pkl'))
        df['HL'] = df['High'] - df['Low']
        df['OC'] = df['Open'] - df['Close']
        df['id'] = stock_file
        df['time'] = df.reset_index().index + 1
        df_filtered = df[['HL','OC','Adj Close','Volume','id','time']].copy()
        df_stocks = df_stocks.append({'stock': stock_file, 
                                      'sector': tickers_sector[tickers_sector['Symbol']==stock_file]['GICS Sector'].values[0],
                                      'df': df_filtered},ignore_index=True)
    
    df_stocks.to_pickle(os.path.join(directory_inter,df_stocks_filename))
    return df_stocks


def feature_generation(df_stocks,fc_parameters,tickers_sector,directory_inter,df_final_filename):
    """Generates tsfresh features and adds Hurst exponent feature on HL, OC, Adj Close, and Volume columns, and
        pickles and returns a new dataframe containing this information.
    Args:
        df_stocks: Dataframe that includes HL, OC, id, and time columns, and in a format compatible with tsfresh. (dataframe)
        fc_parameters: Tsfresh parameters to be calculated. (dict)
        tickers_sector: Dataframe containing a stock ticker and its corresponding GICS Sector. (dataframe)
        directory_inter: Directory to save the resulting dataframe. (str) 
        df_final_filename: Filename of df_final to be save in directory_inter. (str)
    Returns:
        df_final: Dataframe containing all generated features with stocks tickers and GICS sectors. (dataframe)
    """
    
    if os.path.exists(os.path.join(directory_inter,df_final_filename)):
        df_final = pd.read_pickle(os.path.join(directory_inter,df_final_filename))
        return df_final
    
    df_features = df_stocks['df'].apply(lambda stock: extract_features(stock,
                                                                       default_fc_parameters=fc_parameters, 
                                                                       column_id="id", 
                                                                       column_sort="time",
                                                                       disable_progressbar=True))
    
    df_features = pd.concat(df_features.values.tolist()).reset_index()
    df_features.rename(columns={'index':'Symbol'},inplace=True)
    
    # Add Hurst exponent as a feature
    df_features['Volume__hurst'] = df_stocks['df'].apply(lambda stock: compute_Hc(stock['Volume'],kind='change')[0])
    df_features['Adj Close__hurst'] = df_stocks['df'].apply(lambda stock: compute_Hc(stock['Adj Close'],kind='change')[0])
    df_features['OC__hurst'] = df_stocks['df'].apply(lambda stock: compute_Hc(stock['OC'],kind='change')[0])
    df_features['HL__hurst'] = df_stocks['df'].apply(lambda stock: compute_Hc(stock['HL'],kind='change')[0])
    
    # Merge df_features and tickers_sector to obtain the GICS sector for each stock
    df_final = pd.merge(df_features,tickers_sector, on=['Symbol','Symbol'],how='inner')
    
    df_final.to_pickle(os.path.join(directory_inter,df_final_filename))
    
    return df_final

def plot_boxplots(df,useful_features,label,num_rows):
    """Plots boxplots for each useful feature for all GICS sectors
    Args:
        df: Dataframe containing all generated features with stocks tickers and GICS sectors. (dataframe)
        useful_features: A group of chosen features. (str list)
        num_rows: Number of rows in subplots. (int)
    """
    plt.style.use('default')
    fig, axs = plt.subplots(ncols=1, nrows = num_rows, figsize=(10,140),constrained_layout=True)
    #fig.tight_layout()
    axs = axs.flatten()
    i = 0
    #sns.set(rc={'figure.figsize':(2,10)})
    for feature in useful_features:
        df_temp = df.groupby(label)[feature].apply(list)
        ax = sns.boxplot(data=df_temp, width=.3,orient="h",showmeans=True,
                         meanprops={"marker": "o", "markeredgecolor": "yellow","markersize": "5"}, 
                         whis=[0,100], ax = axs[i])
        ax.set_yticklabels(df_temp.index)
        ax.grid(which='both', axis='both')
        ax.set_title(feature)
        ax.tick_params(axis='y', pad= -2)
        ax.yaxis.labelpad = -2
        ax.tick_params(axis='both', which='major', labelsize=10)
        i += 1
    plt.show()
    

def plot_calibration_curves(model,labelencoder,X_test,y_test_enc,mapping,sectors):
    """Plots calibration curves for one vs rest classification problem
    Args:
        model: Model to plot calibration curves for.
        labelencoder: Scikit-learn instance of a label encoder.
        X_test: Features test set.
        y_test_enc: Label test set after applying label encoder.
        mapping: Dictionary that maps integers to GICS Sectors from label encoder classes. (dict)
        sectors: A list of GICS Sectors. (str list)
    """
    sectors_num = len(sectors)
    plt.style.use('default')
    plt.plot([0, 1], [0, 1], linestyle = '--', label = 'Ideally Calibrated')
    for sector in range(sectors_num):
        prob_pos = model.predict_proba(X_test)[:, sector]
        y_test_plot = [int(sector==num) for num in y_test_enc]
        fraction_of_positives, mean_predicted_value = calibration_curve(y_test_plot, prob_pos, n_bins=5)
        plt.plot(mean_predicted_value, fraction_of_positives, "s-", label = f"{mapping[sector]}")
        plt.ylabel("The proportion of samples whose class is the positive class")
        plt.xlabel("The mean predicted probability in each bin")
    
    plt.legend(bbox_to_anchor=(1.35, 1), loc='upper right')
    plt.show() 
    
def train_xgboost(directory_inter,xgboost_filename,params,X_train,y_train_enc):
    """ Split the data, and either trains an xgboost model and saves it or loads a trained one
    Args:
        directory_inter: Directory to save or load an xgboost model. (str) 
        xgboost_filename: Filename of xgboost model to be saved om or loaded from directory_inter. (str)
        params: Xgboost parameters for hyperparameter tuning. (dict)
        X_train: Features to train on. (dataframe)
        y_train_enc: Labels of GICS sectors to train on. (dataframe) 
    Returns:
        xgb_cl: Trained xgboost model. (model)
    """
    
    if os.path.exists(os.path.join(directory_inter,xgboost_filename)):
        xgb_cl = pd.read_pickle(os.path.join(directory_inter,xgboost_filename))
    else:    
        # Initialize classifier
        xgb_cl = xgb.XGBClassifier()

        # Grid Search for hyperparamter tuning
        grid_search = GridSearchCV(estimator = xgb_cl,
                                   param_grid=params,
                                   scoring = 'balanced_accuracy',
                                   cv=5) 

        grid_search.fit(X_train,y_train_enc)

        xgb_cl = xgb.XGBClassifier(**grid_search.best_params_)
        xgb_cl.fit(X_train,y_train_enc)

        # Save (pickle) the xgboost model
        pickle.dump(xgb_cl, open(os.path.join(directory_inter,xgboost_filename), 'wb'))
    
    return xgb_cl