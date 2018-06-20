import numpy as np
import pandas as pd
import os

def get_symbols_list():
    # symbols = ['dash', 'dcr', 'eth', 'ltc', 'sc', 'str', 'xmr', 'xrp']
    symbols = ['eth', 'ltc', 'xrp', 'etc', 'dash', 'xmr', 'xem', 'fct', 'gnt', 'zec'] # used in paper
    return symbols

def get_global_price(column='close'):
    """
    Construct Global Price Matrix
    Read all json data, then take the closing price
    Insert BTC as 1 at the top row
    """
    eth = pd.read_json('./data/json/btc_eth.json')
    eth.set_index('date')
    times = pd.DataFrame(index=eth['date'])

    prices = []
    for sym in get_symbols_list():
        coin = pd.read_json('./data/json/btc_{}.json'.format(sym))
        coin = coin.set_index('date')
        coin_join = times.join(coin[column])
        coin_join = coin_join.fillna(method='bfill')
        prices.append(coin_join[column])
    prices = np.array(prices) # n x t shaped (n = num of assets, t = num of period)
    prices = np.insert(prices, 0, 1, axis=0) # (n+1) x t (btc on the top)
    return prices

def get_daily_return(prices):
    """
    Construct Price Change Matrix
    Element-wise divison of price at (t+1) with price at (t)
    """
    # changes = prices.copy()
    # changes[:, 1:] = changes[:, 1:] / changes[:, :-1]
    # return changes[:, -1]
    changes = prices.copy()
    # print(changes.shape)
    changes[:, 1:] = changes[:, 1:] / changes[:, :-1] - 1
    changes[:, 0] = 0
    return changes