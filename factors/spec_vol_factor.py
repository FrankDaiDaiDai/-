import pandas as pd
import numpy as np
import sys
import os
from sklearn.linear_model import LinearRegression

# 将项目根目录添加到系统的 PATH 中以便导入上一层目录的模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_enigine import DataLoader

def cal_fama_french_3factors(data_engine):
    """
    计算 Fama-French 三因子：市场因子（MKT）、规模因子（SMB）、价值因子（HML）
    返回：以 date 为 index，order_book_id 为 columns 的因子宽表
    """
    # 获取市场收益率（MKT），这里用中证1000的收益率作为市场代理
    return_wide = data_engine.get_field_wide('return')
    mkt_return = return_wide.mean(axis=1)  # 市场平均收益率

    # 计算规模因子（SMB）和价值因子（HML）
    # 这里我们需要获取市值和账面市值比等数据来构建这两个因子
    market_cap = data_engine.get_field_wide('market_cap')  # 市值数据
    book_to_market = data_engine.get_field_wide('book_to_market')  # 账面市值比数据

    # 构建 SMB 因子：小盘股 - 大盘股
    small_cap = market_cap.rank(axis=1, ascending=True) <= (market_cap.shape[1] / 2)
    large_cap = market_cap.rank(axis=1, ascending=True) > (market_cap.shape[1] / 2)
    smb = small_cap.mean(axis=1) - large_cap.mean(axis=1)

    # 构建 HML 因子：高账面市值比 - 低账面市值比
    high_btm = book_to_market.rank(axis=1, ascending=False) <= (book_to_market.shape[1] / 3)
    low_btm = book_to_market.rank(axis=1, ascending=False) > (book_to_market.shape[1] / 3 * 2)
    hml = high_btm.mean(axis=1) - low_btm.mean(axis=1)

    # 将三个因子合并成一个 DataFrame
    factors_df = pd.DataFrame({
        'MKT': mkt_return,
        'SMB': smb,
        'HML': hml
    })

    return factors_df

def fit_fama_french_3factors(returns, factors):
    """
    对每只股票的收益率进行回归，得到每个因子的暴露度（beta）
    returns: 以 date 为 index，order_book_id 为 columns 的股票收益率宽表
    factors: 以 date 为 index 的 Fama-French 因子 DataFrame
    返回：以 order_book_id 为 index，MKT_beta、SMB_beta、HML_beta 为 columns 的 DataFrame
    """
    betas = []
    for stock in returns.columns:
        y = returns[stock].dropna()
        X = factors.loc[y.index]
        model = LinearRegression().fit(X, y)
        betas.append({
            'order_book_id': stock,
            'MKT_beta': model.coef_[0],
            'SMB_beta': model.coef_[1],
            'HML_beta': model.coef_[2]
        })

    betas_df = pd.DataFrame(betas).set_index('order_book_id')
    return betas_df

def cal_spec_vol_factor(data_engine, window=20):
    """
    计算特质波动率因子：每只股票的收益率与 Fama-French 三因子回归后的残差的标准差
    返回：以 date 为 index，order_book_id 为 columns 的特质波动率因子宽表
    """
    # 获取股票收益率宽表
    returns_wide = data_engine.get_field_wide('return')

    # 计算 Fama-French 三因子宽表
    factors_df = cal_fama_french_3factors(data_engine)
    
    # 计算残差的标准差作为特质波动率因子
    spec_vol_factor = pd.DataFrame(index=returns_wide.index, columns=returns_wide.columns)
    
    for stock in returns_wide.columns:
        y = returns_wide[stock].dropna()
        X = factors_df.loc[y.index]
        model = LinearRegression().fit(X, y)
        residuals = y - model.predict(X)
        spec_vol_factor[stock] = residuals.rolling(window).std()

    return spec_vol_factor