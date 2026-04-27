import pandas as pd
import numpy as np
from factors.spec_vol_factor import cal_fama_french_3factors, cal_spec_vol_factor, fit_fama_french_3factors
import rqdatac

def cal_mom_factor(data_engine, window=20):
    """
    动量因子：过去 window 天的收益率
    返回：以 date 为 index，order_book_id 为 columns 的因子宽表
    """
    close_wide = data_engine.get_field_wide('close')
    momentum = close_wide / close_wide.shift(window) - 1
    return momentum

def get_spec_vol_factor(data_engine, window=20):
    """
    获取特质波动率因子宽表
    """
    return cal_spec_vol_factor(data_engine, window)

def winsorize(series, n=3):
    """按 n 倍标准差缩尾，将超过范围的值替换为边界值"""
    mean, std = series.mean(), series.std()
    return series.clip(mean - n * std, mean + n * std)

def zscore(series):
    """标准化：减均值除以标准差"""
    mean, std = series.mean(), series.std()
    return (series - mean) / std if std > 0 else series - mean

def get_factor_wide(data_engine, factor_name, **kwargs):
    """
    因子计算的统一调度函数。
    传入 data_engine 和 factor_name，自动调用对应的函数并返回因子宽表。
    """
    factor_funcs = {
        'momentum': cal_mom_factor,
        'spec_vol': get_spec_vol_factor
    }
    
    if factor_name not in factor_funcs:
        raise ValueError(f"未实现该因子：{factor_name}，当前支持：{list(factor_funcs.keys())}")
        
    # 调用对应的计算函数，传参 **kwargs (例如 window 等参数)
    factor_wide = factor_funcs[factor_name](data_engine, **kwargs)

    # 对因子值进行缩尾和标准化处理
    factor_wide = factor_wide.apply(winsorize).apply(zscore)
    return factor_wide