import os
import sys
# 将项目根目录提前添加到系统路径中，以免本目录下的其他模块引用时找不到根目录包 'factors'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import factors.cal_factors as cal_factors

def calculate_ic(factor_wide: pd.DataFrame, forward_returns_wide: pd.DataFrame, method: str = 'spearman') -> pd.Series:
    """
    计算因子的每日 IC (Information Coefficient)。
    
    参数:
    factor_wide: pd.DataFrame
        因子值宽表，index 为日期，columns 为股票代码。
    forward_returns_wide: pd.DataFrame
        未来一期收益率宽表（如 T+1 收益率），index 为日期，columns 为股票代码。
    method: str
        相关系数计算方法，默认为 'spearman' (Rank IC)。可选 'pearson' (Normal IC)。
    
    返回:
    pd.Series
        每日的因子 IC 值序列。
    """
    # 确保两张表的索引和列完全对齐
    factor_aligned, fwd_ret_aligned = factor_wide.align(forward_returns_wide, join='inner', axis=0)
    factor_aligned, fwd_ret_aligned = factor_aligned.align(fwd_ret_aligned, join='inner', axis=1)
    
    # 逐日计算横截面相关系数
    ic_series = factor_aligned.corrwith(fwd_ret_aligned, axis=1, method=method)
    return ic_series

def calculate_ir(ic_series: pd.Series) -> float:
    """
    计算因子的 IR (Information Ratio)。
    
    参数:
    ic_series: pd.Series
        每日的因子 IC 值序列。
        
    返回:
    float
        因子的 IR 值 (IC均值 / IC标准差)。
    """
    ic_series = ic_series.dropna()
    if ic_series.empty or ic_series.std() == 0:
        return 0.0
        
    ir = ic_series.mean() / ic_series.std()
    return ir

def evaluate_factor_ic_ir(factor_wide: pd.DataFrame, forward_returns_wide: pd.DataFrame, method: str = 'spearman') -> dict:
    """
    一键评估因子，返回均值 IC、IR 和每日 IC 序列。
    """
    ic_series = calculate_ic(factor_wide, forward_returns_wide, method=method)
    ir = calculate_ir(ic_series)
    
    return {
        'IC_Mean': ic_series.mean(),
        'IR': ir,
        'IC_Series': ic_series
    }

if __name__ == '__main__':
    from data_enigine import DataLoader
    
    print("1. 初始化 DataLoader 并拉取数据...")
    engine = DataLoader()
    raw_data = engine.fetch_data()
    engine.data = engine.preprocess_data(raw_data)
    
    print("2. 提取开盘价并计算下期收益率 (forward returns)...")
    open_wide = engine.get_field_wide('open')
    # 参考 backtest.py 的做法：以开盘价至次日开盘价计算 T+1 收益率
    forward_returns = open_wide.shift(-1) / open_wide - 1
    
    print("3. 计算测试因子...")
    # 使用上层已写好的 cal_mom_factor 作为测试样例
    factor_wide = cal_factors.get_spec_vol_factor(engine, window=20)
    
    print("4. 正在评估因子 IC、IR...")
    res = evaluate_factor_ic_ir(factor_wide, forward_returns)
    
    print("\n====== 因子评估结果 ======")
    print(f"IC 均值 (Rank IC): {res['IC_Mean']:.4f}")
    print(f"IR  (信息比率):     {res['IR']:.4f}")
    print(f"正 IC 比例:         {(res['IC_Series'] > 0).mean():.2%}")
    print("======================================\n")
