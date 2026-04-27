import pandas as pd
import numpy as np
import rqdatac
from factors.cal_factors import cal_mom_factor
from factors.cal_factors import get_factor_wide
from factors.spec_vol_factor import cal_fama_french_3factors, fit_fama_french_3factors
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize

class TopNFactorStrategy:
    def __init__(self, start_date, end_date, universe, data_engine, factor_name, initial_capital=1000000, top_n=5, **factor_kwargs):
        self.start_date = start_date
        self.end_date = end_date
        self.universe = universe
        self.capital = initial_capital
        self.positions = {}
        self.current_date = None
        self.top_n = top_n
        self.data_engine = data_engine
        self.factor_name = factor_name
        
        # 初始化策略时，直接一次性算出所需因子的全量宽表
        # 行是交易日期，列是 order_book_id
        self.factor_wide = get_factor_wide(self.data_engine, self.factor_name, **factor_kwargs)
        
        # 缓存收益率宽表，避免在每日调仓时重复进行 set_index 和 unstack 导致极慢的操作
        self.returns_wide = self.data_engine.get_field_wide("return")

    def select_stocks(self, date, signals, top_n=10):
        """选股与权重模块：获取打分最高的前 top_n 只股票，采用过去120天窗口生成最大化夏普的权重目标持仓"""
        if signals is None or signals.empty:
            return {}
        top_stocks = signals.nsmallest(top_n).index.tolist()
        if not top_stocks:
            return {}
        
        # 使用预先缓存的全部个股收益率宽表并截取过去120天的窗口
        returns_wide = self.returns_wide
        loc_idx = returns_wide.index.get_indexer([date], method='ffill')[0]
        
        if loc_idx < 120:
            # 窗口期不足120天，采用等权
            weight = 1.0 / len(top_stocks)
            return {stock: weight for stock in top_stocks}
            
        # 获取过去 120 天的前N只股票的收益率数据
        window_returns = returns_wide.iloc[loc_idx - 120 : loc_idx][top_stocks].fillna(0)
        
        mean_returns = window_returns.mean()
        cov_matrix = window_returns.cov()
        
        num_assets = len(top_stocks)
        
        def neg_sharpe(weights):
            p_ret = np.sum(mean_returns * weights)
            p_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            # 年化夏普通常还需要乘以 sqrt(252)，但最优化不影响方向
            return -p_ret / p_vol if p_vol > 0 else 0
            
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(num_assets))
        initial_weights = np.array(num_assets * [1. / num_assets])
        
        # 使用 SLSQP 最小化负夏普比率
        result = minimize(neg_sharpe, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
        
        if result.success:
            weights = result.x
        else:
            weights = initial_weights
            
        return {stock: weight for stock, weight in zip(top_stocks, weights)}

    def execute_trades(self, target_weights):
        """交易执行模块：简单模拟将目标权重无摩擦地覆盖当前持仓"""
        self.positions = target_weights

    def generate_signals(self, date):
        """
        每日调仓时获取当天这一截面上的所有股票因子值
        """
        try:
            # 获取单日的因子打分 Series (index 为 order_book_id)
            signals = self.factor_wide.loc[date]
            
            # 过滤涨跌停：剔除当日常规涨跌幅绝对值 >= 9.5% 的股票，防范无法买入/卖出导致的回测幻觉
            if date in self.returns_wide.index:
                valid_mask = self.returns_wide.loc[date].abs() < 0.095
                signals = signals[valid_mask]
                
            return signals.dropna()
        except KeyError:
            # 如果当天没有计算出因子数据，返回空的 Series
            return pd.Series(dtype=float)

    def rebalance(self, date):
        """
        日度调仓入口：计算信号并选取打分最高的前 top_n 只股票。
        """
        self.current_date = date
        signals = self.generate_signals(date)
        
        if not signals.empty:
            target_weights = self.select_stocks(date, signals, top_n=self.top_n)
            self.execute_trades(target_weights)

