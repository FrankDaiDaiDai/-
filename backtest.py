import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import os
import rqdatac
from data_enigine import DataLoader
from strategy_module import TopNFactorStrategy

class BacktestEngine:
    def __init__(self, strategy, data_engine):
        self.strategy = strategy
        self.data_engine = data_engine
        self.portfolio_returns = None

    def run(self, fee_rate=0.001):
        # 提取回测时间轴：从因子宽表的索引获取所有交易日
        dates = self.strategy.factor_wide.index
        # 提取开盘价宽表，通过当前开盘价到次日开盘价来模拟“次日开盘执行”的收益率
        open_wide = self.data_engine.get_field_wide('open')
        res_wide = open_wide.shift(-1) / open_wide - 1
        
        daily_returns = {}
        
        print("== 开始回测 ==")
        # 剔除交易日序列的最后一天，因为最后一天买入无法计算次日开盘收益
        for date in dates[:-1]:
            old_positions = self.strategy.positions.copy()
            # 1. 盘前/盘中：在获取今天的因子和调仓前，先计算昨日调仓延续到今日的持仓收益
            today_ret = 0.0
            
            # 记录资产漂移后的当前股票实际权重（价格涨跌导致权重变化）
            drifted_weights = {}
            if old_positions and date in res_wide.index:
                ret_today = res_wide.loc[date]
                for stock, weight in old_positions.items():
                    r = ret_today[stock] if stock in ret_today.index and pd.notna(ret_today[stock]) else 0.0
                    today_ret += weight * r
                    drifted_weights[stock] = weight * (1 + r)
                
                # 归一化漂移后的权重
                sum_drifted = sum(drifted_weights.values())
                if sum_drifted != 0:
                    drifted_weights = {k: v / sum_drifted for k, v in drifted_weights.items()}

            # 2. 盘后：根据今天跑出的最新因子重新打分、调仓
            self.strategy.rebalance(date)
            new_positions = self.strategy.positions
            
            # 3. 计算调仓产生的换手率和真实交易手续费
            all_stocks = set(drifted_weights.keys()).union(set(new_positions.keys()))
            traded_weight = 0.0
            for stock in all_stocks:
                w_drifted = drifted_weights.get(stock, 0.0)
                w_new = new_positions.get(stock, 0.0)
                traded_weight += abs(w_new - w_drifted)
                
            transaction_fee = traded_weight * fee_rate
            
            # 单日最终净收益 = 组合理论收益 - 调仓手续费
            daily_returns[date] = today_ret - transaction_fee
            
        self.portfolio_returns = pd.Series(daily_returns)
        print(f"== 回测结束，共回测 {len(dates)} 个交易日 ==\n")
        
        self.evaluate_performance()

    def evaluate_performance(self, benchmark="000852.XSHG"):
        rets = self.portfolio_returns
        
        # 计算净值曲线
        nav = (1 + rets).cumprod()
        
        # 获取并计算基准收益率和净值曲线
        bm_rets = None
        bm_nav = None
        if benchmark:
            try:
                bm_px = rqdatac.get_price(benchmark, start_date=rets.index[0], end_date=rets.index[-1], fields=['close'], adjust_type='none', expect_df=True)
                if isinstance(bm_px, pd.DataFrame):
                    if 'date' in bm_px.columns:
                         bm_px = bm_px.set_index('date')
                    elif isinstance(bm_px.index, pd.MultiIndex):
                         bm_px = bm_px.reset_index(level=0, drop=True)
                         
                    bm_rets = bm_px['close'].pct_change().reindex(rets.index).fillna(0)
                    bm_nav = (1 + bm_rets).cumprod()
            except Exception as e:
                print(f"获取基准 {benchmark} 失败: {e}")
        
        def calc_metrics(r, n):
            if r is None or r.empty:
                return 0, 0, 0, 0, 0
                
            total_return = n.iloc[-1] - 1
            trading_days = len(r)
            annual_return = (1 + total_return) ** (252 / trading_days) - 1 if trading_days > 0 else 0
            
            non_zero_days = r[r != 0]
            win_rate = (r > 0).sum() / len(non_zero_days) if len(non_zero_days) > 0 else 0

            running_max = n.expanding().max()
            drawdown = n / running_max - 1
            max_drawdown = drawdown.min()
            
            rf = 0.02
            daily_rf = rf / 252
            if r.std() != 0:
                sharpe_ratio = ((r.mean() - daily_rf) / r.std()) * np.sqrt(252)
            else:
                sharpe_ratio = 0
                
            return total_return, annual_return, max_drawdown, win_rate, sharpe_ratio

        # 计算并整合数据结果文字
        s_tr, s_ar, s_mdd, s_wr, s_sr = calc_metrics(rets, nav)
        
        perf_text = (
            "====== 策略与基准绩效分析 ======\n"
            f"【策略 (含费)】\n"
            f"总收益率:   {s_tr:.2%}\n"
            f"年化收益率: {s_ar:.2%}\n"
            f"最大回撤:   {s_mdd:.2%}\n"
            f"胜 率:      {s_wr:.2%}\n"
            f"夏普比率:   {s_sr:.2f}\n"
        )
        
        if bm_nav is not None:
            b_tr, b_ar, b_mdd, b_wr, b_sr = calc_metrics(bm_rets, bm_nav)
            perf_text += (
                f"\n【基准 ({benchmark})】\n"
                f"总收益率:   {b_tr:.2%}\n"
                f"年化收益率: {b_ar:.2%}\n"
                f"最大回撤:   {b_mdd:.2%}\n"
                f"胜 率:      {b_wr:.2%}\n"
                f"夏普比率:   {b_sr:.2f}\n"
            )
            
        perf_text += "==========================\n"
        print(perf_text)
        
        # 确保 output 文件夹存在
        os.makedirs("output", exist_ok=True)
        # 将结果保存到文本文件
        with open("output/performance.txt", "w", encoding="utf-8") as f:
            f.write(perf_text)
            
        # 绘制收益率曲线
        self._plot_equity_curve(nav, bm_nav, benchmark)

    def _plot_equity_curve(self, nav, bm_nav, benchmark_name):
        fig, ax = plt.subplots(figsize=(14, 6))

        # --- 对数区间轴（Log Scale）净值曲线 ---
        ax.plot(nav.index, nav.values, label='Strategy NAV', color='firebrick', linewidth=2.5)
        if bm_nav is not None:
            ax.plot(bm_nav.index, bm_nav.values, label=f'Benchmark: {benchmark_name}', color='black', linestyle='--', linewidth=1.5)

        ax.set_yscale('log')
        # 设置格式系
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g}"))
        
        # 添加固定的主刻度
        max_nav = int(max(nav.max(), bm_nav.max() if bm_nav is not None else nav.max())) + 1
        ticks = [1] + list(range(2, max_nav + 2)) 
        ax.set_yticks(ticks)
        
        ax.set_title('Backtest Equity Curve (Log Scale)', fontsize=16)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Log NAV', fontsize=12)
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.legend(loc='upper left', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6, which='both')
        ax.axhline(1.0, color='gray', linestyle='-', linewidth=1)
        
        plt.tight_layout()
        plt.savefig("output/equity_curve.png", dpi=300)
        plt.show()

if __name__ == '__main__':
    print("1. 初始化 DataLoader 并拉取数据...")
    engine = DataLoader()
    raw_data = engine.fetch_data()
    engine.data = engine.preprocess_data(raw_data)
    
    # 2. 从 settings 读取回测基础信息
    start_date = None
    end_date = None
    universe = []
    with open('settings.yaml', 'r', encoding='utf-8') as f:
        for line in f:
            if '=' in line:
                k, v = line.strip().split('=')
                if k.strip() == 'START_DATE': start_date = eval(v.strip())
                if k.strip() == 'END_DATE': end_date = eval(v.strip())
 
    print("2. 实例化策略并计算因子宽表...")
    strategy = TopNFactorStrategy(
        start_date=start_date, 
        end_date=end_date, 
        universe=universe,
        data_engine=engine,
        factor_name='spec_vol',
        top_n=10,
        window=20
    )

    print("3. 启动回测引擎...")
    bt = BacktestEngine(strategy, engine)
    bt.run()
