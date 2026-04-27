import pandas as pd
import numpy as np
import rqdatac
import os
from dotenv import load_dotenv

load_dotenv()  # 从 .env 文件加载环境变量

class DataLoader:
    def __init__(self):
        # 优先读取环境变量，如果没有，则使用备用的硬编码 PASSWD
        self.passwd = os.getenv("PASSWD")
        if self.passwd:
            rqdatac.init('license', self.passwd)
            print("rqdatac 初始化成功")
        else:
            print("请先在 .env 文件中设置 PASSWD，再运行本单元")

    def fetch_data(self):
        # 从 settings.yaml 中读取配置
        with open('settings.yaml', 'r') as f:
            settings = {}
            for line in f:
                key, value = line.strip().split('=')
                settings[key.strip()] = eval(value.strip())
        
        start_date = settings.get("START_DATE")
        end_date = settings.get("END_DATE")
        
        cache_dir = "data"
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
            
        data_cache_path = os.path.join(cache_dir, f"market_data_{start_date}_{end_date}.csv")
        mask_cache_path = os.path.join(cache_dir, f"universe_mask_{start_date}_{end_date}.csv")

        if os.path.exists(data_cache_path):
            print(f"发现本地缓存，正在从 {data_cache_path} 加载数据 (跳过 API 拉取)...")
            data = pd.read_csv(data_cache_path, parse_dates=['date'])
            if os.path.exists(mask_cache_path):
                # 动态股票池读取，以第一列(date)为索引，并解析时间
                self.universe_mask = pd.read_csv(mask_cache_path, index_col=0, parse_dates=True)
                # 列名需要显式转换为原样，防止数字列等引起问题
            else:
                self.universe_mask = None
            return data
        
        try:
            print("正在获取中证1000历史成分股及其权重数据构建动态股票池...")
            # 获取这段时期内中证1000的成分股权重历史
            weights_df = rqdatac.index_weights('000852.XSHG', start_date=start_date, end_date=end_date)
            # 拿到所有曾是成分股的股票代码
            pool = weights_df.index.get_level_values('order_book_id').unique().tolist()
            
            # 这里构造（日期，股票代码）的0-1矩阵（掩码）
            w_df = weights_df.reset_index()
            w_df['is_member'] = 1
            mask_wide = w_df.pivot(index='date', columns='order_book_id', values='is_member')
            
            # 获取期间所有交易日来扩充时间范围，并且前向填充。
            # （因为 index_weights 是月度定期调整的数据，月中的每天成分股保持和上期一致）
            trading_dates = rqdatac.get_trading_dates(start_date, end_date)
            trading_dates = pd.to_datetime(trading_dates)
            self.universe_mask = mask_wide.reindex(trading_dates).ffill().fillna(0)
            print(f"动态股票池构建完成，包含 {len(pool)} 只不重复成分股。")
        except Exception as e:
            print("动态获取指数成分股失败，降级为使用单一天的静态股票池:", e)
            pool = rqdatac.index_components('000852.XSHG', date=end_date)
            self.universe_mask = None

        data = rqdatac.get_price(pool, start_date=start_date, end_date=end_date, fields=['close', 'open', 'volume'], adjust_type='none', expect_df=True).reset_index()

        # 确保按股票代码和时间排序
        data = data.sort_values(["order_book_id", "date"]).reset_index(drop=True)

        # -- 新增：参考用户代码进行手动前复权 --
        try:
            print("正在获取复权因子进行手动前复权...")
            ex_df = rqdatac.get_ex_factor(pool, start_date=start_date, end_date=end_date)
            if ex_df is not None and not ex_df.empty:
                # get_ex_factor 返回的 DataFrame 包含 order_book_id, ex_factor (或者 ex_cum_factor) 列，索引一般为除权除息日
                ex_df = ex_df.reset_index()
                # 寻找日期列，可能是 'ex_date' 或 'index'，我们统一重命名为 'date'
                date_col = 'ex_date' if 'ex_date' in ex_df.columns else ('date' if 'date' in ex_df.columns else ex_df.columns[0])
                
                # 提取需要的列（兼容用户代码使用 ex_factor）
                ex_series = ex_df[['order_book_id', date_col, 'ex_factor']].copy()
                ex_series.rename(columns={date_col: 'date'}, inplace=True)
                
                # 将复权因子映射到时间序列上 (左连接合并)
                data = data.merge(ex_series, on=['order_book_id', 'date'], how='left')
                
                # 分组按时间前推填充复权因子，缺失值补 1.0
                data['ex_factor'] = data.groupby('order_book_id')['ex_factor'].ffill().fillna(1.0)
                
                # 获取每只股票对应最新（期末）的累积复权因子
                latest_factor = data.groupby('order_book_id')['ex_factor'].transform('last')
                
                # 前复权公式：未复权价格 * (当期累积复权因子 / 最新累积复权因子)
                data['close'] = data['close'] * data['ex_factor'] / latest_factor
                data['open'] = data['open'] * data['ex_factor'] / latest_factor
                
                # 丢弃辅助列
                data.drop(columns=['ex_factor'], inplace=True)
                print("手动前复权完成。")
        except Exception as e:
            print(f"没有复权因子或拉取失败时保持未复权价格: {e}")

        # 再计算收益率
        data["return"] = data.groupby("order_book_id")["close"].pct_change()

        market_cap = rqdatac.get_factor(pool, 'a_share_market_val_in_circulation', start_date=start_date, end_date=end_date).reset_index()
        data = data.merge(market_cap, on=['order_book_id', 'date'], how='left')

        book_to_market = rqdatac.get_factor(pool, 'book_to_market_ratio_lf', start_date=start_date, end_date=end_date).reset_index()
        data = data.merge(book_to_market, on=['order_book_id', 'date'], how='left')

        data.rename(columns={'a_share_market_val_in_circulation': 'market_cap', 'book_to_market_ratio_lf': 'book_to_market'}, inplace=True)

        # 保存到本地缓存
        print(f"数据获取完毕，正在保存至本地缓存 {data_cache_path}...")
        data.to_csv(data_cache_path, index=False)
        if hasattr(self, 'universe_mask') and self.universe_mask is not None:
            self.universe_mask.to_csv(mask_cache_path)

        return data
    
    def get_field_wide(self, field):
        """需要算因子时调用：获取某字段的宽表"""
        # 修正：直接从 self.data 里面取出对应的字段，将 order_book_id 和 date 作为索引变为宽表
        # 不要通过 rqdatac 再次拉取网络请求，否则遇到我们自定义加工出来的 return 会报错。
        wide = self.data.set_index(['date', 'order_book_id'])[field].unstack(level=1)
        
        # 这一步非常关键：使用动态股票池掩码过滤
        # 将当天不属于中证1000的股票数据设为 NaN，它们就无法参与当天的横截面排序或标准化（避免未来函数）
        if hasattr(self, 'universe_mask') and self.universe_mask is not None:
            common_cols = wide.columns.intersection(self.universe_mask.columns)
            mask = self.universe_mask[common_cols].replace(0, np.nan)
            wide[common_cols] = wide[common_cols] * mask
            
        return wide

    def get_stock_series(self, universe):
        """回测时调用：获取某只股票的全数据（如某 order_book_id）"""
        return self.data.loc[self.data['order_book_id'] == universe]
    
    def preprocess_data(self, data):
        # 数据预处理
        data = data.dropna()  # 删除缺失值
        data = data[(data["volume"] > 0) & (data["return"].notnull())]  # 过滤掉成交量为0或收益率为NaN的行
        return data
    
