import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

delta_t = 0.25
market = pd.read_csv("/home/mantani/IEEE/data/spot_summary_2022.csv", encoding='shift_jis')
        
def extract_market_prices(market_df, start_row, end_row):
    """6列目のデータを抽出し、エピソードごとに使用"""
    market_prices = market_df.iloc[start_row:end_row, 5].astype(float).dropna().values
    prices_per_episode = []
    for i in range(0, len(market_prices) - 1, 2):
        episode_prices = market_prices[i:i+2]
        prices_per_episode.append(episode_prices)
    return np.array(prices_per_episode)

market_prices = extract_market_prices(market, 8690, 8690+24*2*7)
market_prices_mean = np.mean(market_prices, axis=1)

solar_radiation_all = np.load("/home/mantani/IEEE/data/sample_data_pv.npy") 
solar_radiation = solar_radiation_all[4345:4345 + 24*7]

generation_revenue = sum(market_prices_mean * solar_radiation* delta_t)

battery_cost_year = 6000 #1年間で6000円/kWh相当
battery_cost_week = battery_cost_year*7/365
battery_cost = battery_cost_week * max(solar_radiation)

data_1 = pd.read_csv("/home/mantani/IEEE/data/average_max0.1results.csv")
data_2 = pd.read_csv("/home/mantani/IEEE/data/average_max0.2results.csv")
data_3 = pd.read_csv("/home/mantani/IEEE/data/average_max0.3results.csv")
data_4 = pd.read_csv("/home/mantani/IEEE/data/average_max0.4results.csv")
data_5 = pd.read_csv("/home/mantani/IEEE/data/average_max0.5results.csv")
data_6 = pd.read_csv("/home/mantani/IEEE/data/average_max0.6results.csv")
data_7 = pd.read_csv("/home/mantani/IEEE/data/average_max0.7results.csv")
data_8 = pd.read_csv("/home/mantani/IEEE/data/average_max0.8results.csv")
data_9 = pd.read_csv("/home/mantani/IEEE/data/average_max0.9results.csv")
data_10 = pd.read_csv("/home/mantani/IEEE/data/average_max1.0results.csv")

data_sets = [
    data_1, data_2, data_3, data_4, data_5,
    data_6, data_7, data_8, data_9, data_10
]

categories = ['10%','20%','30%','40%','50%','60%','70%','80%','90%','100%']
# スタック棒グラフの作成
bar_width = 0.5
x = np.arange(len(categories))  # x 軸の位置を生成

# スタック棒グラフの作成
for i in range(10):
    # データを取り出す
    penalty_last_total = data_sets[i].iloc[:, 0].tolist()
    battery_penalty_last_total = data_sets[i].iloc[:, 1].tolist()
    reward_total = data_sets[i].iloc[:, 2].tolist()
    battery_cost_factor = (i + 1) * 0.1  # バッテリーコスト係数

    # 各バースタックをプロット
    plt.bar(x[i], generation_revenue, color='#2ca02c', width=bar_width)
    plt.bar(x[i], reward_total, bottom=generation_revenue, color='#ff7f0e', width=bar_width)
    plt.bar(x[i], penalty_last_total, bottom=generation_revenue + reward_total, color='#1f77b4', width=bar_width)
    plt.bar(x[i], battery_penalty_last_total, bottom=generation_revenue + reward_total + penalty_last_total, color='#d62728', width=bar_width)
    plt.bar(x[i], -battery_cost * battery_cost_factor, 
            bottom=generation_revenue + reward_total + penalty_last_total + battery_penalty_last_total, 
            label='Battery_cost' if i == 0 else "", color='gray', width=bar_width, edgecolor='black', fill=False, hatch='//')

plt.ylabel('Profit(k¥)',fontsize = 12)     
plt.xlabel('Battery Capacity',fontsize = 12)
plt.xticks(x, categories)  # Set x-axis labels
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=7)

# Save the figure
plt.savefig('/home/mantani/IEEE/plot/bar_graph_average.png')

# Display the graph
plt.show()