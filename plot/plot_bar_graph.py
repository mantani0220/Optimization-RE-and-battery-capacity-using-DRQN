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

market_prices = extract_market_prices(market, 4370, 4370+24*2*7)
market_prices_mean = np.mean(market_prices, axis=1)
solar_radiation_all = np.load("/home/mantani/IEEE/data/sample_data_pv.npy") 
solar_radiation = solar_radiation_all[4345:4345 + 24*7]
generation_revenue = sum(market_prices_mean * solar_radiation* delta_t)

# Paths for each seed file
seeds = [1, 2, 3, 4, 5]
penalty_lists, battery_penalty_lists, reward_lists, xD_t_lists = [], [], [], []

# Load all data files in a loop
for seed in seeds:
    penalty_lists.append(pd.read_csv(f'/home/mantani/IEEE/action/episode_penalty_20241003_1651_seed_{seed}.csv'))
    battery_penalty_lists.append(pd.read_csv(f'/home/mantani/IEEE/action/episode_battery_penalty_20241003_1651_seed_{seed}.csv'))
    reward_lists.append(pd.read_csv(f'/home/mantani/IEEE/action/step_rewards_20241003_1651_seed_{seed}.csv'))
    xD_t_lists.append(pd.read_csv(f'/home/mantani/IEEE/action/episode_battery_xDt_20241003_1651_seed_{seed}.csv'))

# Calculate total penalties, battery penalties, and rewards
penalties, battery_penalties, rewards = [], [], []
for i in range(len(seeds)):
    penalties.append((penalty_lists[i].iloc[-1, :] * delta_t * market_prices_mean).sum())
    battery_penalties.append((battery_penalty_lists[i].iloc[-1, :] * delta_t).sum())
    rewards.append(reward_lists[i].iloc[-1, :].sum())

mean_penalty = np.mean(penalties)
mean_battery_penalty = np.mean(battery_penalties)
mean_reward = np.mean(rewards)

# 平均値をデータフレームに変換
data = {
    'Mean Penalty': [mean_penalty],
    'Mean Battery Penalty': [mean_battery_penalty],
    'Mean Reward': [mean_reward]
}

df = pd.DataFrame(data)

# CSVファイルとして保存
output_path = '/home/mantani/IEEE/data/average_max1.0results.csv'
df.to_csv(output_path, index=False)


categories = ['100%','100%','100%','100%','100%']

# スタック棒グラフの作成
bar_width = 0.5
x = np.arange(len(categories))  # x 軸の位置を生成

for i, seed in enumerate(seeds):
    plt.bar(x[i], generation_revenue, color='#2ca02c', width=bar_width)
    plt.bar(x[i], rewards[i], bottom=generation_revenue, color='#ff7f0e', width=bar_width)
    plt.bar(x[i], penalties[i], bottom=generation_revenue + rewards[i], color='#1f77b4', width=bar_width)
    plt.bar(x[i], battery_penalties[i], bottom=generation_revenue + rewards[i] + penalties[i], color='#d62728', width=bar_width)

# Add labels
plt.ylabel('Profit(k¥)',fontsize = 12)     
plt.xlabel('Battery Capacity',fontsize = 12)
plt.xticks(x, categories)  # Set x-axis labels
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2)

# Save the figure
plt.savefig('/home/mantani/IEEE/plot/bar_graph_1.0max.png')

# Display the graph
plt.show()