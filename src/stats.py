import pandas as pd
from scipy.stats import kstest, entropy


df = pd.read_csv("/home/twb/Downloads/Baseline Data - cow.csv")
print(df.head())
stat, p_value = kstest(df['foundation-1x'], df['ppo-cow-deleter-1x'])
kl = entropy(df['ppge-cow-deleter'], df['ppo-cow-deleter-1x'])
print(f" Kolmogorov-Smirnov Test: statistic={stat:.4f}, p-value={p_value:.5f}")
print(f"KL-Divergence: {kl}")
