import numpy as np
import pandas as pd
import matplotlib as plt
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split


df = pd.read_excel("data/訂正版20211012_Nback.xlsx")
df_1 = df.iloc[:, :-2]  # ターゲットではない文字の正答率と反応時間を抽出
df_2 = df.drop(df.columns[[-3, -4]], axis=1)  # ターゲットである文字の正答率と反応時間を抽出
df_encoding = pd.get_dummies(df_2.loc[:, "Session"], prefix="session")

print(df_encoding.head())
print(df_2.head())

# plt.scatter(df['Mean Target.ACC'], df['Mean Target.RT'])
# plt.grid()
# plt.show()