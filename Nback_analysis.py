import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split


df = pd.read_excel("./data/訂正版20211012_Nback.xlsx")
df_1 = df.iloc[:, :-2]  # ターゲットではない文字の正答率と反応時間を抽出
df_2 = df.drop(df.columns[[-3, -4]], axis=1)  # ターゲットである文字の正答率と反応時間を抽出
df_encoding_session = pd.get_dummies(df_2.loc[:, "Session"], prefix="session")  # セッション番号においてのOne_Hotエンコーディング
df_encoding_procedure = pd.get_dummies(df_2.loc[:, "Procedure[Block]"], prefix="procedure")  # テスト内容においてのOne_Hotエンコーディング
df_encoding_sex = pd.get_dummies(df_2.loc[:, "Sex"], prefix="sex")  # 性別におけるOne_Hotエンコーディング
df_preprocess = df_2.drop(df_2.columns[[1, 2, 3]], axis=1)
df_preprocessed = pd.concat([df_preprocess, df_encoding_sex, df_encoding_session, df_encoding_procedure], axis=1)

# print(df_encoding_session.head())
# print(df_encoding_procedure.head())
print(df_preprocessed.head())

# plt.scatter(df['Mean Target.ACC'], df['Mean Target.RT'])
# plt.grid()
# plt.show()