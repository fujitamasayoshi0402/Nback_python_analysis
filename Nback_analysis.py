import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df = pd.read_excel("./data/訂正版20211012_Nback.xlsx")
df_1 = df.iloc[:, :-2]  # ターゲットではない文字の正答率と反応時間を抽出
df_2 = df.drop(df.columns[[-3, -4]], axis=1)  # ターゲットである文字の正答率と反応時間を抽出
df_encoding_session = pd.get_dummies(df_2.loc[:, "Session"], prefix="session")  # セッション番号においてのOne_Hotエンコーディング
df_encoding_procedure = pd.get_dummies(df_2.loc[:, "Procedure[Block]"], prefix="procedure")  # テスト内容においてのOne_Hotエンコーディング
df_encoding_sex = pd.get_dummies(df_2.loc[:, "Sex"], prefix="sex")  # 性別におけるOne_Hotエンコーディング
df_preprocess = df_2.drop(df_2.columns[[1, 2, 3]], axis=1)  # エンコードした分、カラムを消去
df_preprocessed = pd.concat([df_preprocess, df_encoding_sex, df_encoding_session, df_encoding_procedure],
                            axis=1)  # 前処理したデータを列方向に結合

# print(df_encoding_session.head())
# print(df_encoding_procedure.head())
# print(df_preprocessed.head())

# plt.scatter(df['Mean Target.ACC'], df['Mean Target.RT'])
# plt.grid()
# plt.show()

df_session1_twoBack = df_preprocessed[(df_preprocessed["session_1"] == True) & (df_preprocessed["procedure_twoBack"] == True)]
df_session2_twoBack = df_preprocessed[(df_preprocessed["session_2"] == True) & (df_preprocessed["procedure_twoBack"] == True)]
fig = plt.figure()
axes = fig.subplots(2)
x1, y1 = df_session1_twoBack.loc[:, ["Mean Target.RT"]].values, df_session1_twoBack.loc[:, ["Mean Target.ACC"]].values
x2, y2 = df_session2_twoBack.loc[:, ["Mean Target.RT"]].values, df_session2_twoBack.loc[:, ["Mean Target.ACC"]].values
model_lr = LinearRegression()
model_lr.fit(x1, y1)
model_lr.fit(x2, y2)
axes[0].scatter(x1, y1)
axes[1].scatter(x2, y2)

axes[0].plot(x1, model_lr.predict(x1), linestyle="solid")
axes[1].plot(x1, model_lr.predict(x1), linestyle="solid")
plt.show()
# print(df_session1_twoBack.head())
