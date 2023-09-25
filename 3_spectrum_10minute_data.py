# 風速のパワースペクトルを計算するプログラム
import pandas as pd
import numpy as np
from scipy import signal
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 地点データのダウンロード
df_location = pd.read_csv("file path")
# 処理した地域をまとめる
list_block = []

for l in range(len(df_location)):
    no = df_location.loc[l,"no"]
    block_no = df_location.loc[l,"block_no"]
    for year in range(2022,2022+1):
        #気象データのダウンロード
        # 補間後のデータ
        df_original = pd.read_csv("file path")
        # 平均データ
        df_average = pd.read_csv("file path")

        # カラム'mean'の欠損値の個数
        missing_num = df_original['mean'].isnull().sum()
        # カラム'mean'の欠損率
        N = len(df_original)
        missing_rate = (missing_num / N) * 100
        # 欠損がない場合のみ処理
        if missing_rate == 0:
            #---------------------------------------------------------高速フーリエ変換----------------------------------------------------------
            signal = np.array(df_original["mean"])
            # 線形トレンドの除去
            signal_d = signal - np.polyval(np.polyfit(np.arange(len(signal)), signal, 1), np.arange(len(signal)))

            # サンプリングレート(サンプル/秒)
            sampling_rate = 1.0 / 600
            # サンプル数
            N = len(df_original)
            # 周波数軸の作成
            freq = np.fft.fftfreq(N, 1/sampling_rate)
            freq = freq[1:len(freq)//2]         
            # FFTを計算
            fft_result = np.fft.rfft(signal_d)
            fft_result = np.abs(fft_result)
            fft_result = fft_result[1:]
            
            # 移動平均の窓サイズ
            window_size = 12
            # 移動平均フィルタを適用
            fft_result = np.convolve(fft_result, np.ones(window_size)/window_size, mode='same')

            # 強度の計算
            spectrum = freq * fft_result ** 2 / sampling_rate /(N/2)
            exec(f"spectrum_{block_no}_{year} = spectrum")

            #------------------------------------------選択した範囲でのスペクトルの傾き-----------------------------------------------
            # 1日から3時間の周波数領域のみを取り出す
            selected_freq_all = [f for f in freq if  1/86400 <= f <= 1/10800]
            selected_spectrum_all = [spectrum[i] for i, f in enumerate(freq) if f in selected_freq_all]
            
            #スペクトルの最小二乗法
            x = np.log10(np.array(selected_freq_all).reshape(-1, 1))
            y = np.log10(np.array(selected_spectrum_all).reshape(-1, 1))
            regressor = LinearRegression().fit(x, y)

            # 傾きと切片の取得
            exec(f"slope_{block_no}_all = regressor.coef_[0]")
            exec(f"intercept_{block_no}_all = regressor.intercept_")
            
            # 1日から6時間の周波数領域のみを取り出す
            selected_freq_low = [f for f in freq if  1/86400 <= f <= 1/32400]
            selected_spectrum_low = [spectrum[i] for i, f in enumerate(freq) if f in selected_freq_low]
            
            #スペクトルの最小二乗法
            x = np.log10(np.array(selected_freq_low).reshape(-1, 1))
            y = np.log10(np.array(selected_spectrum_low).reshape(-1, 1))
            regressor = LinearRegression().fit(x, y)

            # 傾きと切片の取得
            exec(f"slope_{block_no}_low = regressor.coef_[0]")
            exec(f"intercept_{block_no}_low = regressor.intercept_")
            
            # 6時間から3時間の周波数領域のみを取り出す
            selected_freq_high = [f for f in freq if  1/32400 <= f <= 1/14400]
            selected_spectrum_high = [spectrum[i] for i, f in enumerate(freq) if f in selected_freq_high]
            
            #スペクトルの最小二乗法
            x = np.log10(np.array(selected_freq_high).reshape(-1, 1))
            y = np.log10(np.array(selected_spectrum_high).reshape(-1, 1))
            regressor = LinearRegression().fit(x, y)

            # 傾きと切片の取得
            exec(f"slope_{block_no}_high = regressor.coef_[0]")
            exec(f"intercept_{block_no}_high = regressor.intercept_")

            # 計算した地点をまとめる
            list_block.append(block_no)
            # 平均風速を求める
            mean_wind = df_original["mean"].mean()
            exec(f"mean_wind_{block_no} = mean_wind")
            
            print(l, block_no)
#-----------------------------------------------------平均風速とスペクトルの傾きをまとめる---------------------------------------------------
# 新しいデータフレームを作る
df_summary = pd.DataFrame(columns = ["block_no", "mean_wind", "slope_all", "slope_high", "slope_low"])
df_summary["block_no"] = list_block
for i in range(len(df_summary)):
    block_no = df_summary.loc[i, "block_no"]
    df_summary.loc[i, "mean_wind"] = eval(f"mean_wind_{block_no}")
    df_summary.loc[i, "slope_all"] = eval(f"slope_{block_no}_all")
    df_summary.loc[i, "slope_high"] = eval(f"slope_{block_no}_high")
    df_summary.loc[i, "slope_low"] = eval(f"slope_{block_no}_low")

print(df_summary)
print(len(df_summary))
#---------------------------------------------------風のスペクトルを図で見る-------------------------------------------------
plt.figure()

year = 2022
for i in range(len(df_summary)):
    block_no = df_summary.loc[i, "block_no"]
    plt.plot(freq, eval(f"spectrum_{block_no}_{year}"), color='red')
    
    slope_all = df_summary.loc[i, "slope_all"]
    slope_high = df_summary.loc[i, "slope_high"]
    slope_low = df_summary.loc[i, "slope_low"]
    plt.plot(selected_freq_all, np.power(10, slope_all*np.log10(selected_freq_all)+eval(f"intercept_{block_no}_all")), color = 'blue')
    plt.plot(selected_freq_high, np.power(10, slope_high*np.log10(selected_freq_high)+eval(f"intercept_{block_no}_high")), color = 'blue')
    plt.plot(selected_freq_low, np.power(10, slope_low*np.log10(selected_freq_low)+eval(f"intercept_{block_no}_low")), color = 'blue')
    plt.plot(selected_freq_all, np.power(10, -2/3*np.log10(selected_freq_all)-3.5), color = 'green')
    
plt.legend(loc="upper right")
plt.xlabel("f[1/s]", fontsize=14)
plt.ylabel("fE(f)[m²/s²]", fontsize=14)
plt.tick_params(labelsize=11)
plt.xscale('log')
plt.yscale('log')
plt.grid(which="major", axis="x")
plt.grid(which="minor", axis="x", linestyle="--")
plt.xlim([10**-7, 1/1200])
# plt.ylim([5*10**-3, 10*3])

# 周波数帯域に対応する時間スケールの表示
plt.text(1/(2592000), plt.ylim()[1], '1M', ha='center', va='bottom', fontsize=11)
plt.text(1/(604800), plt.ylim()[1], '1W', ha='center', va='bottom', fontsize=11)
plt.text(1/(259200), plt.ylim()[1], '3D', ha='center', va='bottom', fontsize=11)
plt.text(1/(86400), plt.ylim()[1], '1D', ha='center', va='bottom', fontsize=11)
plt.text(1/(43200), plt.ylim()[1], '12H', ha='center', va='bottom', fontsize=11)
plt.text(1/(21600), plt.ylim()[1], '6H', ha='center', va='bottom', fontsize=11)
plt.text(1/(3600), plt.ylim()[1], '1H', ha='center', va='bottom', fontsize=11)
plt.text(1/(1200), plt.ylim()[1], '20min', ha='center', va='bottom', fontsize=11)

plt.show()
#---------------------------------------------------風速とパワースペクトルの傾きの関係を図示-------------------------------------------------
plt.figure()
plt.clf()
for i in range(len(df_summary)):
    mean_wind = df_summary.loc[i, "mean_wind"]
    slope_all = df_summary.loc[i, "slope_all"]
    if i == 0:
        plt.scatter(mean_wind, slope_all, color = "green", label = "3hour~24hour")
    else:
        plt.scatter(mean_wind, slope_all, color = "green")
for i in range(len(df_summary)):
    mean_wind = df_summary.loc[i, "mean_wind"]
    slope_low = df_summary.loc[i, "slope_low"]
    if i == 0:
        plt.scatter(mean_wind, slope_low, color = "red", label = "9hour~24hour")
    else:
        plt.scatter(mean_wind, slope_low, color = "red")
for i in range(len(df_summary)):
    mean_wind = df_summary.loc[i, "mean_wind"]
    slope_high = df_summary.loc[i, "slope_high"]
    if i == 0:
        plt.scatter(mean_wind, slope_high, color = "cyan", label = "6hour~9hour")
    else:
        plt.scatter(mean_wind, slope_high, color = "cyan")
    

plt.legend(loc="upper right")
plt.xlabel("Wind Speed(m/s)", fontsize=14)
plt.ylabel("Slope", fontsize=14)
plt.tick_params(labelsize=11)
plt.show()
