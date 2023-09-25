import pandas as pd
import numpy as np
from scipy import signal
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# --------------------------------------------------------データのダウンロード-----------------------------------------------------------------
df_location = pd.read_csv("C:\\Users\\AKITA KOSUKE\\Box\\1_修士課程研究\\プログラム\\地点情報\\location.csv")
# 処理した地域をまとめる
list_block = []

for l in [4, 153, 154]:#range(len(df_location)):
    no = df_location.loc[l,"no"]
    block_no = df_location.loc[l,"block_no"]
    for year in range(2022,2022+1):
        #データのダウンロード
        # 補間後のデータ
        df_original = pd.read_csv("D:\\master_research\\地上データ\\10minute_data_surface\\interplolation\\"+str(year)+"_"+str(block_no)+"_10minute_interpolation.csv")
       
        # カラム'mean'の欠損値の個数
        missing_num = df_original['mean'].isnull().sum()
        # カラム'mean'の欠損率
        N = len(df_original)
        missing_rate = (missing_num / N) * 100
        # 欠損がない場合のみ処理
        if missing_rate == 0:
            #---------------------------------------------------------高速フーリエ変換----------------------------------------------------------
            # df_dataの平均風速のデータに対して観測値から平均値を引く
            signal = np.array(df_original["mean"])
            # 線形トレンドの除去
            signal_d = signal - np.polyval(np.polyfit(np.arange(len(signal)), signal, 1), np.arange(len(signal)))

            # サンプリングレート(サンプル/秒)
            sampling_rate = 1.0 / 600
            # サンプル数
            N = len(df_original)
            # 周波数軸の作成
            freq = np.fft.rfftfreq(N, 1/sampling_rate)
            freq = freq[1:]
                    
            # FFTを計算
            fft_result = np.fft.rfft(signal_d)
            fft_result = np.abs(fft_result)
            fft_result = fft_result[1:]
            
            # # リストの中の数を平均して減らす
            # def average_of_groups(lst, group_sizes):
            #     averages = []
            #     start = 0  
            #     for group_size in group_sizes:
            #         end = start + group_size
            #         group = lst[start:end]
            #         if len(group) > 0:
            #             average = sum(group) / len(group)
            #             averages.append(average)
            #         start = end
            #     return averages
            # # グループサイズのリスト
            # group_sizes = []
            # for i in range(10**5):
            #     group_sizes.append(int(10**(i/1000)))
            # # 平均を計算
            # freq = average_of_groups(freq, group_sizes)
            # fft_result = average_of_groups(fft_result, group_sizes)

            # 移動平均の窓サイズ
            window_size = 12
            # 移動平均フィルタを適用
            fft_result = np.convolve(fft_result, np.ones(window_size)/window_size, mode='same')

            # 強度の計算
            spectrum = freq * fft_result ** 2 / sampling_rate /(N/2)
            exec(f"spectrum_{block_no}_{year} = spectrum")
            
# ----------------------------------------------------多項式フィッティング------------------------------------------------------
            selected_freq = freq#[f for f in freq if  0 < f]
            selected_spectrum = [spectrum[i] for i, f in enumerate(freq) if f in selected_freq]
            
            x = np.log10(np.array(selected_freq))
            y = np.log10(np.array(selected_spectrum))

            # 近似パラメータakを算出
            coe = np.polyfit(x, y, 10)
            
            # 得られたパラメータakからカーブフィット後の波形を作成
            y_fit = np.polyval(coe, x)
            # 出力結果をまとめる
            exec(f"freq_fit_{block_no}_{year} = x")
            exec(f"spectrum_fit_{block_no}_{year} = y_fit")

#------------------------------------------方法１：選択した範囲でのスペクトルの傾き-----------------------------------------------
            # スライドするウィンドウのサイズとステップサイズを設定
            window_size = 100  # ウィンドウサイズ
            step_size = 1  # ステップサイズ
            
            # 1日から3時間の周波数領域のみを取り出す
            selected_freq = freq
            selected_spectrum_fit = [np.power(10, y_fit)[i] for i, f in enumerate(freq) if f in selected_freq]

            # 傾きと対応する中心のfreq値を格納するリスト
            slopes = []
            intercepts = []
            center_freqs = []
            slopes_limit = []
            center_freqs_limit = []

            # スライドウィンドウを移動しながら傾きを計算
            for i in range(0, len(selected_freq) - window_size, step_size):
                # ウィンドウ内のデータを取得
                window_freq = selected_freq[i:i+window_size]
                window_spectrum = selected_spectrum_fit[i:i+window_size]
                
                # リニアリグレッションを実行して傾きを計算
                x = np.log10(np.array(window_freq).reshape(-1, 1))
                y = np.log10(np.array(window_spectrum).reshape(-1, 1))
                regressor = LinearRegression().fit(x, y)

                # 傾きと切片の取得
                slope = regressor.coef_[0]
                intercept = regressor.intercept_
                
                # 傾きと中心のfreq値をリストに追加
                slopes.append(slope)
                intercepts.append(intercept)
                center_freqs.append(np.mean(window_freq))
                
                # 条件に合致する場合、傾きと中心のfreq値をリストに追加
                if -0.7 < slope < -0.62:
                    slopes_limit.append(slope)
                    center_freqs_limit.append(np.mean(window_freq))

            # 傾きが最も-2/3に近づく領域を見つける
            target_slope = -2/3  # 目標の傾き
            closest_idx = np.argmin(np.abs(np.array(slopes) - target_slope))
            
            list_block.append(block_no)
            # 平均風速を求める
            mean_wind = df_original["mean"].mean()
            exec(f"mean_wind_{block_no} = mean_wind")
            
            print(l, block_no)
            
            # 結果を表示
            exec(f"slope_{block_no} = slopes[closest_idx]")
            exec(f"intercept_{block_no} = intercepts[closest_idx]")
            exec(f"center_freq_{block_no} = center_freqs[closest_idx]")
            if center_freqs_limit != []:
                exec(f"freq_limit_first_{block_no} = center_freqs_limit[0]")
                exec(f"freq_limit_last_{block_no} = center_freqs_limit[-1]")
            else:
                exec(f"freq_limit_first_{block_no} = np.nan")
                exec(f"freq_limit_last_{block_no} = np.nan")

#-------------------------------------------------方法１：平均風速とスペクトルの傾きをまとめる---------------------------------------------------
# 新しいデータフレームを作る
df_summary = pd.DataFrame(columns = ["block_no", "mean_wind", "slope", "intercept", "center_freq", "freq_limit_first", "freq_limit_last"])
df_summary["block_no"] = list_block
for i in range(len(df_summary)):
    block_no = df_summary.loc[i, "block_no"]
    df_summary.loc[i, "mean_wind"] = eval(f"mean_wind_{block_no}")
    df_summary.loc[i, "slope"] = eval(f"slope_{block_no}")
    df_summary.loc[i, "intercept"] = eval(f"intercept_{block_no}")
    df_summary.loc[i, "center_freq"] = eval(f"center_freq_{block_no}")
    df_summary.loc[i, "freq_limit_first"] = eval(f"freq_limit_first_{block_no}")
    df_summary.loc[i, "freq_limit_last"] = eval(f"freq_limit_last_{block_no}")

print(df_summary)
print(len(df_summary))

# 慣性小領域が形成されていない場合の風速を確認する
list_nan = []
for i in range(len(df_summary)):
    if df_summary.loc[i, "freq_limit_first"] == np.nan or \
        df_summary.loc[i, "freq_limit_first"] == df_summary.loc[i, "freq_limit_last"]:
        list_nan.append(df_summary.loc[i, "mean_wind"])
print(list_nan)

#------------------------------------------方法２：選択した範囲でのスペクトルの傾き-----------------------------------------------
#             for hour in range(4,25):
#                 f_1 = 1/(hour*3600)
#                 f_2 = 1/((hour-3)*3600)
#                 # 指定した周波数領域のみを取り出す
#                 selected_freq = [f for f in freq if  f_1 < f < f_2]
#                 selected_spectrum = [np.power(10, y_fit)[i] for i, f in enumerate(freq) if f in selected_freq]

#                 # リニアリグレッションを実行して傾きを計算
#                 x = np.log10(np.array(selected_freq).reshape(-1, 1))
#                 y = np.log10(np.array(selected_spectrum).reshape(-1, 1))
#                 regressor = LinearRegression().fit(x, y)

#                 # 傾きと切片の取得
#                 slope = regressor.coef_[0]
#                 intercept = regressor.intercept_
                 
#                 # 結果を表示
#                 exec(f"slope_{block_no}_{hour} = slope")
#                 exec(f"intercept_{block_no}_{hour} = intercept")
                
#             list_block.append(block_no)
#             # 平均風速を求める
#             mean_wind = df_original["mean"].mean()
#             exec(f"mean_wind_{block_no} = mean_wind")
            
#             print(l, block_no)

# # -------------------------------------------------方法２：平均風速とスペクトルの傾きをまとめる---------------------------------------------------
# # 新しいデータフレームを作る
# df_summary = pd.DataFrame(columns = ["block_no", "mean_wind", "slope_24", "slope_23", "slope_22", "slope_21", "slope_20", "slope_19", "slope_18",
#                                      "slope_17", "slope_16", "slope_15", "slope_14", "slope_13", "slope_12", "slope_11", "slope_10",
#                                      "slope_9", "slope_8", "slope_7", "slope_6", "slope_5", "slope_4"])
# df_summary["block_no"] = list_block
# for i in range(len(df_summary)):
#     block_no = df_summary.loc[i, "block_no"]
#     df_summary.loc[i, "mean_wind"] = eval(f"mean_wind_{block_no}")
#     df_summary.loc[i, "slope_24"] = eval(f"slope_{block_no}_24")
#     df_summary.loc[i, "slope_23"] = eval(f"slope_{block_no}_23")
#     df_summary.loc[i, "slope_22"] = eval(f"slope_{block_no}_22")
#     df_summary.loc[i, "slope_21"] = eval(f"slope_{block_no}_21")
#     df_summary.loc[i, "slope_20"] = eval(f"slope_{block_no}_20")
#     df_summary.loc[i, "slope_19"] = eval(f"slope_{block_no}_19")
#     df_summary.loc[i, "slope_18"] = eval(f"slope_{block_no}_18")
#     df_summary.loc[i, "slope_17"] = eval(f"slope_{block_no}_17")
#     df_summary.loc[i, "slope_16"] = eval(f"slope_{block_no}_16")
#     df_summary.loc[i, "slope_15"] = eval(f"slope_{block_no}_15")
#     df_summary.loc[i, "slope_14"] = eval(f"slope_{block_no}_14")
#     df_summary.loc[i, "slope_13"] = eval(f"slope_{block_no}_13")
#     df_summary.loc[i, "slope_12"] = eval(f"slope_{block_no}_12")
#     df_summary.loc[i, "slope_11"] = eval(f"slope_{block_no}_11")
#     df_summary.loc[i, "slope_10"] = eval(f"slope_{block_no}_10")
#     df_summary.loc[i, "slope_9"] = eval(f"slope_{block_no}_9")
#     df_summary.loc[i, "slope_8"] = eval(f"slope_{block_no}_8")
#     df_summary.loc[i, "slope_7"] = eval(f"slope_{block_no}_7")
#     df_summary.loc[i, "slope_6"] = eval(f"slope_{block_no}_6")
#     df_summary.loc[i, "slope_5"] = eval(f"slope_{block_no}_5")
#     df_summary.loc[i, "slope_4"] = eval(f"slope_{block_no}_4")

# print(df_summary)
# print(len(df_summary))

# # 慣性小領域の位置を調べるための情報整理
# df_graph = pd.DataFrame(columns = ["block_no", "mean_wind", "freq_first", "freq_last"])
# # 新しいリストを作成
# freq_first_list = []
# freq_last_list = []
# block_no_list = []
# wind_list = []

# for i in range(len(df_summary)):
#     # 基本情報の取り出し
#     block_no = df_summary.loc[i, "block_no"]
#     mean_wind = df_summary.loc[i, "mean_wind"]
#     # 新しいリストを作成
#     selected_columns = []
    
#    # 指定の行のデータを見る
#     row = df_summary.iloc[i]
#     for column in df_summary.columns[2:]:
#         if -0.7 < row[column] <-0.62:
#             selected_columns.append(column)
            
    
#     if selected_columns != []:
#         for j in range(len(selected_columns)):
#             hour = selected_columns[j]
#             if len(hour) == 7:
#                 hour = hour[6]
#             elif len(hour) == 8:
#                 hour = hour[6:8]
            
#             hour = int(hour)
#             f_1 = 1/(hour*3600)
#             f_2 = 1/((hour-3)*3600)
        
#             block_no_list.append(block_no)
#             wind_list.append(mean_wind)
#             freq_first_list.append(f_1)
#             freq_last_list.append(f_2)
            
        
#     elif selected_columns == []:
#         block_no_list.append(block_no)
#         wind_list.append(mean_wind)
#         freq_first_list.append(np.nan)
#         freq_last_list.append(np.nan)
        
# # データフレームに格納
# df_graph["block_no"] = block_no_list
# df_graph["mean_wind"] = wind_list
# df_graph["freq_first"] = freq_first_list
# df_graph["freq_last"] = freq_last_list

# print(df_graph)

# # どれくらいのデータで慣性小領域の傾きが見られたのかをチェック
# # df_dataの列の長さ
# N = len(df_graph)
# # カラム'mean'の欠損値の個数
# missing_num = df_graph['freq_first'].isnull().sum()
# # カラム'mean'の欠損率
# missing_rate = (missing_num / N) * 100
# #欠損率を出力
# print(missing_rate)
#---------------------------------------------------風のスペクトルを図で見る-------------------------------------------------
plt.figure()
plt.clf()
year = 2022
for i in range(len(df_summary)):
    block_no = df_summary.loc[i, "block_no"]
    plt.plot(freq, eval(f"spectrum_{block_no}_{year}"), color=cm.jet(i/len(df_summary)), label = str(block_no))
    
    # フィッティング関数をプロット
    plt.plot(np.power(10, eval(f"freq_fit_{block_no}_{year}")), np.power(10, eval(f"spectrum_fit_{block_no}_{year}")), color="red")
    
    slope = df_summary.loc[i, "slope"]
    intercept = df_summary.loc[i, "intercept"]
    
    plt.plot(freq, np.power(10, slope*np.log10(freq)+intercept), color = 'blue')
    
plt.legend(loc="upper right")
plt.xlabel("f[1/s]", fontsize=14)
plt.ylabel("fE(f)[m²/s²]", fontsize=14)
plt.tick_params(labelsize=11)
plt.xscale('log')
plt.yscale('log')
plt.grid(which="major", axis="x")
plt.grid(which="minor", axis="x", linestyle="--")
plt.xlim([10**-7, 1/1200])
plt.ylim([5*10**-3, 10])

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
#---------------------------------------------------風速と慣性小領域が現れる周波数帯の関係を図で見る-------------------------------------------------
plt.figure()
plt.clf()
for i in range(len(df_summary)):
    mean_wind = df_summary.loc[i, "mean_wind"]
    center_freq = df_summary.loc[i, "center_freq"]
    slope = df_summary.loc[i, "slope"]
    
    if -0.67 < slope < -0.65:
        plt.scatter(center_freq, mean_wind, color = "green")

# plt.legend(loc="upper right")
plt.xscale('log')
plt.xlabel("Center Frequency", fontsize=14)
plt.ylabel("Wind Speed(m/s)", fontsize=14)
plt.tick_params(labelsize=11)
plt.show()

#---------------------------------------------------方法１に対応：風速と慣性小領域が現れる周波数帯の関係を図で見る-------------------------------------------------
# plt.figure()
# plt.clf()

# for i in range(len(df_summary)):

#     mean_wind = df_summary.loc[i, "mean_wind"]
#     freq_limit_first = df_summary.loc[i, "freq_limit_first"]
#     freq_limit_last = df_summary.loc[i, "freq_limit_last"]

#     freq_plot = [freq_limit_first,freq_limit_last]
#     wind_plot = [mean_wind,mean_wind]
    
#     plt.plot(freq_plot, wind_plot, color = "green")

# # plt.legend(loc="upper right")
# plt.xscale('log')
# plt.xlabel("Center Frequency", fontsize=14)
# plt.ylabel("Wind Speed(m/s)", fontsize=14)
# plt.tick_params(labelsize=11)
# plt.show()

#---------------------------------------------------方法２に対応：風速と慣性小領域が現れる周波数帯の関係を図で見る-------------------------------------------------
# plt.figure()
# plt.clf()

# for i in range(len(df_graph)):
#     mean_wind = df_graph.loc[i, "mean_wind"]
#     freq_first = df_graph.loc[i, "freq_first"]
#     freq_last = df_graph.loc[i, "freq_last"]
    
#     freq_plot = [freq_first, freq_last]
#     wind_plot = [mean_wind, mean_wind]
    
#     if freq_first > 1/86400:
#         plt.plot(freq_plot, wind_plot, color = "green")

# # plt.legend(loc="upper right")
# plt.xscale('log')
# plt.xlabel("Spectrum Gap Frequency(1/s)", fontsize=14)
# plt.ylabel("Wind Speed(m/s)", fontsize=14)
# plt.tick_params(labelsize=11)
# plt.xlim([1/86400, 1/3600])
# plt.ylim([0, 6.9])
    
# # 周波数帯域に対応する時間スケールの表示
# plt.text(1/(86400), plt.ylim()[1], '24H', ha='center', va='bottom', fontsize=11)
# plt.text(1/(43200), plt.ylim()[1], '12H', ha='center', va='bottom', fontsize=11)
# plt.text(1/(21600), plt.ylim()[1], '6H', ha='center', va='bottom', fontsize=11)
# plt.text(1/(10800), plt.ylim()[1], '3H', ha='center', va='bottom', fontsize=11)
# plt.text(1/(7200), plt.ylim()[1], '2H', ha='center', va='bottom', fontsize=11)
# plt.text(1/(3600), plt.ylim()[1], '1H', ha='center', va='bottom', fontsize=11)
# plt.show()