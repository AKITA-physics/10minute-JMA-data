# 欠損データを補間するプログラム
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# 地点データのダウンロード
df_location = pd.read_csv("file path")

for l in range(64,len(df_location)):
    no = df_location.loc[l,"no"]
    block_no = df_location.loc[l,"block_no"]
    for year in range(2022,2022+1):
        #うるう年の判定
        if year % 4 == 0:
            all_day = 366
            feb_day = 29
        else:
            all_day = 365
            feb_day = 28
            
        # 気象データのダウンロード
        file_path = "file path"
        # データが取得できる場合のみ実行する
        if os.path.exists(file_path):
            df_original = pd.read_pickle(file_path)
            
            # 新しいデータフレームの作成
            df_data = pd.DataFrame(columns = ['year','month','day','hour','minute','mean'])
            # 必要なデータだけ入れる
            df_data['year'] = df_original['year']
            df_data['month'] = df_original['month']
            df_data['day'] = df_original['day']
            df_data['mean'] = df_original['平均風速']

            # データを全てフロート型へ
            df_data["mean"] = df_data["mean"].replace('×', np.nan)
            df_data["mean"] = df_data["mean"].replace('#', np.nan)
            df_data["mean"] = df_data["mean"].replace('', "Nan")
            df_data["mean"] = df_data["mean"].replace(r' \]', np.nan, regex=True).str.replace(r' \)', '', regex=True).astype(float)
                    
            # ---------------------------------------------------------1カ月の時間ごとの平均風速をdf_averageにまとめる--------------------------------------------------------------
            # 新しいデータフレームの作成
            df_average = pd.DataFrame(columns = ['year','month','day','hour',"minute"])
            current_time = datetime(year=year, month=1, day=1, hour=0, minute=10)

            # 日時データをdf_averageに追加
            for i in range(144*all_day):
                df_average = df_average._append({
                    'year': current_time.year,
                    'month': current_time.month,
                    'day': current_time.day,
                    'hour': current_time.hour,
                    'minute': current_time.minute,
                }, ignore_index=True)
                current_time = current_time + timedelta(minutes=10)

            # 風速のカラムの列を月ごとに日数×144の行列へ分解
            split_mean_1 = pd.DataFrame(np.reshape(df_data.loc[0:144*31-1,'mean'].values, (31, 144)), columns =  [str(i) for i in range(144)])
            split_mean_2 = pd.DataFrame(np.reshape(df_data.loc[144*31:144*(31+feb_day)-1,'mean'].values, (feb_day, 144)), columns =  [str(i) for i in range(144)])
            split_mean_3 = pd.DataFrame(np.reshape(df_data.loc[144*(31+feb_day):144*(31+feb_day+31)-1,'mean'].values, (31, 144)), columns =  [str(i) for i in range(144)])
            split_mean_4 = pd.DataFrame(np.reshape(df_data.loc[144*(31+feb_day+31):144*(31+feb_day+31+30)-1,'mean'].values, (30, 144)), columns =  [str(i) for i in range(144)])
            split_mean_5 = pd.DataFrame(np.reshape(df_data.loc[144*(31+feb_day+31+30):144*(31+feb_day+31+30+31)-1,'mean'].values, (31, 144)), columns =  [str(i) for i in range(144)])
            split_mean_6 = pd.DataFrame(np.reshape(df_data.loc[144*(31+feb_day+31+30+31):144*(31+feb_day+31+30+31+30)-1,'mean'].values, (30, 144)), columns =  [str(i) for i in range(144)])
            split_mean_7 = pd.DataFrame(np.reshape(df_data.loc[144*(31+feb_day+31+30+31+30):144*(31+feb_day+31+30+31+30+31)-1,'mean'].values, (31, 144)), columns =  [str(i) for i in range(144)])
            split_mean_8 = pd.DataFrame(np.reshape(df_data.loc[144*(31+feb_day+31+30+31+30+31):144*(31+feb_day+31+30+31+30+31+31)-1,'mean'].values, (31, 144)), columns =  [str(i) for i in range(144)])
            split_mean_9 = pd.DataFrame(np.reshape(df_data.loc[144*(31+feb_day+31+30+31+30+31+31):144*(31+feb_day+31+30+31+30+31+31+30)-1,'mean'].values, (30, 144)), columns =  [str(i) for i in range(144)])
            split_mean_10 = pd.DataFrame(np.reshape(df_data.loc[144*(31+feb_day+31+30+31+30+31+31+30):144*(31+feb_day+31+30+31+30+31+31+30+31)-1,'mean'].values, (31, 144)), columns =  [str(i) for i in range(144)])
            split_mean_11 = pd.DataFrame(np.reshape(df_data.loc[144*(31+feb_day+31+30+31+30+31+31+30+31):144*(31+feb_day+31+30+31+30+31+31+30+31+30)-1,'mean'].values, (30, 144)), columns =  [str(i) for i in range(144)])
            split_mean_12 = pd.DataFrame(np.reshape(df_data.loc[144*(31+feb_day+31+30+31+30+31+31+30+31+30):144*(31+feb_day+31+30+31+30+31+31+30+31+30+31)-1,'mean'].values, (31, 144)), columns =  [str(i) for i in range(144)])
            
            # 時間ごとに平均をとる
            split_mean_1 = split_mean_1.mean()
            split_mean_2 = split_mean_2.mean()
            split_mean_3 = split_mean_3.mean()
            split_mean_4 = split_mean_4.mean()
            split_mean_5 = split_mean_5.mean()
            split_mean_6 = split_mean_6.mean()
            split_mean_7 = split_mean_7.mean()
            split_mean_8 = split_mean_8.mean()
            split_mean_9 = split_mean_9.mean()
            split_mean_10 = split_mean_10.mean()
            split_mean_11 = split_mean_11.mean()
            split_mean_12 = split_mean_12.mean()
                
            # その月の日分長くする
            split_mean_1 = np.tile(split_mean_1, 31)
            split_mean_2 = np.tile(split_mean_2, feb_day)
            split_mean_3 = np.tile(split_mean_3, 31)
            split_mean_4 = np.tile(split_mean_4, 30)
            split_mean_5 = np.tile(split_mean_5, 31)
            split_mean_6 = np.tile(split_mean_6, 30)
            split_mean_7 = np.tile(split_mean_7, 31)
            split_mean_8 = np.tile(split_mean_8, 31)
            split_mean_9 = np.tile(split_mean_9, 30)
            split_mean_10 = np.tile(split_mean_10, 31)
            split_mean_11 = np.tile(split_mean_11, 30)
            split_mean_12 = np.tile(split_mean_12, 31)
                
            # 平均データを一つにまとめる
            split_mean = np.concatenate((split_mean_1, split_mean_2, split_mean_3, split_mean_4, 
                                            split_mean_5, split_mean_6, split_mean_7, split_mean_8, 
                                            split_mean_9, split_mean_10, split_mean_11, split_mean_12))
            # df_averageに追加
            df_average['average_mean'] = split_mean
                
            # df_dataの列の長さ
            N = len(df_data)
            # カラム'mean'の欠損値の個数
            missing_num = df_data['mean'].isnull().sum()
            # カラム'mean'の欠損率
            missing_rate = (missing_num / N) * 100
            #欠損率を出力
            print(missing_rate)
            
            # 出力ファイルのパス
            output_path = os.path.join("Folder Path", '%04d_%05d_10minute_average_data.csv' %(year, block_no))
            # ファイルに書き出す
            df_average.to_csv(output_path, index=False)
            #-----------------------------------------------------------補間を行う------------------------------------------------------------------
            # 連続する np.nan の範囲を取得する処理
            nan_ranges = []
            start_nan_index = None
            nan_count = 0

            for index, value in df_data['mean'].items():
                if pd.isna(value):
                    if nan_count == 0:
                        start_nan_index = index
                    nan_count = nan_count + 1
                else:
                    if nan_count > 0:
                        nan_ranges.append((start_nan_index, index - 1))
                        nan_count = 0
            if nan_count > 0:
                nan_ranges.append((start_nan_index, len(df_data) - 1))

            # np.nanが含まれている場合
            if len(nan_ranges) > 0:
                for i, (start, end) in enumerate(nan_ranges):

                    # データフレームの生成
                    newframe = pd.DataFrame(columns=["linear","average_mean","linear_average"])
                    if start == 0:
                        df_data['mean'][start:end+1] = df_average['average_mean'][start:end+1]
                    elif end == len(df_data)-1:
                        df_data['mean'][start:end+1] = df_average['average_mean'][start:end+1]
                    else:
                        # -----------------------------------------補間する---------------------------------------------------
                        # df_dataの欠損部分を線形補間してnewframeに代入
                        newframe["linear"] = df_data.iloc[start-1:end+1+1, df_data.columns.get_loc('mean')].interpolate()
                        # インデックスナンバーをリセット
                        newframe = newframe.reset_index(drop=True)
                                
                        # df_averageの指定された行をnewframeに代入
                        newframe_average = df_average['average_mean'][start-1:end+1+1]
                        # インデックスナンバーをリセット
                        newframe_average = newframe_average.reset_index(drop=True)
                        newframe["average_mean"] =newframe_average

                        # 特定の値をnp.nanに置き換える
                        newframe_average[1:end+1] = np.nan
                        # np.nanにしたところを線形補間
                        newframe_linear_average = newframe_average.interpolate()
                        # newframeに代入
                        newframe["linear_average"] = newframe_linear_average
                                
                        #newframeの端をけずる（欠損値のみを扱うため）
                        newframe = newframe[1:len(newframe)-1]

                        # 元データの線形補完に平均データの変動を加える
                        df_data.iloc[start:end+1, df_data.columns.get_loc('mean')]\
                            = newframe["linear"]+newframe["average_mean"]-newframe["linear_average"]

            # カラム'mean'の欠損値の個数
            missing_num = df_data['mean'].isnull().sum()
            # カラム'mean'の欠損率
            missing_rate = (missing_num / N) * 100
            #欠損率を出力
            print(missing_rate)
            # np.nanが存在する行のインデックスを取得する
            # nan_indices = df_data.index[df_data['mean'].isnull()]
            # print(nan_indices)
            
            # df_dataのカラムの整理
            df_data["hour"] = df_average["hour"]
            df_data["minute"] = df_average["minute"]
            new_column_order = ['year','month','day','hour','minute','mean']
            df_data = df_data[new_column_order]

            # 出力ファイルのパス
            output_path = os.path.join("Folder Path", '%04d_%05d_10minute_interpolation.csv' %(year, block_no))
            # ファイルに書き出す
            df_data.to_csv(output_path, index=False)
            
            print(block_no, len(df_data), year)

