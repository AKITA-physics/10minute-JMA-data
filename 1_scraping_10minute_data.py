#10分データをスクレイピングするプログラム
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import os

def day_in_month(year, month):
    if month in [1,3,5,7,8,10,12]:
        return 31
    elif month in [4,6,9,11]:
        return 30
    elif month in [2] and year % 4 == 0:
        return 29
    elif month in [2] and year % 4 != 0:
        return 28
# 地点情報のダウンロード
df_location = pd.read_csv("地点情報に関するデータのパス")

for l in range(len(df_location)):
    no = df_location.loc[l,"no"]
    block_no = df_location.loc[l,"block_no"]
    for year in range(2022, 2022+1):
        for month in range(1, 12+1):
            for day in range(1, day_in_month(year, month) + 1):
                #result.getの引数にURLを指定する
                result = requests.get('https://www.data.jma.go.jp/obd/stats/etrn/view/10min_s1.php?prec_no='+str(no)+'&block_no='+str(block_no)+'&year='+str(year)+'&month='+str(month)+'&day='+str(day)+'&view=')
                
                #1秒ごとにリクエストを送る
                time.sleep(1)

                #data_1t_0,data_0_1bをdata_0_0として処理
                content = result.content.decode("utf-8").replace('data_1t_0','data_0_0')#.replace('data_0_1b','data_0_0')
                #BeautifulSoupの処理に解析したい文字列(result.content)と処理の種類(html.parser)を指定
                soup = BeautifulSoup(content, 'html.parser')
                #class_=data_0_0の部分を全て取り出す
                all_data = soup.find_all(class_='data_0_0')

                #以降で用いるリスト作る
                list_data = []
                #新しい変数を置く
                count = 0

                #dfという変数で一行目の見出しを定義
                df_day = pd.DataFrame(columns =  ['hour','day','month','year','現地気圧','海面気圧','降水量','気温','相対湿度','平均風速','平均風向','最大瞬間風速','風向','日照時間'])

                #日ごとのデータに入れる
                for hour in range(1,144+1):
                    #観測量データリストに追加する
                    for t in range(10):
                        list_data.append(all_data[count].text)
                        count = count + 1
                    #データフレームにkを追加し、df_monthに追加する
                    list_data.insert(0, hour)
                    list_data.insert(1, day)
                    list_data.insert(2, month)
                    list_data.insert(3, year)
                    df_day.loc[hour-1] = list_data
                    list_data = []

                # df_monthへ詰め替え
                if day == 1:
                # 1日の場合は新しいファイルを作成
                    df_month = df_day
                else:
                # 2日以降の場合は既存のDataFrameに追加
                    df_month = pd.concat([df_month, df_day], ignore_index=True)
                        
            # ファイルへの書き出し
            if month == 1:
            # 1月の場合は新しいファイルを作成
                df_all = df_month
            else:
            # 2月以降の場合は既存のDataFrameに追加
                df_all = pd.concat([df_all, df_month], ignore_index=True)
            if month == 12:
            # 12月の場合は全データをまとめたDataFrameをファイルに出力
                # 出力ファイルのパス
                output_path = os.path.join("ファイルを格納するフォルダーのパス", '%04d_%05d_10minute_surface.pkl' %(year, block_no))
                # ファイルに書き出す
                df_all.to_pickle(output_path)
            
            if month == 12:
                print(year, month, block_no, len(df_all))
            else:
                print(year, month, block_no)
