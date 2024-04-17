import pickle

import random
import akde
import info_leak
import numpy as np

#ウェブサイト数
WEBSITE_NUMBER=20
#WEBSITE_NUMBER=2
#特徴の数
#FEATURE_NUMBER=1
FEATURE_NUMBER=3043
#特徴量のインスタンス数
FEATURES_INSTANCE=1000
#FEATURES_INSTANCE=10

def input_pkl():
    #ファイル名指定
    #features_file = "/home/r-tao/info_leak/extract/features_pickle/features.pkl"
    features_file = "/Users/r-tao/info_leakage_use_kdepy/my_wefde/features_pickle/features.pkl"
    #特徴量データをバイナリファイルに書き込み
    with open(features_file,"br") as xr:
        features = pickle.load(xr)

    return features

def output_pkl(leakage):
    #ファイル名指定
    #leakage_file = "/home/r-tao/info_leak/extract/features_pickle/leakage.pkl"
    leakage_file = "/Users/r-tao/info_leakage_use_kdepy/my_wefde/features_pickle/leakage.pkl"
    #特徴量データをバイナリファイルに書き込み
    with open(leakage_file,"wb") as xr:
        pickle.dump(leakage,xr)

def main():
    #pickleファイルからfeatures配列取り出し
    #features[特徴名][webサイト番号][特徴量サンプル]作成
    features = input_pkl()
    
    #features = [[1,1,2,2,2,3,3,3,3,3],[1,1,1,1,1,2,2,2,3,3]]
        
    #各特徴の情報漏洩量(H(w)-H(W|F))
    leakage = []

    #H(W)について
    h_w = info_leak.cal_hw(WEBSITE_NUMBER)
    print("hw:" + str(h_w))
    
    #各特徴について
    for i in range(0,FEATURE_NUMBER):
        print("特徴" + str(i+1) + "個目")

        #サンプル定義
        sample_list = []
        for j in range(0,WEBSITE_NUMBER):
            #sample_list.extend(random.sample(features[j],10))
            sample_list.extend(random.sample(features[i][j],int(5000/WEBSITE_NUMBER)))
        

    #H(W|F)  ←H(W|F) = 1/k ΣH(W|F(l))
        h_wf = 0
        #各webサイトの条件付き確率密度関数(モンテカルロのpdf(wi|F(i))のやつ)
        #pdf(f|w)のリスト
        #[ウェブサイト][サンプル]の二次元配列
        con_pdf_list = []
        #pdf(f)のリスト
        total_pdf_list = []

        
        #pdf(f)用にウェブサイト関係なしに特徴を1つにまとめる
        total_features= sum(features[i],[])
        #total_features = sum(features, [])
        #print(len(total_features))
        total_pdf_list = akde.akde_cal(total_features,sample_list)

        #print("total features:" + str(total_features))
        #print("sample:" + str(sample_list))
        #print("total_pdf_list" + str(total_pdf_list))
        

        #各webサイトについて
        for j in range(0,WEBSITE_NUMBER):
            #ある特徴について考える
            feature_each_web = features[i][j]
            #feature_each_web = features[j]
            print("webサイト" + str(j+1) + "個目")
            #モンテカルロ法に使用するサンプル数(5000÷ウェブサイト数)分の確率密度を推定し、エントロピーを求める

            #print("each_web_features:" + str(feature_each_web))
            
            conditional_pdf = akde.akde_cal(feature_each_web,sample_list)
        
            con_pdf_list.append(conditional_pdf)

        #ある特徴のh(W|F)を計算
        h_wf = info_leak.cal_hwf(con_pdf_list,total_pdf_list)
        print("hwf:" + str(h_wf))

        #特徴の情報漏洩量計算
        leakage.append(h_w - h_wf)

        print("leakage 特徴" + str(i+1) + "個目:" + str(h_w-h_wf))

    #結果をpklファイルに保存
    output_pkl(leakage)

    #結果出力
    print("leakage" + str(leakage))

    """
    #packet count 追加
    for i in range(0,13):
        print("packet count:" + str(leakage[i]))
        #time statistics 追加
    for i in range(13,37):
        print("time statistics:" + str(leakage[i]))
        #ngram 追加
    for i in range(37,50):
        print("ngram:" + str(leakage[i]))
    print("...")

        #transposition 追加
    for i in range(161,170):
        print("transposition:" + str(leakage[i]))
    print("...")

        #intervalI 追加
    for i in range(765,775):
        print("intervalI:" + str(leakage[i]))
    print("...")

        #intervalII 追加
    for i in range(1365,1375):
        print("intervalII:" + str(leakage[i]))
    print("...")
        
        #intervalIII 追加
    for i in range(1967,1977):
        print("intervalIII:" + str(leakage[i]))
    print("...")
        
        #packet distribution 追加
    for i in range(2553,2563):
        print("packet distribution:"  + str(leakage[i]))
    print("...")
        
        #bursts 追加
    for i in range(2778,2789):
        print("bursts:" + str(leakage[i]))
        
        #first 20 packets 追加
    for i in range(2789,2809):
        print("first 20 packets:" + str(leakage[i]))
        
        #first 30 packets 追加
    for i in range(2809,2811):
        print("first 30 packets:" + str(leakage[i]))
        
        #last 30 packets 追加
    for i in range(2811,2813):
        print("last 30 packets:" + str(leakage[i]))
        
        #packet count per second 追加
    for i in range(2813,2823):
        print("packet count per second:" + str(leakage[i]))
    print("...")
        
        #cumul 追加
    for i in range(2939,2949):
        print("cumul:" + str(leakage[i]))
    print("...")
    """

    
main()