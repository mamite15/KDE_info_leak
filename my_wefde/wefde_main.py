import pickle

import akde
import info_leak

#ウェブサイト数
#WEBSITE_NUMBER=20
WEBSITE_NUMBER=2
#特徴の数
#FEATURE_NUMBER=3043
FEATURE_NUMBER=1
#特徴量のインスタンス数
#FEATURES_INSTANCE=1000
FEATURES_INSTANCE=100

def input_pkl():
    #ファイル名指定
    features_file = "/home/r-tao/info_leak/extract/features_pickle/features.pkl"
    #特徴量データをバイナリファイルに書き込み
    with open(features_file,"br") as xr:
        features = pickle.load(xr)

    return features

def output_pkl(leakage):
    #ファイル名指定
    leakage_file = "/home/r-tao/info_leak/extract/features_pickle/leakage.pkl"

    #特徴量データをバイナリファイルに書き込み
    with open(leakage_file,"wb") as xr:
        pickle.dump(leakage,xr)

def main():
    #pickleファイルからfeatures配列取り出し
    #features[特徴名][webサイト番号][特徴量サンプル]作成
    #features = input_pkl()
    
    #"""
    one = [[1] * 20,[1] * 20]
    two = [[2] * 30,[2] * 30]
    three = [[3] * 50,[3] * 50]
    features = one
    for i in range(2):
        features[i].extend(two[i])
        features[i].extend(three[i])
    print(features)
    #"""

        
    #各特徴の情報漏洩量(H(w)-H(W|F))
    leakage = []

    #H(W)について
    h_w = info_leak.cal_hw(WEBSITE_NUMBER)


    #各特徴について
    for i in range(0,FEATURE_NUMBER):

    #H(W|F)  ←H(W|F) = 1/k ΣH(W|F(l))
        h_wf = 0
        
        #各webサイトについて
        for j in range(0,WEBSITE_NUMBER):

            #ある特徴について考える(昇順にしておく)
            #print("i:"+ str(i) + ",j:" + str(j))
            #feature_kind = sorted(features[i][j])
            feature_kind = features[j]
            #各webサイトの条件付き確率密度関数(モンテカルロのpdf(wi|F(i))のやつ)
            pdf_list = []

            #モンテカルロ法に使用するサンプル数(5000÷ウェブサイト数)分の確率密度を推定し、エントロピーを求める
            pdf_list.extend(akde.akde_cal(feature_kind,WEBSITE_NUMBER))

        #ある特徴のh(W|F)を計算
        h_wf = info_leak.cal_hwf(pdf_list)

        print("hwf:" + str(h_wf))

        #特徴の情報漏洩量計算
        leakage.append(h_w - h_wf)

    #結果をpklファイルに保存
    #output_pkl(leakage)

    #結果出力
    print("leakage" + str(leakage))

    """
    #packet count 追加
    for i in range(0,13):
        print("packet count:" + leakage[i])
        #time statistics 追加
    for i in range(13,37):
        print("time statistics:" + leakage[i])
        #ngram 追加
    for i in range(37,50):
        print("ngram:" + leakage[i])
    print("...")

        #transposition 追加
    for i in range(161,170):
        print("transposition:" + leakage[i])
    print("...")

        #intervalI 追加
    for i in range(765,775):
        print("intervalI:" + leakage[i])
    print("...")

        #intervalII 追加
    for i in range(1365,1375):
        print("intervalII:" + leakage[i])
    print("...")
        
        #intervalIII 追加
    for i in range(1967,1977):
        print("intervalIII:" + leakage[i])
    print("...")
        
        #packet distribution 追加
    for i in range(2553,2563):
        print("packet distribution:" + leakage[i])
    print("...")
        
        #bursts 追加
    for i in range(2778,2789):
        print("bursts:" + leakage[i])
        
        #first 20 packets 追加
    for i in range(2789,2809):
        print("first 20 packets:" + leakage[i])
        
        #first 30 packets 追加
    for i in range(2809,2811):
        print("first 30 packets:" + leakage[i])
        
        #last 30 packets 追加
    for i in range(2811,2813):
        print("last 30 packets:" + leakage[i])
        
        #packet count per second 追加
    for i in range(2813,2823):
        print("packet count per second:" + leakage[i])
    print("...")
        
        #cumul 追加
    for i in range(2939,2949):
        print("cumul:" + leakage[i])
    print("...")
    """

main()