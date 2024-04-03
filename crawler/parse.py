import pyshark
import ipaddress
import numpy as np
import os
import glob
import nest_asyncio
import re
import pickle

def label(domain_name):
    label_number=0
    print(domain_name)
    if(str(domain_name) == "www.google.com"):
        label_number=1
    elif(str(domain_name) == "www.youtube.com"):
        label_number=2
    elif(str(domain_name) == "www.amazon.co.jp"):
        label_number=3
    elif(str(domain_name) == "www.twitter.com"):
        label_number=4
    else:
        label_number=5

    return label_number

def src_or_dst(file_list,vec,domain,traffic_number):
    #1つずつpcapファイルを読み取って解析
    for pcap_file in file_list:
        packet_number=0
        cap=pyshark.FileCapture(pcap_file,display_filter="tcp.port == 443")
        print(str(cap) + "の解析")
        #ドメイン名取り出してラベルにする
        sp=re.split("[/_]",str(cap))
        domain[traffic_number]=label(sp[1])

        for packet in cap:
            hexdata=packet[1]
            if(hexdata.version == "4"):#ipv4ならこちら
                ip = ipaddress.IPv4Address(hexdata.src)
                #送信元が自分自身なら1をvecに入れる
                if(ip.compressed == "192.168.10.150" and packet_number < 5000):
                    vec[traffic_number][packet_number]=1
                    packet_number+=1
                #送信元が自分自身でないなら-1をvecにいれる
                elif(ip.compressed != "192.168.10.150" and packet_number < 5000):
                    vec[traffic_number][packet_number]=-1
                    packet_number+=1
                else:
                    break
            else:#ipv6ならこちら
                ip = ipaddress.IPv6Address(hexdata.src)
                #送信元が自分自身なら1をvecに入れる
                if(ip.compressed == "fe80::cee1:d5ff:fe0d:3d69" and packet_number < 5000):
                    vec[traffic_number][packet_number]=1
                    packet_number+=1
                #送信元が自分自身でないなら-1をvecにいれる
                elif(ip.compressed != "fe80::cee1:d5ff:fe0d:3d69" and packet_number < 5000):
                    vec[traffic_number][packet_number]=-1
                    packet_number+=1
                else:#パケット数が5000を超えたら終了
                    break
        print("特徴")
        print(vec[traffic_number])
        print("ラベル")
        print(domain[traffic_number])
        print()
        traffic_number+=1
    
        cap.close()
    return vec, domain, traffic_number

def pkl(traffic_number,vec,domain):
    #ファイル名指定
    x_train_file = "/r-tao/my_fingerprinting/deeplearning/pickle/x_train.pkl"
    y_train_file = "/r-tao/my_fingerprinting/deeplearning/pickle/y_train.pkl"
    x_valid_file = "/r-tao/my_fingerprinting/deeplearning/pickle/x_valid.pkl"
    y_valid_file = "/r-tao/my_fingerprinting/deeplearning/pickle/y_valid.pkl"
    x_test_file = "/r-tao/my_fingerprinting/deeplearning/pickle/x_test.pkl"
    y_test_file = "/r-tao/my_fingerprinting/deeplearning/pickle/y_test.pkl"
    
    #学習データ数計算
    train_number=(traffic_number//10)*8
    #検証データ数計算
    valid_number=(traffic_number-train_number)//2
    #テストデータ数計算
    test_number=traffic_number-valid_number-train_number
    print("総データ数:" + str(traffic_number))
    print("訓練データ数:" + str(train_number))
    print("検証データ数:" + str(valid_number))
    print("テストデータ数:" + str(test_number))
    #データセットを各種類のデータに分割
    x_train, x_valid, x_test = np.split(vec,[train_number,train_number+valid_number])
    y_train, y_valid, y_test = np.split(domain,[train_number,train_number+valid_number])
    #特徴量データをバイナリファイルに書き込み
    with open(x_train_file,"wb") as xtr:
        pickle.dump(x_train,xtr)
    with open(x_valid_file,"wb") as xv:
        pickle.dump(x_valid,xv)
    with open(x_test_file,"wb") as xte:
        pickle.dump(x_test,xte)
    #ラベルをバイナリファイルに書き込み
    with open(y_train_file,"wb") as ytr:
        pickle.dump(y_train,ytr)
    with open(y_valid_file,"wb") as yv:
        pickle.dump(y_valid,yv)
    with open(y_test_file,"wb") as yte:
        pickle.dump(y_test,yte)

def parse_main(trace_cnt):
    traffic_number=0
    nest_asyncio.apply()
    #特徴量初期化
    vec=np.zeros((trace_cnt,5000))
    #ラベル初期化
    domain=np.zeros(trace_cnt)
    #読み取るpcapファイル指定
    dir_path="pcap/"
    file_list=glob.glob(os.path.join(dir_path,"*.pcap"))
    #パケット解析
    vec,domain,traffic_number=src_or_dst(file_list,vec,domain,traffic_number)

    print("結果:")
    print("特徴")
    print(vec)
    print("ラベル")
    print(domain)
    #バイナリファイルに変換
    pkl(traffic_number,vec,domain)