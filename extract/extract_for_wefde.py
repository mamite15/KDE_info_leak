import pyshark
import ipaddress
import numpy as np
import os
import glob
import re
import pickle

import packet_count
import time_statistics
import ngram
import transposition
import interval
import packet_distribution
import bursts
import first_and_last_packets
import packet_count_per_sec
import cumul

#ウェブサイト数
WEBSITE_NUMBER=20
#特徴の数
FEATURE_NUMBER=3043
#特徴量のインスタンス数 
FEATURES_INSTANCE=1000
#トレースの長さ
TRACE_LENGTH=5000


domain_array=[]

def label(domain_name):

    check_domain = False

    for i in range(len(domain_array)):
        if domain_name == domain_array[i]:
            check_domain = True
            domain_index = i
            break
    
    if not check_domain:
        domain_array.append(domain_name)
        domain_index = len(domain_array)-1

    return domain_index


def change_trace_structure(pcap):

    #トレース格納配列T(c)={(t0,l0),(t1,l1),...,(tn,ln)}
    #trace=np.zeros((TRACE_LENGTH,2))
    trace = [[0]*2 for j in range(TRACE_LENGTH)]
    packet_number=0

    for packet in pcap:
        hexdata=packet[1]
        if(hexdata.version == "4"):#ipv4ならこちら
            ip = ipaddress.IPv4Address(hexdata.src)
            #print(ip.compressed)
            #print(ip)
            #送信元が自分自身なら負(送信)
            if(ip.compressed == '192.168.10.150' and packet_number < TRACE_LENGTH):
                #print("送信")
                time=float(packet.frame_info.time_epoch)
                trace[packet_number][0]=time
                len=-int(packet.captured_length)
                trace[packet_number][1]=len
                packet_number +=1

            #送信元が自分自身でないなら正(受信)
            elif(ip.compressed != '192.168.10.150' and packet_number < TRACE_LENGTH):
                #print("受信")
                time=float(packet.frame_info.time_epoch)
                trace[packet_number][0]=time
                len=int(packet.captured_length)
                trace[packet_number][1]=len

                packet_number +=1
            else:
                break
        else:#ipv6ならこちら
            ip = ipaddress.IPv6Address(hexdata.src)
            #print(ip.compressed)
            #送信元が自分自身なら負(送信)
            if(ip.compressed == 'fe80::cee1:d5ff:fe0d:3d69' and packet_number < TRACE_LENGTH):
                time=float(packet.frame_info.time_epoch)
                trace[packet_number][0]=time
                len=-int(packet.captured_length)
                trace[packet_number][1]=len
                packet_number +=1
            #送信元が自分自身でないなら正(受信)
            elif(ip.compressed != 'fe80::cee1:d5ff:fe0d:3d69' and packet_number < TRACE_LENGTH):
                time=float(packet.frame_info.time_epoch)
                trace[packet_number][0]=time
                len=int(packet.captured_length)
                trace[packet_number][1]=len
                packet_number +=1
            else:#パケット数が5000を超えたら終了
                break
    #print(packet_number)
    #print(trace)
    if(packet_number > 50):
        return trace
    else:
        message="error"
        return message

#特徴抽出　f={web1,f1,f2,f3,...,fn}
def create_features(time,size):

    features=[]

    #packet count 抽出
    features.extend(packet_count.pkt_cnt_features(size))
    #Time statistics 抽出
    features.extend(time_statistics.time_st(time,size))
    ##Ngram 抽出
    features.extend(ngram.ngram_extract(size))
    #Transposition 抽出
    features.extend(transposition.trans(size))
    #Interval 抽出
    features.extend(interval.interval_extract(size))
    #Packet Distribution 抽出
    features.extend(packet_distribution.pkt_dis(size))
    #Bursts 抽出
    features.extend(bursts.bursts(size))
    #First 20 Packets,First 30 Packets,Last 30 Packets 抽出
    features.extend(first_and_last_packets.f_l_pkt(size))
    #Packet Count per Second 抽出
    features.extend(packet_count_per_sec.pkt_cnt_per_sec(time,size))
    #CUMUL Features 抽出
    features.extend(cumul.cumul_features(size))

    return features

def pkl(features):
    #ファイル名指定
    features_file = "/home/r-tao/info_leak/extract/features_pickle/features.pkl"

    #特徴量データをバイナリファイルに書き込み
    with open(features_file,"wb") as xr:
        pickle.dump(features,xr)

def main():

    #featuresに追加するfのインスタンスのカウント
    count=0

    #features[webサイト番号][特徴名][特徴量サンプル]作成
    #features = [[[0 for i in range(FEATURES_INSTANCE)] for j in range(FEATURE_NUMBER)] for k in range(WEBSITE_NUMBER)]
    
    #features[特徴名][webサイト番号][特徴量サンプル]作成
    features = [[[0 for i in range(FEATURES_INSTANCE)] for j in range(WEBSITE_NUMBER)] for k in range(FEATURE_NUMBER)]
    #webサイトごとのインスタンス数カウント
    count = [0] * WEBSITE_NUMBER
    
    print("start\n")
    #読み取るpcapファイル指定
    dir_path="pcap/"
    file_list=glob.glob(os.path.join(dir_path,"*.pcap"))

    print("抽出開始\n")
    #抽出開始   
    for pcap_file in file_list:

        cap=pyshark.FileCapture(pcap_file,display_filter="tcp.port == 443")
        print(str(cap) + "の解析")
        #壊れたpcapなら無視
        #if(len(cap) < 50):
        #    continue

        #ドメイン名取り出してラベルにする
        sp=re.split("[/_]",str(cap))
        domain_index = label(sp[1])

        #トレースの構造を変更T(c)={(t0,l0),(t1,l1),...,(tn,ln)}
        trace=change_trace_structure(cap)

        cap.close()
        #こわれたpcapなら抽出しない
        if(not isinstance(trace,list)):
            continue

        #タイムスタンプと符号付きサイズ定義
        time=[]
        size=[]
        #print(len(trace))
        #print(trace[100][0])
        for i in range(0,len(trace)):
            #print("\n")
            time.append(trace[i][0])
            size.append(trace[i][1])
            #print("要素数"+i)

        #print(time)
        
        #トレースから出力した特徴の配列f={web1,f1,f2,f3,...,fn}
        #features=np.zeros(FEATURE_NUMBER+1)
        f=[]
        
        #f={web1,f1,f2,f3,...,fn}の生成
        #1つ目の要素はwebサイト番号
        print("ドメインインデックス"+str(domain_index))
        f.append(domain_index)
        f.extend(create_features(time,size))
    
        print(f)

        #packet count 追加
        for i in range(0,13):
            #features[f[0]][i][count] = f[i+1]
            features[i][f[0]][count[f[0]]] = f[i+1]
        #time statistics 追加
        for i in range(13,37):
            #features[f[0]][i][count] = f[i+1]
            features[i][f[0]][count[f[0]]] = f[i+1]
        #ngram 追加
        for i in range(37,161):
            #features[f[0]][i][count] = f[i+1]
            features[i][f[0]][count[f[0]]] = f[i+1]
        #transposition 追加
        for i in range(161,765):
            #features[f[0]][i][count] = f[i+1]
            features[i][f[0]][count[f[0]]] = f[i+1]
        #intervalI 追加
        for i in range(765,1365):
            #features[f[0]][i][count] = f[i+1]
            features[i][f[0]][count[f[0]]] = f[i+1]
        #intervalII 追加
        for i in range(1365,1967):
            #features[f[0]][i][count] = f[i+1]
            features[i][f[0]][count[f[0]]] = f[i+1]
        #intervalIII 追加
        for i in range(1967,2553):
            #features[f[0]][i][count] = f[i+1]
            features[i][f[0]][count[f[0]]] = f[i+1]
        #packet distribution 追加
        for i in range(2553,2778):
            #features[f[0]][i][count] = f[i+1]
            features[i][f[0]][count[f[0]]] = f[i+1]
        #bursts 追加
        for i in range(2778,2789):
            #features[f[0]][i][count] = f[i+1]
            features[i][f[0]][count[f[0]]] = f[i+1]
        #first 20 packets 追加
        for i in range(2789,2809):
            #features[f[0]][i][count] = f[i+1]
            features[i][f[0]][count[f[0]]] = f[i+1]
        #first 30 packets 追加
        for i in range(2809,2811):
            #features[f[0]][i][count] = f[i+1]
            features[i][f[0]][count[f[0]]] = f[i+1]
        #last 30 packets 追加
        for i in range(2811,2813):
            #features[f[0]][i][count] = f[i+1]
            features[i][f[0]][count[f[0]]] = f[i+1]
        #packet count per second 追加
        for i in range(2813,2939):
            #features[f[0]][i][count] = f[i+1]
            features[i][f[0]][count[f[0]]] = f[i+1]
        #cumul 追加
        for i in range(2939,3043):
            #features[f[0]][i][count] = f[i+1]
            features[i][f[0]][count[f[0]]] = f[i+1]

        count[f[0]] += 1
        
        print(count)

        if(count[f[0]] == 1000):
            break


    #作成したfeaturesをバイナリファイルに書き込み
    pkl(features)

    print("done.")

main()