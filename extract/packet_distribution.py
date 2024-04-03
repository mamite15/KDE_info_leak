import numpy as np

def pkt_dis(size):
    features=[]
    count = 0
    temp = []
    #パケットシーケンスを30パケットの重複しないチャンクに分割した時の最初の200チャンクの送信パケット数
    for i in range(0, min(len(size), 6000)):
        if size[i] > 0:
            count += 1
        if i % 30 == 29:
            features.append(count)
            temp.append(count)
            count = 0
    for i in range(len(size) // 30, 200):
        features.append(0)
        temp.append(0)
    #標準偏差
    features.append(np.std(temp))
    #平均
    features.append(np.mean(temp))
    #中央値
    features.append(np.median(temp))
    #最大値
    features.append(np.max(temp))

    # alternative packet distribution list (k-anonymity)
    # could be considered packet distributions with larger intervals
    num_bucket = 20
    bucket = [0] * num_bucket
    for i in range(0, 200):
        ib = i // (200 // num_bucket)
        bucket[ib] = bucket[ib] + temp[i]
    features.extend(bucket)
    features.append(np.sum(bucket))
    return features