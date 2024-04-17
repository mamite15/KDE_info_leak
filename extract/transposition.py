import numpy as np

def trans(size):

    #追加する特徴
    features = []

    #送信パケットの最初の300個のパケット順序(送信パケットが300個あるかどうか。ない場合は0をパディング)、標準偏差、平均
    count = 0
    temp = []
    for i in range(0,len(size)):
        if size[i] < 0:
            count+=1
            features.append(i)
            temp.append(i)

        if count == 300:
            break

    for i in range(count,300):
        features.append(0)

    #標準偏差
    features.append(np.std(temp))

    #平均
    features.append(np.mean(temp))

    #受信パケットの最初の300個のパケット順序(受信パケットが300個あるかどうか。ない場合は0をパディング)、標準偏差、平均
    count = 0
    temp = []
    for i in range(0,len(size)):
        if size[i] > 0:
            count+=1
            features.append(i)
            temp.append(i)

        if count == 300:
            break

    for i in range(count,300):
        features.append(0)

    #標準偏差
    features.append(np.std(temp))

    #平均
    features.append(np.mean(temp))
    
    return features