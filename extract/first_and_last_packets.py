def first_20_packets(size):

    features = []

    for i in range(0,20):
        try:
            features.append(abs(size[i]))
        except:
            features.append(0)

    return features


def first_30_packets(size):

    features = []

    out_cnt=0
    in_cnt=0

    for i in range(0,30):
        if i < len(size):
            if size[i] < 0:
                out_cnt+=1
            else:
                in_cnt+=1
    
    features.append(out_cnt)
    features.append(in_cnt)

    return features


def last_30_packets(size):
    
    features = []

    out_cnt=0
    in_cnt=0

    for i in range(1,31):
        if i <= len(size):
            if size[-i] < 0:
                out_cnt+=1
            else:
                in_cnt+=1
    
    features.append(out_cnt)
    features.append(in_cnt)

    return features

def f_l_pkt(size):

    #追加する特徴
    features = []


    #最初の20パケット
    features.extend(first_20_packets(size))

    #最初の30パケットの送信/受信パケット数
    features.extend(first_30_packets(size))

    #最後の30パケットの送信/受信パケット数
    features.extend(last_30_packets(size))

    return features
