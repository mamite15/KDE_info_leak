def roundarbitrary(x,base):
    return int(base * round(float(x) / base))

def pkt_cnt_features(size):

    #追加する特徴
    features = []

    #総パケット数
    total_cnt=0
    for i in size:
        if(i != 0):
            total_cnt+=1
    features.append(total_cnt)

    inpkt_cnt=0
    outpkt_cnt=0
    for s in size:
        #受信パケット数
        if(s > 0):
            inpkt_cnt+=1
        #送信パケット数
        elif(s < 0):
            outpkt_cnt+=1
    features.append(outpkt_cnt)
    features.append(inpkt_cnt)

    #受信パケット数と総パケット数の比率
    in_ratio = inpkt_cnt / total_cnt
    features.append(in_ratio * 100)

    #送信パケット数と総パケット数の比率
    out_ratio = outpkt_cnt / total_cnt
    features.append(out_ratio * 100)

    #1~3を15刻みで四捨五入した値
    features.append(roundarbitrary(total_cnt,15))
    features.append(roundarbitrary(outpkt_cnt,15))
    features.append(roundarbitrary(inpkt_cnt,15))

    #4,5を5刻みで四捨五入した値
    features.append(roundarbitrary(in_ratio * 100 , 5))
    features.append(roundarbitrary(out_ratio * 100 , 5))

    #総/受信/送信パケットの合計サイズ Torは512byteで統一
    features.append(total_cnt * 512)
    features.append(inpkt_cnt * 512)
    features.append(outpkt_cnt * 512)

    return features