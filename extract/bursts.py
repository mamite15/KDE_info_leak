def bursts(size):

    #追加する特徴
    features = []

    bursts = []
    curburst = 0
    stopped = 0

    #バーストを調査
    for x in range(0,len(size)):
        if size[x] > 0:
            stopped = 0
            curburst += x
        if size[x] < 0 and stopped == 0:
            stopped = 1
        elif size[x] < 0 and stopped == 1:
            stopped = 0
            if curburst != 0:
                bursts.append(curburst)
            curburst = 0
        else:
            pass

    # バーストがない場合もあるのでifで確認
    #バースト長の最大値、平均値、総バースト数計算
    if len(bursts) != 0:
        features.append(max(bursts))
        features.append(sum(bursts) / len(bursts))
        features.append(len(bursts))
    else:
        features.append(0)
        features.append(0)
        features.append(0)

    #5パケット以上、10パケット以上、20パケット以上のバースト数計算
    counts = [0, 0, 0]
    for x in bursts:
        if x > 5:
            counts[0] += 1
        if x > 10:
            counts[1] += 1
        if x > 15:
            counts[2] += 1

    features.append(counts[0])
    features.append(counts[1])
    features.append(counts[2])

    #最初の5つのバースト追加
    for i in range(0, 5):
        try:
            features.append(bursts[i])
        except:
            features.append(0)

    return features