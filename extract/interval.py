
def interval_extract(size):

    #追加する特徴
    features = []

    #受信IntervalI
    count=0
    prev=0
    for i in range(0,len(size)):
        if size[i] > 0:
            count+=1
            features.append(i-prev)
            prev=i
        if count == 300:
            break
    
    for i in range(count,300):
        features.append(0)

    #送信IntervalI
    count=0
    prev=0
    for i in range(0,len(size)):
        if size[i] < 0:
            count+=1
            features.append(i-prev)
            prev=i
        if count == 300:
            break
    
    for i in range(count,300):
        features.append(0)

    #受信IntervalII
    count=0
    prev=0
    interval_freq_in = [0] * 301
    for i in range(0,len(size)):
        if size[i] > 0:
            inv = i - prev - 1
            prev = i
            if inv > 300:
                inv = 300
            interval_freq_in[inv] += 1

    features.extend(interval_freq_in)

    #送信IntervalII
    count=0
    prev=0
    interval_freq_out = [0] * 301
    for i in range(0,len(size)):
        if size[i] < 0:
            inv = i - prev - 1
            prev = i
            if inv > 300:
                inv = 300
            interval_freq_out[inv] += 1

    features.extend(interval_freq_out)

    #受信IntervalIII
    features.extend(interval_freq_in[0:3])
    features.append(sum(interval_freq_in[3:6]))
    features.append(sum(interval_freq_in[6:9]))
    features.append(sum(interval_freq_in[9:14]))
    features.extend(interval_freq_in[14:])

    #送信IntervalIII
    features.extend(interval_freq_out[0:3])
    features.append(sum(interval_freq_out[3:6]))
    features.append(sum(interval_freq_out[6:9]))
    features.append(sum(interval_freq_out[9:14]))
    features.extend(interval_freq_out[14:])

    return features