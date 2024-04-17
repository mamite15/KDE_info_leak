import numpy as np

def pkt_cnt_per_sec(time,size):

    #追加する特徴
    features = []
    
    count = [0] * 100

    for i in range(0,len(size)):
        t = int(np.floor(time[i]))
        if t < 100:
            count[t] = count[t] + 1
    features.extend(count)

    features.append(np.mean(count))
    features.append(np.std(count))
    features.append(np.min(count))
    features.append(np.max(count))
    features.append(np.median(count))

    buckets_num=20
    bucket = [0] * buckets_num
    for i in range(0,100):
        ib = i //(100 // buckets_num)
        bucket[ib] = bucket[ib] + count[i]

    features.extend(bucket)
    features.append(np.sum(bucket))

    return features