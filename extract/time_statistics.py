import numpy as np

def max_ave_std_percentile75(times):
    #パケット間の時間
    res = []

    for i in range(1,len(times)):
        prev = times[i-1]
        cur = times[i]
        res.append(cur-prev)

    if(len(res) == 0):
        #0はX
        return 0, 0, 0, 0
    else:
        return np.max(res), np.mean(res), np.std(res), np.percentile(res,75)
    
def timestamp(times):
    return np.percentile(times,25), np.percentile(times,50), np.percentile(times,75), np.percentile(times,100)

def time_st(time,size):

    #追加する特徴サイズ
    features = []

    #総パケット、送信パケット、受信パケットの時間
    total = []
    time_out = []
    time_in = []

    for i in range(0,len(size)):
        total.append(time[i])
        if(size[i] < 0):
            time_out.append(time[i])
        else:
            time_in.append(time[i])

    #総パケット到着時間の最大値、平均値、標準偏差、第3四分位数
    total_max, total_ave, total_std, total_per = max_ave_std_percentile75(total)
    features.append(total_max)
    features.append(total_ave)
    features.append(total_std)
    features.append(total_per)

    #受信パケット到着時間の最大値、平均値、標準偏差、第3四分位数
    in_max, in_ave, in_std, in_per = max_ave_std_percentile75(time_in)
    features.append(in_max)
    features.append(in_ave)
    features.append(in_std)
    features.append(in_per)

    #送信パケット到着時間の最大値、平均値、標準偏差、第3四分位数
    out_max, out_ave, out_std, out_per = max_ave_std_percentile75(time_out)
    features.append(out_max)
    features.append(out_ave)
    features.append(out_std)
    features.append(out_per)

    #総パケットタイムスタンプの第1、第2、第3四分位値、総送信時間
    total_per1, total_per2, total_per3, total_transtime = timestamp(total)
    features.append(total_per1)
    features.append(total_per2)
    features.append(total_per3)
    features.append(total_transtime)

    #受信パケットタイムスタンプの第1、第2、第3四分位値、総送信時間
    in_per1, in_per2, in_per3, in_transtime = timestamp(time_in)
    features.append(in_per1)
    features.append(in_per2)
    features.append(in_per3)
    features.append(in_transtime)

    #送信パケットタイムスタンプの第1、第2、第3四分位値、総送信時間
    out_per1, out_per2, out_per3, out_transtime = timestamp(time_out)
    features.append(out_per1)
    features.append(out_per2)
    features.append(out_per3)
    features.append(out_transtime)

    return features