import numpy as np
import random
import statistics
from sympy import *
from sklearn import preprocessing

import KDEpy

import distributions as distr

#ガウスカーネル関数の定義
def gaussian_kernel(x):
    return 1/np.sqrt(2*np.pi)*np.exp(-(x**2)/2)

def wmean(x, w):
    '''
    Weighted mean
    '''
    return sum(x * w) / float(sum(w))


def wvar(x, w):
    '''
    Weighted variance
    '''
    return sum(w * (x - wmean(x, w)) ** 2) / float(sum(w) - 1)

def dnorm(x):
    return distr.normal.pdf(x, 0.0, 1.0)

#plug-in推定によるバンド幅選択
def sj(x, h):

    phi6 = lambda x: (x ** 6 - 15 * x ** 4 + 45 * x ** 2 - 15) * dnorm(x)
    phi4 = lambda x: (x ** 4 - 6 * x ** 2 + 3) * dnorm(x)

    n = len(x)
    #n=1
    one = np.ones((1, n))

    lam = np.percentile(x, 75) - np.percentile(x, 25)
    a = 0.92 * lam * n ** (-1 / 7.0)
    b = 0.912 * lam * n ** (-1 / 9.0)

    W = np.tile(x, (n, 1))
    W = W - W.T

    W1 = phi6(W / b)
    tdb = np.dot(np.dot(one, W1), one.T)
    tdb = -tdb / (n * (n - 1) * b ** 7)

    W1 = phi4(W / a)
    sda = np.dot(np.dot(one, W1), one.T)
    sda = sda / (n * (n - 1) * a ** 5)

    alpha2 = 1.357 * (abs(sda / tdb)) ** (1 / 7.0) * h ** (5 / 7.0)

    W1 = phi4(W / alpha2)
    sdalpha2 = np.dot(np.dot(one, W1), one.T)
    sdalpha2 = sdalpha2 / (n * (n - 1) * alpha2 ** 5)

    return (distr.normal.pdf(0, 0, np.sqrt(2)) /
            (n * abs(sdalpha2[0, 0]))) ** 0.2 - h

def hnorm(x, weights=None):

    x = np.asarray(x)

    if weights is None:
        weights = np.ones(len(x))

    n = float(sum(weights))

    if len(x.shape) == 1:
        sd = np.sqrt(wvar(x, weights))
        return sd * (4 / (3 * n)) ** (1 / 5.0)

    # TODO: make this work for more dimensions
    # ((4 / (p + 2) * n)^(1 / (p+4)) * sigma_i
    if len(x.shape) == 2:
        ndim = x.shape[1]
        sd = np.sqrt(np.apply_along_axis(wvar, 1, x, weights))
        return (4.0 / ((ndim + 2.0) * n) ** (1.0 / (ndim + 4.0))) * sd
    
def plug_in(x, weights=None):


    h0 = hnorm(x)
    v0 = sj(x, h0)

    if v0 > 0:
        hstep = 1.1
    else:
        hstep = 0.9

    h1 = h0 * hstep
    v1 = sj(x, h1)

    while v1 * v0 > 0:
        h0 = h1
        v0 = v1
        h1 = h0 * hstep
        v1 = sj(x, h1)

    return h0 + (h1 - h0) * abs(v0) / (abs(v0) + abs(v1))
#rule-of-thumbによるバンド幅選択
def rule_of_thumb(features):

    size = len(features)
    #標準偏差
    a=statistics.pstdev(features)

    q75, q25 = np.percentile(features,[75,25])
    #四分位範囲
    iqr = q75 - q25

    bw = (0.9 * min(a,iqr/1.34)) * (size ** (-1/5))
    
    return bw

#バンド幅選択
def select_bw(features):

    print("select bandwidth ...")
    #閾値
    threshold = 100

    #同じ値カウント
    #equal_cnt=[]

    bw_list = []

    p_in = plug_in(features)
    #p_in = KDEpy.bw_selection.improved_sheather_jones(features)
    r_o_t = rule_of_thumb(features)

    #離散的な特徴量なら0.001、連続的ならプラグイン推定 or rule-of-thumb用いる
    for feature in features:
        cnt=0
        #全ての特徴に対して同じ値があるか確認
        #for j in range(len(feature)):
        #    if feature[i] == feature[j]:
        #        cnt += 1

        cnt = features.count(feature)

        #equal_cnt.append(cnt-1)

        #閾値より値が大きければ離散的
        if cnt >= threshold:
            bw_list.append(0.001)
        else:#閾値より値が小さければプラグイン推定、ダメならrule-of-thumb適用
            bw = p_in
            
            #bw = rule_of_thumb(features)
            if np.isnan(bw) or np.isinf(bw):
                bw = r_o_t
            bw_list.append(bw)

    return bw_list


def discrete(features):
        #閾値
    threshold = 100
    discrete_or_not = []

        #離散的な特徴量なら0.001、連続的ならプラグイン推定 or rule-of-thumb用いる
    for feature in features:
        cnt=0
        #全ての特徴に対して同じ値があるか確認
        cnt += 1

        cnt = features.count(feature)

        #equal_cnt.append(cnt-1)

        #閾値より値が大きければ離散的
        if cnt >= threshold:
            discrete_or_not.append(0)
        else:#閾値より値が小さければプラグイン推定、ダメならrule-of-thumb適用
            discrete_or_not.append(1)

    return discrete_or_not
#確率密度推定
def akde_cal(feature,web_num):
    
    print("start calculate akde ...")
    #インスタンス数
    size = len(feature)
    #モンテカルロ用サンプル取得 (k/webサイト数)個 今回k=5000
    #sample_list = random.sample(feature,int(5000/web_num))
    #sample_list = random.sample(feature,int(50/web_num))
    #sample_list = [1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3]

    #サンプルの確率密度のリスト
    pdf_list = []

    np_feature = np.array(feature)
    
    #バンド幅選択
    band_width = select_bw(feature)
    #discrete_or_not = discrete(feature)

    
    #x,y = KDEpy.FFTKDE(kernel='gaussian', bw='ISJ').fit(feature).evaluate()
    x,y = KDEpy.TreeKDE(kernel='gaussian', bw=band_width).fit(feature).evaluate()
    #x,y = FFTKDE(kernel='gaussian', bw=0.001).fit(feature).evaluate()
    #z = FFTKDE(kernel='gaussian', bw= 0.001).fit(feature).evaluate(x)
    lx=x.tolist()
    sample_list = random.sample(lx,int(50/web_num))

    #print("size:" + str(size) + ",bw:" + str(band_width))
    #print("sample:" + str(sample_list))
    
    #サンプルごとに確率密度計算
    for sample in sample_list:
        #確率密度計算用変数 シグマの中のやつ
        #xx=[]
        
        xi=lx.index(sample)

        pdf_list.append(y[xi])

        #確率密度計算　pdf=1/n * Σ 1/h * K((f-p)/h)
        #for j in range(0,size):

            #カーネル関数
         #   kernel = gaussian_kernel((sample - feature[j])/band_width[j])

        #    xx.append((1/(web_num * band_width[j])) * kernel)
        
        #pdf_list.append(sum(x))

    
    #print("pdf_list:" + str(pdf_list))
  
    #ms_pdf_list = preprocessing.minmax_scale(pdf_list)
    
    #ms_pdf_list = []
    #for pdf in pdf_list:
    #    norm = pdf / sum(pdf_list)
    #    ms_pdf_list.append(norm)
    #print("ms_pdf_list:" + str(ms_pdf_list))
    

    #return ms_pdf_list
    return pdf_list