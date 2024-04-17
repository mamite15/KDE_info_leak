import numpy as np
import random
import statistics
from sympy import *

from scipy import stats
import KDEpy
import distributions as distr

#ガウスカーネル関数の定義
def gaussian_kernel(x):
    return (1 / np.sqrt( 2 * np.pi )) * np.exp(-(x**2)/2)

def ksizeHall(X):
        """
        Find optimal kernel bandwidth using the "plug-in" method described by Hall et. al.

        The method will fail to find valid bandwidths when the variance between samples is zero.
        The caller needs to handle these scenarios.
        Method details can be found in DOI: 10.2307/2337251

        Parameters
        ----------
        X : ndarray
            Numpy array containing data samples, of shape (n_samples, n_features)

        Returns
        -------
        ndarray
            Numpy array containing optimal bandwidth for each dimension, of shape (n_features,)
        """
        X = np.transpose(X)
    
        N1, N2 = X.shape
        sig = np.std(X, axis=1)
        lamS = .7413 * np.transpose(stats.iqr(np.transpose(X)))
        if np.amax(lamS) == 0:
            lamS = sig

        BW = 1.0592 * lamS * np.power(float(N2), -1 / (4 + N1))
        BW = np.tile(BW, (1, N2))
    
        t = np.transpose(X[:, :, None], (0, 2, 1))
        dX = np.tile(t, (1, N2, 1))
    
        for i in range(N2):
            dX[:, :, i] = np.divide(dX[:, :, i] - X, BW)
        for i in range(N2):
            dX[:, i, i] = 2e22
        dX = np.reshape(dX, (N1, N2*N2))
    
        def h_findI2(n, dXa, alpha):
            t = np.exp(-0.5*np.sum(np.power(dXa, 2), axis=0))
            t = (np.power(dXa, 2) - 1) * (1/np.sqrt(2*np.pi)) * np.tile(t, (dXa.shape[0], 1))
            s = np.sum(t, axis=1)
            return np.divide(s, n*(n-1)*np.power(alpha, 5))
    
        def h_findI3(n, dXb, beta):
            t = np.exp(-0.5*np.sum(np.power(dXb, 2), axis=0))
            t = (np.power(dXb, 3) - (3*dXb)) * (1/np.sqrt(2*np.pi)) * np.tile(t, (dXb.shape[0], 1))
            s = np.sum(t, axis=1)
            return -np.divide(s, n*(n-1) * np.power(beta, 7))
    
        I2 = h_findI2(N2, dX, BW[:, 1])
        I3 = h_findI3(N2, dX, BW[:, 1])
    
        RK, mu2, mu4 = 0.282095, 1.000000, 3.000000
    
        J1 = (RK / mu2**2) * (1./I2)
        J2 = (mu4 * I3) / (20 * mu2) * (1./I2)
        h = np.power((J1/N2).astype(dtype=np.complex), 1.0/5) + \
            (J2 * np.power((J1/N2).astype(dtype=np.complex), 3.0/5))
        h = h.real.astype(dtype=np.float64)
    
        return np.transpose(h)




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

    if len(x.shape) == 2:
        ndim = x.shape[1]
        sd = np.sqrt(np.apply_along_axis(wvar, 1, x, weights))
        return (4.0 / ((ndim + 2.0) * n) ** (1.0 / (ndim + 4.0))) * sd
    
def plug_in(x, weights=None):


    h0 = hnorm(x)
    print("h0:" + str(h0))

    v0 = sj(x, h0)
    print("v0:" + str(v0))

    if v0 > 0:
        hstep = 1.1
    else:
        hstep = 0.9

    h1 = h0 * hstep
    v1 = sj(x, h1)
    print("v1:" + str(v1))

    while v1 * v0 > 0:
        h0 = h1
        v0 = v1
        h1 = h0 * hstep
        v1 = sj(x, h1)
        print("v1その2:" + str(v1))

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
def select_bw(features,sample_list):
    np_features = np.array(features)
    np_features_t = np_features.reshape(-1,1)
    #np.transpose(np_features_t)
    #print(np_features_t)
    x,y = np_features_t.shape
    #print(x)
    #print(y)

    #閾値
    #threshold = 2
    threshold = 300

    #同じ値カウント
    #equal_cnt=[]

    bw_list = []
    #"""
    try:
        #with np.errstate(all = 'raise'):
            #bw = p_in
            #p_in = plug_in(features)
        p_in = KDEpy.bw_selection.improved_sheather_jones(np_features_t)
        print("p_in:" + str(p_in))
    except:
        p_in = np.array([np.nan])
        print("cant solve plugin")
    #p_in = plug_in(features)
    #"""
    
    #r_o_t = rule_of_thumb(features)
    r_o_t = KDEpy.bw_selection.silvermans_rule(np_features_t)
    print("r_o_t:" + str(r_o_t))

    #離散的な特徴量なら0.001、連続的ならプラグイン推定 or rule-of-thumb用いる
    for feature in features:
        cnt=0
        #全ての特徴に対して同じ値があるか確認
        #for j in range(len(feature)):
        #    if feature[i] == feature[j]:
        #        cnt += 1

        cnt = features.count(feature)
        #print("cnt" + str(cnt))

        #equal_cnt.append(cnt-1)

        #閾値より値が大きければ離散的
        if cnt >= threshold:
            bw_list.append(0.001)
        else:#閾値より値が小さければプラグイン推定、ダメならrule-of-thumb適用
            bw = p_in
            #bw = r_o_t
            if np.isnan(bw) or np.isinf(bw):
                bw = r_o_t
            bw_list.append(bw)

    #print(bw_list)
    for i in range(0,len(bw_list)):
        if bw_list[i] == 0:
            bw_list[i] = 0.001

    #print("bw: " + str(bw_list))
    #print("bw_list:" + str(bw_list))
    return bw_list


#確率密度推定
def akde_cal(features,sample_list):
    #インスタンス数
    size = len(features)
    
    #サンプルの確率密度のリスト
    #pdf(f|w)のリスト
    pdf_list = []

    #バンド幅選択
    band_width = select_bw(features,sample_list)
    #print("hello")

    #サンプルごとに確率密度計算

    for sample in sample_list:
    #    #確率密度計算用変数 シグマの中のやつ
        xx=[]
        #確率密度計算　pdf=1/n * Σ 1/h * K((f-p)/h)
        #pdf(f|c)計算
        for j in range(0,size):

            #カーネル関数
            kernel = gaussian_kernel((sample - features[j])/band_width[j])

           #xx.append((1/(web_num * band_width[j])) * kernel)
            xx.append((kernel/band_width[j]))
        
        pdf_list.append(sum(xx)/size)

        #if sum(xx)/size == 0:
        #    print("sample" + str(sample))
        #    print("features:" + str(features))
            #print("xx" + str(xx))


    return pdf_list