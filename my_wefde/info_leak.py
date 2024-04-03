from fractions import Fraction
import math


#H(W)計算
def cal_hw(web_num):
    return math.log2(web_num)

#H(W|F)計算
#モンテカルロ法のサンプルごとのpdfを引数で受け取る
#ここでモンテカルロ法を計算
def cal_hwf(pdf_list):
    size = len(pdf_list)

    hwfi = []
    for pdf in pdf_list:
        hwfi.append(pdf*math.log2(pdf))

    return sum(hwfi)/size
