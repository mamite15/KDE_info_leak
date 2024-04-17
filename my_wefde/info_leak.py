from fractions import Fraction
import math


#H(W)計算
def cal_hw(web_num):
    return math.log2(web_num)

#H(W|F)計算
#モンテカルロ法のサンプルごとのpdfを引数で受け取る
#ここでモンテカルロ法を計算
def cal_hwf(con_pdf_list,total_pdf_list):
    size = len(con_pdf_list)
    total_size = len(total_pdf_list)

    #print("web数:" + str(size))
    #print("サンプル数:" + str(total_size))
    #print("con_pdf_list:" + str(con_pdf_list))

    
    hwf_list = []
    for i in range(0,total_size):
        hwfi = []
        #print("サンプル" + str(i) + "個目")
        for j in range(0,size):
            #print("web" + str(j) + "個目のサンプルhwfi")
            q = con_pdf_list[j][i] / (size * total_pdf_list[i])
            #print("con_pdf_list[j][i]" + str(con_pdf_list[j][i]))
            #print("q:" + str(q))
            #print("hwfi:" + str(-q*math.log2(q)))
            #hwfi.append(total_pdf_list[i]*q*math.log2(q))
            if q != 0:
                hwfi.append(-q*math.log2(q))
            else:
                hwfi.append(0)

        #hwf_list.append(total_pdf_list[i] * sum(hwfi))
        hwf_list.append(sum(hwfi))


    #print("hwf_list:" + str(hwf_list))
    return sum(hwf_list)/total_size
