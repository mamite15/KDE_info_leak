def ngram_locater(sample,n):
    index=0
    for i in range(0,n):
        if sample[i] == 1:
            bit = 1
        else:
            bit = 0    
        index = index + bit * (2 ** (n - i - 1))
    
    return index


def ngram_calculation(sizes,n):
    counter = 0

    buckets = [0] * (2 ** n)

    for i in range(0,len(sizes) - n+1):
        index = ngram_locater(sizes[i:i + n], n)
        buckets[index] = buckets[index] + 1
        counter = counter + 1

    return buckets

def ngram_extract(size):

    #追加する特徴
    features = []
    
    #2-gram
    buckets_2=ngram_calculation(size,2)
    features.extend(buckets_2)

    #3-gram
    buckets_3=ngram_calculation(size,3)
    features.extend(buckets_3)
    
    #4-gram
    buckets_4=ngram_calculation(size,4)
    features.extend(buckets_4)
    
    #5-gram
    buckets_5=ngram_calculation(size,5)
    features.extend(buckets_5)
    
    #6-gram
    buckets_6=ngram_calculation(size,6)
    features.extend(buckets_6)

    return features