import itertools
import numpy as np

def cumul_features(size):
    
    separateClassifier = True
    # Calculate Features

    features = []

    total = []
    cum = []
    pos = []
    neg = []
    inSize = 0
    outSize = 0
    inCount = 0
    outCount = 0

    # Process trace
    for packetsize in itertools.islice(size, None):

        # CUMUL uses positive to denote incoming, negative to be outgoing,
        # different from dataset
        packetsize = - packetsize

        # incoming packets
        if packetsize > 0:
            inSize += packetsize
            inCount += 1
            # cumulated packetsizes
            if len(cum) == 0:
                cum.append(packetsize)
                total.append(packetsize)
                pos.append(packetsize)
                neg.append(0)
            else:
                cum.append(cum[-1] + packetsize)
                total.append(total[-1] + abs(packetsize))
                pos.append(pos[-1] + packetsize)
                neg.append(neg[-1] + 0)

        # outgoing packets
        if packetsize < 0:
            outSize += abs(packetsize)
            outCount += 1
            if len(cum) == 0:
                cum.append(packetsize)
                total.append(abs(packetsize))
                pos.append(0)
                neg.append(abs(packetsize))
            else:
                cum.append(cum[-1] + packetsize)
                total.append(total[-1] + abs(packetsize))
                pos.append(pos[-1] + 0)
                neg.append(neg[-1] + abs(packetsize))

    # Should already be removed by outlier Removal
    # if len(cum) < 2:
    # something must be wrong with this capture
    # continue

    # add feature
    # features.append(classLabel)
    features.append(inCount)
    features.append(outCount)
    features.append(outSize)
    features.append(inSize)

    if separateClassifier:
        # cumulative in and out
        posFeatures = np.interp(np.linspace(int(total[0]), int(total[-1]), 50), total, pos)
        negFeatures = np.interp(np.linspace(int(total[0]), int(total[-1]), 50), total, neg)
        for el in itertools.islice(posFeatures, None):
            features.append(el)
        for el in itertools.islice(negFeatures, None):
            features.append(el)
    else:
        # cumulative in one
        cumFeatures = np.interp(np.linspace(total[0], total[-1], 100 + 1), total, cum)
        for el in itertools.islice(cumFeatures, 1, None):
            features.append(el)

    return features
