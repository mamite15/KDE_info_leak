from KDEpy import FFTKDE
from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np
from KDEpy import FFTKDE
from scipy.stats import norm

# Generate a distribution and some multimodal data
#dist1 = norm(loc=0, scale=1)
#dist2 = norm(loc=10, scale=1)
#data = np.hstack([dist1.rvs(2**8), dist2.rvs(2**8)])

one = [[1] * 20,[1] * 20]
two = [[2] * 30,[2] * 30]
three = [[3] * 50,[3] * 50]
features = one
for i in range(2):
    features[i].extend(two[i])
    features[i].extend(three[i])
data = features
# Compute density estimates using 'silverman'
#x, y = FFTKDE(kernel='gaussian', bw='silverman').fit(data).evaluate()
#plt.plot(x, y, label='KDE /w silverman')
#print(x,y)
# Compute density estimates using 'ISJ' - Improved Sheather Jones
x,y = FFTKDE(kernel='gaussian', bw=0.001).fit(data).evaluate()

plt.plot(x, y, label='KDE /w ISJ')

#plt.plot(x, data.pdf(x), label='True pdf')
plt.grid(True, ls='--', zorder=-15); plt.legend();

plt.show()