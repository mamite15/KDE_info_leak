import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.distributions import norm

np.random.seed(0)
x = np.concatenate([norm(-1, 1.).rvs(200), norm(1, 0.3).rvs(100)])

# ヒストグラムで確認する
fig, ax = plt.subplots(figsize=(17, 4))
ax.hist(x, 10, fc='gray', histtype='stepfilled', alpha=0.5, normed=True)
ax.set_xlim(x.min()-1, x.max()+1)
plt.show()