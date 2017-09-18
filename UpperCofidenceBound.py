import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('Ads_CTR_Optimisation.csv')

import math
N = 10000
d = 10
ads_selected=[]
number_of_selections=[0] * d
sum_of_rewards=[0] * d
total_reward = 0
for i in range(0,N):
    ad = 0
    max_upper_bound = 0
    for j in range(0,d):
        if(number_of_selections[j]>0):
            avg_reward = sum_of_rewards[j]/number_of_selections[j]
            delta = math.sqrt( 3/2 * math.log(i+1)/number_of_selections[j])
            upper_bound = avg_reward + delta
        else:
            upper_bound = 1e400
        if upper_bound>max_upper_bound:
            max_upper_bound = upper_bound
            ad = j
    ads_selected.append(ad)
    number_of_selections[ad] = number_of_selections[ad] + 1
    reward = data.values[i,ad]
    sum_of_rewards[ad] = sum_of_rewards[ad] + reward
    total_reward = total_reward + reward

plt.hist(ads_selected)
plt.show()