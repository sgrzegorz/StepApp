import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

#https://www.mathworks.com/help/matlabmobile_android/ug/counting-steps-by-capturing-acceleration-data.html
df = pd.read_csv("data/20steps/Accelerometer6.csv",names=["time","seconds_elapsed","z","y","x"],skiprows=1)
# t = list(df.time)
t = np.arange(0,df.shape[0])
x = list(df.x)
y = list(df.y)
z = list(df.z)

mag = np.sqrt(np.power(z,2)+np.power(x,2)+np.power(y,2))

avgmag = np.average(np.array((x,y,z),dtype=float),axis=0)
mag = mag -avgmag

def find_highest_peaks(ecg,peaks_all,treshold=0.25):
    values = np.take(ecg, indices=peaks_all)
    return(peaks_all[values > treshold])

# peaks_all, _ = find_peaks(mag)
#
# plt.scatter(peaks_all,np.take(mag, indices=peaks_all),color='red')
# plt.plot(t,mag)
# plt.show()

def window_average(mag,WINDOW = 50):
    mag1=[]
    for i in range(len(mag)-WINDOW+1):
        sum =0
        for j in range(WINDOW):
            sum += mag[i+j]
        mag1.append(sum / WINDOW)

    return np.array(mag1)

mag1 = window_average(mag,WINDOW = 50)
mag1 = window_average(mag1,WINDOW = 20)


peaks_all1, _ = find_peaks(mag1)


highestPeaks = list(find_highest_peaks(mag1,peaks_all1,treshold=0.13))


print(df)

t = np.arange(0,len(mag1))

print(np.std(mag))
plt.scatter(highestPeaks,np.take(mag1, indices=highestPeaks),color='red')
plt.plot(t,mag1)
plt.show()

print('Number of counted steps: ',)