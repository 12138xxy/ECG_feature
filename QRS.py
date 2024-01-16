import numpy as np
from scipy.signal import cwt


class QRSDetector:
    def __init__(self, sig, t, rate, z):
        self.sig = sig
        self.t = t
        self.rate = rate
        self.z = z
        self.rvalue = None
        self.qvalue = None
        self.svalue = None
        self.wtsig2 = cwt(self.sig, 8, 'mexh')  # 计算 wtsig2

    def detect_r_peaks(self):
        thrtemp = np.sort(self.sig)
        thrlen = len(self.sig)
        thr = np.sum(thrtemp[-8:]) / 8
        thrmax = thr / 8

        zerotemp = np.sort(self.z)
        zerovalue = np.sum(zerotemp[:100]) / 100
        thr = (thrmax - zerovalue) * 0.3

        self.rvalue = np.array([i for i, val in enumerate(self.sig) if val > thr])

        i = 1
        while i < len(self.rvalue):
            if (self.rvalue[i] - self.rvalue[i - 1]) * self.rate < 0.4:
                if abs(self.sig[self.rvalue[i]]) > abs(self.sig[self.rvalue[i - 1]]):
                    self.rvalue = np.delete(self.rvalue, i - 1)
                else:
                    self.rvalue = np.delete(self.rvalue, i)
                i -= 1
            i += 1

        for i in range(len(self.rvalue)):
            if self.sig[self.rvalue[i]] > 0:
                k = np.arange(self.rvalue[i] - 5, self.rvalue[i] + 6)
                b = np.argmax(self.sig[k])
                self.rvalue[i] = self.rvalue[i] - 6 + b
            else:
                k = np.arange(self.rvalue[i] - 5, self.rvalue[i] + 6)
                b = np.argmin(self.sig[k])
                self.rvalue[i] = self.rvalue[i] - 6 + b

    def detect_q_peaks(self):
        self.qvalue = []
        for i in range(len(self.rvalue)):
            for j in range(self.rvalue[i], self.rvalue[i] - 30, -1):
                if self.sig[self.rvalue[i]] > 0:
                    if self.wtsig2[j] < self.wtsig2[j - 1] and self.wtsig2[j] < self.wtsig2[j + 1]:
                        tempqvalue = j - 10
                        break
                else:
                    if self.wtsig2[j] > self.wtsig2[j - 1] and self.wtsig2[j] > self.wtsig2[j + 1]:
                        tempqvalue = j - 10
                        break
            self.qvalue.append(tempqvalue)

    def detect_s_peaks(self):
        self.svalue = []
        for i in range(len(self.rvalue) - 1):
            for j in range(self.rvalue[i], self.rvalue[i] + 100):
                if self.sig[self.rvalue[i]] > 0:
                    if self.wtsig2[j] < self.wtsig2[j - 1] and self.wtsig2[j] < self.wtsig2[j + 1]:
                        tempsvalue = j + 10
                        break
                else:
                    if self.wtsig2[j] > self.wtsig2[j - 1] and self.wtsig2[j] > self.wtsig2[j + 1]:
                        tempsvalue = j + 10
                        break
            self.svalue.append(tempsvalue)

    def get_r_peaks(self):
        return self.rvalue

    def get_q_peaks(self):
        return self.qvalue

    def get_s_peaks(self):
        return self.svalue
