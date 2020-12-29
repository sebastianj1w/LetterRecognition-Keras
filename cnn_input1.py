import numpy as np
import csv
import Utils
import os
import re
import time


class Data_Control:
    def __init__(self, filepath):
        self.files = os.listdir(filepath)
        self.alldata, self.Xlen, self.alllabel, self.flen, self.allindex, self.alluser = self.loadfile(filepath)
        self.padding_data, _, self.length = self.cnn_padding(self.alldata, self.Xlen, self.flen)
        self.alldata = np.array(self.padding_data)
        trainindex, testindex = self.indexsplit(self.alldata.shape[0], True)
        print(len(testindex))
        self.traindata = self.alldata[trainindex]
        self.trainlabel = self.alllabel[trainindex]
        self.trainuser = self.alluser[trainindex]
        self.testdata = self.alldata[testindex]
        self.testlabel = self.alllabel[testindex]
        self.testuser = self.alluser[testindex]

    def indexsplit(self, indexlength, israndom):
        if israndom is True:
            randomind = list(range(indexlength))
            np.random.shuffle(randomind)
            trainindex = randomind[:int(len(randomind) * 0.8)]
            testindex = list(filter(lambda j: j not in trainindex, list(randomind)))
        else:
            trainindex = []
            testindex = []
            for i in range(indexlength):
                if self.alluser[i] < 100:

                    if self.alluser[i] == 0 and self.rot_ind[i] == -1:
                        testindex.append(i)
                    else:
                        if self.alluser[i] != 0:
                            trainindex.append(i)
            print(len(trainindex))
            np.random.shuffle(trainindex)
        return trainindex, testindex

    def loadfile(self, filepath):
        raw_data = []
        raw_data_len = []
        raw_label = []
        raw_index = []
        raw_user = []
        starttime = time.time()
        lasttime = time.time()
        kk = 0
        for file in self.files:
            pattern = re.compile(r'\d+')
            res = re.findall(pattern, file)
            if len(res) == 3 and int(res[1]) < 1000:
                filename = filepath + file
                data = np.load(filename)
                sample = data['datapre']
                featurelen = sample.shape[1]
                raw_data.append(sample)
                raw_data_len.append(sample.shape[0])
                raw_label.append(int(res[0]))
                raw_index.append(int(res[1]))
                raw_user.append(int(res[2]))
                kk = kk + 1
                if kk % 1000 == 0:
                    nowtime = time.time()
                    print("%d, %0fs" % (kk, nowtime - starttime))

        return np.array(raw_data), raw_data_len, np.array(raw_label), featurelen, np.array(raw_index), np.array(
            raw_user)

    def cnn_padding(self, data, slen, flen):
        raw_data = data
        lengths = slen
        median_length = 252
        # median_length = int(np.median(lengths))
        num_samples = len(lengths)
        padding_data = np.zeros([num_samples, median_length, flen])
        for idx, sample in enumerate(raw_data):
            temp = [Utils.resampling(np.arange(0, len(x), 1), x, [0, len(x)], median_length)[1] for x in
                    np.array(sample).transpose()]
            padding_data[idx, :, :] = np.array(temp).transpose()
        return padding_data, np.array(slen), median_length

    def cnn_padding1(self, data, slen, flen):
        raw_data = data
        lengths = np.array(slen)
        # median_length = int(np.median(lengths))
        lengthsort = np.array(sorted(lengths))
        median_length = int(np.percentile(np.array(lengths), 99))
        num_samples = len(slen)
        padding_data = np.zeros([num_samples, median_length, flen])
        for idx, sample in enumerate(raw_data):
            temp = np.zeros([flen, median_length])

            if slen[idx] < median_length:
                sample = np.transpose(sample)
                len_diff = median_length - slen[idx]
                len_diff1 = len_diff // 2
                len_diff2 = len_diff - len_diff1
                for xidx, x in enumerate(sample):
                    temp[xidx, :] = [0] * len_diff1 + x.tolist() + [0] * len_diff2
                # temp = [Utils.resampling(np.arange(0, len(x), 1), x, [0, len(x)], median_length)[1] for x in
                #         np.array(sample).transpose()]
            if slen[idx] > median_length:
                sample = np.transpose(sample)
                len_diff = slen[idx] - median_length
                len_diff1 = len_diff // 2
                len_diff2 = len_diff - len_diff1
                temp = sample[:, len_diff1:slen[idx] - len_diff2]
                # temp = [Utils.resampling(np.arange(0, len(x), 1), x, [0, len(x)], median_length)[1] for x in
                #         np.array(sample).transpose()]

                # print()
            if slen[idx] == median_length:
                temp = np.transpose(sample)
            # plt.figure()
            # for i in range(6, 9):
            #     plt.plot(temp[i, :])
            # plt.show()
            padding_data[idx, :, :] = np.array(temp).transpose()
        padding_data = np.array(padding_data)
        return padding_data, np.array(slen), median_length
