#!usr/bin/env python
# encoding:utf-8
from __future__ import division

import math
import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau

import sys

# reload(sys)
# sys.setdefaultencoding('utf-8')


def pearsonrSim(x, y):
    '''
	pearsonrSim
	'''
    return pearsonr(x, y)[0]


def spearmanrSim(x, y):
    '''
	spearmanrSim
	'''
    return spearmanr(x, y)[0]


def kendalltauSim(x, y):
    '''
	kendalltauSim
	'''
    return kendalltau(x, y)[0]


def cosSim(x, y):
    '''
	cosSim
	'''
    tmp = sum(a * b for a, b in zip(x, y))
    non = np.linalg.norm(x) * np.linalg.norm(y)
    return round(tmp / float(non), 3)


def eculidDisSim(x, y):
    '''
	eculidDisSim
	'''
    return math.sqrt(sum(pow(a - b, 2) for a, b in zip(x, y)))


def manhattanDisSim(x, y):
    '''
	manhattanDisSim
	'''
    return sum(abs(a - b) for a, b in zip(x, y))


def minkowskiDisSim(x, y, p):
    '''
	minkowskiDisSim
	'''
    sumvalue = sum(pow(abs(a - b), p) for a, b in zip(x, y))
    tmp = 1 / float(p)
    return round(sumvalue**tmp, 3)


def MahalanobisDisSim(x, y):
    '''
	MahalanobisDisSim
	'''
    npvec1, npvec2 = np.array(x), np.array(y)
    npvec = np.array([npvec1, npvec2])
    sub = npvec.T[0] - npvec.T[1]
    inv_sub = np.linalg.inv(np.cov(npvec1, npvec2))
    return math.sqrt(np.dot(inv_sub, sub).dot(sub.T))


def levenshteinDisSim(x, y):
    '''
	levenshteinDisSim
	'''
    res = Levenshtein.distance(x, y)
    similarity = 1 - (res / max(len(x), len(y)))
    return similarity


def jaccardDisSim(x, y):
    '''
	jaccardDisSim
	'''
    res = len(set.intersection(*[set(x), set(y)]))
    union_cardinality = len(set.union(*[set(x), set(y)]))
    return res / float(union_cardinality)


def Matchsize(x, y):
    # print('pearsonrSim（皮尔森相似度）:', pearsonrSim(x, y))
    # print('spearmanrSim（斯皮尔曼相似度）:', spearmanrSim(x, y))
    # print('kendalltauSim（肯德尔相似度）:', kendalltauSim(x, y))
    print('cosSim（余弦相似度计算方法）:', cosSim(x, y))
    # print('eculidDisSim（欧几里得相似度计算方法）:', eculidDisSim(x, y))
    # print('manhattanDisSim（曼哈顿距离计算方法）:', manhattanDisSim(x, y))
    # print('minkowskiDisSim（明可夫斯基距离计算方法）:', minkowskiDisSim(x, y, 2))
    # print('MahalanobisDisSim（马氏距离计算方法）:', MahalanobisDisSim(x, y))
    print('jaccardDisSim（杰卡德相似度计算）:', jaccardDisSim(x, y))


if __name__ == '__main__':
    x = [2, 7, 18, 88, 157, 90, 177, 570]
    y = [6, 10, 30, 180, 360, 176, 320, 1160]
    print('pearsonrSim:', pearsonrSim(x, y))
    print('spearmanrSim:', spearmanrSim(x, y))
    print('kendalltauSim:', kendalltauSim(x, y))
    print('cosSim:', cosSim(x, y))
    print('eculidDisSim:', eculidDisSim(x, y))
    print('manhattanDisSim:', manhattanDisSim(x, y))
    print('minkowskiDisSim:', minkowskiDisSim(x, y, 2))
    print('MahalanobisDisSim:', MahalanobisDisSim(x, y))
    print('jaccardDisSim:', jaccardDisSim(x, y))
