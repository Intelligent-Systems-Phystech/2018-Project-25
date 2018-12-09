from imparse import Parser
import cv2 as cv
import re
import numpy as np
from descriptor import *

des = DESCRIPTOR()
cell_lbp_hist = des.cell_lbp_hist
hog_hist = des.hog_hist

def makehist_positive(stat): #stat = test or train
    par = Parser(stat)
    pdims = par.parse('INRIAPerson')
    ret = []
    for key in pdims.keys():
        cords = pdims[key]
        image = cv.imread('INRIAPerson/'+key, 1)
        for cns in cords:
            pimg = image[cns[0][1]:cns[1][1], cns[0][0]:cns[1][0]]
            pimg = cv.resize(pimg, (64,128), interpolation=cv.INTER_LINEAR)
            hist = hog_hist(pimg)
            ret.append(hist)
            #print len(hist)
            #cv.imshow('wnd', pimg)
            #cv.waitKey()
            #cv.destroyAllWindows()
    for h in ret:
        h.reshape(3780)
    return ret

def makehist_positive_norm(path, dsc = 'HOG+LBP'):
    poslist = open(path + '/' + 'pos.lst', 'r')
    ret = []
    for imname in poslist:
        imname = re.split(r'/', imname[:-1])[2]
        imname = path + '/pos/' + imname
        image = cv.imread(imname, 0)
        image = cv.resize(image, (64, 128), interpolation=cv.INTER_LINEAR)
        hist = []
        if dsc == 'HOG+LBP':
            hist = des.descript(image)
        if dsc == 'HOG':
            hist = des.hog_hist(image)
        if dsc == 'LBP':
            hist = des.cell_lbp_hist(image)
        ret.append(hist)

    return ret

def makehist_negative(stat, dsc = 'HOG+LBP'):
    neglstfname = 'INRIAPerson/'+ stat +'/neg.lst'
    neglist = open(neglstfname, 'r')
    ret = []
    for negname in neglist:
        negname = 'INRIAPerson/' + negname[:-1]
        image = cv.imread('INRIAPerson/train_64x128_H96/pos/crop_000010a.png', 0)
        nfeatures = 0
        hist = np.zeros(1)
        if dsc == 'HOG+LBP':
            hist = des.descript(image)
            nfeatures = des.n_lbp_features() + des.n_hog_features()
        if dsc == 'HOG':
            hist = des.hog_hist(image)
            nfeatures = des.n_hog_features()
        if dsc == 'LBP':
            hist = des.hog_hist(image)
            nfeatures = des.n_lbp_features()
        len = np.shape(hist)[0]
        left = 0
        right = nfeatures
        while (right<len):
            hst = hist[left:right]
            ret.append(hst)
            left += nfeatures
            right += nfeatures
            #print hst.shape
    return ret



