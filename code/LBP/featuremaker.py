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

def makehist_positive_norm(path):
    poslist = open(path + '/' + 'pos.lst', 'r')
    ret = []
    for imname in poslist:
        imname = re.split(r'/', imname[:-1])[2]
        imname = path + '/pos/' + imname
        image = cv.imread(imname, 0)
        image = cv.resize(image, (64, 128), interpolation=cv.INTER_LINEAR)
        hist = des.descript(image)
        ret.append(hist)

    return ret

def makehist_negative(stat):
    neglstfname = 'INRIAPerson/'+ stat +'/neg.lst'
    neglist = open(neglstfname, 'r')
    ret = []
    for negname in neglist:
        negname = 'INRIAPerson/' + negname[:-1]
        image = cv.imread('INRIAPerson/train_64x128_H96/pos/crop_000010a.png', 0)
        hist = des.descript(image)
        len = np.shape(hist)[0]
        nfeatures = des.n_lbp_features()+des.n_hog_features()
        left = 0
        right = nfeatures
        while (right<len):
            hst = hist[left:right]
            ret.append(hst)
            left += nfeatures
            right += nfeatures
    return ret



