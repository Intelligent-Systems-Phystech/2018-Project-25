from imparse import Parser
import cv2 as cv
import numpy as np
# todo: add LBP-descriptor
def hog_hist(img):
   winSize = (64,128)
   blockSize = (16,16)
   blockStride = (8,8)
   cellSize = (8,8)
   nbins = 9
   derivAperture = 1
   winSigma = 4.
   histogramNormType = 0
   L2HysThreshold = 2.0000000000000001e-01
   gammaCorrection = 0
   nlevels = 64
   hog = cv.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
                           histogramNormType,L2HysThreshold,gammaCorrection,nlevels)
   winStride = (1,1)
   padding = (8,8)
   locations = ((10,20),)
   hist = hog.compute(img, padding)
   return hist

def makehist_positive(stat):
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
    return ret

hst = makehist_positive(['Test'] )
print len(hst)
