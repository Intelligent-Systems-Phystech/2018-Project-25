def sum(a,b):
    return (a[0]+b[0], a[1]+b[1])

def left_start(corners, win_size):
    ret = ((0, corners[0][1]), (win_size[0], corners[1][1]), (win_size[0], corners[2][1]), (0, corners[3][1]))
    return ret


def stride(corners, stride, way = 'right'):
    up_left = corners[0]
    up_right = corners[1]
    b_right = corners[2]
    b_left = corners[3]
    if way == 'right':
        up_left = sum(up_left, (stride[0], 0))
        up_right = sum(up_right, (stride[0], 0))
        b_left = sum(b_left, (stride[0], 0))
        b_right = sum(b_right, (stride[0], 0))
    if way == 'down':
        up_left = sum(up_left, (0, stride[1]))
        up_right = sum(up_right, (0, stride[1]))
        b_left = sum(b_left, (0, stride[1]))
        b_right = sum(b_right, (0, stride[1]))
    return (up_left, up_right, b_right, b_left)


class DESCRIPTOR:

    def __init__(self, winSize = (64,128), blockSize = (16,16),
                 blockStride = (8,8), cellSize = (8,8),
                 winStride = (16,16), radius = 2, npoints = 8):
        self.winSize = winSize
        self.blockSize = blockSize
        self.blockStride = blockStride
        self.cellSize = cellSize
        self.winStride = winStride
        self.radius = radius
        self.npoints = npoints

    def hog_hist(self, img):
        import cv2 as cv
        winSize = self.winSize
        blockSize = self.blockSize
        blockStride = self.blockStride
        cellSize = self.cellSize
        nbins = 9
        derivAperture = 1
        winSigma = 4.
        histogramNormType = 0
        L2HysThreshold = 2.0000000000000001e-01
        gammaCorrection = 0
        nlevels = 64
        hog = cv.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma,
                               histogramNormType, L2HysThreshold, gammaCorrection, nlevels)
        winStride = self.winStride
        padding = (0, 0)
        hist = hog.compute(img, winStride, padding)
        return hist

    def n_lbp_features(self):
        ret = (1+(self.winSize[0]-self.blockSize[0])/self.blockStride[0])*(1+(self.winSize[1]-self.blockSize[1])/self.blockStride[1])
        ret *= (self.blockSize[0]/self.cellSize[0])*(self.blockSize[1]/self.cellSize[1])
        ret *= self.npoints+2
        return ret

    def n_hog_features(self):
        ret = (1+(self.winSize[0]-self.blockSize[0])/self.blockStride[0])*(1+(self.winSize[1]-self.blockSize[1])/self.blockStride[1])
        ret *= (self.blockSize[0]/self.cellSize[0])*(self.blockSize[1]/self.cellSize[1])
        ret *= 9
        return ret

    def cell_lbp_hist(self, img, trans = True):
        if trans == True:
            img = img.transpose()
        winSize = self.winSize
        blockSize = self.blockSize
        blockStride = self.blockStride
        cellSize = self.cellSize
        winStride = self.winStride
        radius = self.radius
        npoints = self.npoints
        from skimage.feature import local_binary_pattern as lbp
        import numpy as np
        METHOD = 'uniform'
        flag_1 = (winSize[0] % blockSize[0] == 0 and
               winSize[1] % blockSize[1] == 0 and
               blockSize[0] % cellSize[0] == 0 and
               blockSize[0] % cellSize[0] == 0)
        flag_2 = (img.shape[0] >= 64 and img.shape[1] >=128)

        if flag_2 == False:
            print('Size error')
            print img.shape
            return
        lbpim = lbp(img, npoints, radius, METHOD)
        up_left = (0, 0)
        b_left = (0, winSize[1])
        up_right = (winSize[0], 0)
        b_right = winSize

        win_corners = (up_left, up_right, b_right, b_left)
        hist = np.array([])
        while win_corners[2][1]<=lbpim.shape[1]:
            while win_corners[2][0]<=lbpim.shape[0]:

                win = lbpim[win_corners[0][0]:win_corners[1][0], win_corners[0][1]:win_corners[3][1]]
                bl_corners = ((0, 0), (blockSize[0], 0), blockSize, (0, blockSize[1]))

                while bl_corners[2][1] <=win.shape[1]:
                    while bl_corners[2][0] <= win.shape[0]:
                        block = win[bl_corners[0][0]:bl_corners[1][0], bl_corners[0][1]:bl_corners[3][1]]
                        cell_corners = ((0,0), (cellSize[0],0), cellSize, (0, cellSize[1]))
                        while cell_corners[2][1] <= block.shape[1]:
                            while cell_corners[2][0] <= block.shape[0]:
                                cell = block[cell_corners[0][0]:cell_corners[1][0], cell_corners[0][1]:cell_corners[3][1]]
                                local_hist = np.zeros(npoints+2)
                                for line in cell:
                                    for i in line:
                                        index = int(i)
                                        local_hist[index] +=1
                                hist = np.concatenate((hist, local_hist))
                                cell_corners = stride(cell_corners, cellSize, 'right')
                            cell_corners = stride(cell_corners, cellSize, 'down')
                            cell_corners = left_start(cell_corners, cellSize)
                        bl_corners = stride(bl_corners, blockStride, 'right')
                    bl_corners = stride(bl_corners, blockStride, 'down')
                    bl_corners = left_start(bl_corners, blockSize)



                win_corners = stride(win_corners, winStride, 'right')
            win_corners = stride(win_corners, winStride, 'down')
            win_corners = left_start(win_corners, winSize)
        return hist.reshape((hist.shape[0], 1))

    def descript(self, img):
        import numpy as np
        hog_hist = self.hog_hist(img)
        lbp_hist = self.cell_lbp_hist(img)
        return np.concatenate((hog_hist, lbp_hist))
