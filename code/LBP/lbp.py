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

def cell_lbp_hist(img, winSize = (64,128), blockSize = (16,16),
             blockStride = (8,8), cellSize = (8,8),
             winStride = (16,16), radius = 2, npoints = 8):
    from skimage.feature import local_binary_pattern as lbp
    METHOD = 'uniform'
    flag_1 = (winSize[0] % blockSize[0] == 0 and
           winSize[1] % blockSize[1] == 0 and
           blockSize[0] % cellSize[0] == 0 and
           blockSize[0] % cellSize[0] == 0)
    flag_2 = (img.shape[0] >= 64 and img.shape[1] >=128)

    if flag_1 and flag_2 == False:
        print('Size error')
        return
    lbpim = lbp(img, npoints, radius, METHOD)
    up_left = (0, 0)
    b_left = (0, winSize[1])
    up_right = (winSize[0], 0)
    b_right = winSize

    win_corners = (up_left, up_right, b_right, b_left)
    hist = []
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
                            local_hist = [0,0,0,0,0,0,0,0,0]
                            hist = hist + local_hist
                            cell_corners = stride(cell_corners, cellSize, 'right')
                        cell_corners = stride(cell_corners, cellSize, 'down')
                        cell_corners = left_start(cell_corners, cellSize)
                    bl_corners = stride(bl_corners, blockStride, 'right')
                bl_corners = stride(bl_corners, blockStride, 'down')
                bl_corners = left_start(bl_corners, blockSize)



            win_corners = stride(win_corners, winStride, 'right')
        win_corners = stride(win_corners, winStride, 'down')
        win_corners = left_start(win_corners, winSize)
    return hist


"""


    while b_right[0]<=lbpim.shape[0] or b_right[1]<=lbpim.shape[1]:

        win = lbpim[up_left[0]:up_right[0],up_left[1]:b_left[1]]
        bl_up_left = (0, 0)
        bl_b_left = (0, blockSize[1])
        bl_up_right = (blockSize[0], 0)
        bl_b_right = blockSize
        while bl_b_right[0] <= win.shape[0] or bl_b_right[1] <= win.shape[1]:
            block = win[bl_up_left[0]:bl_up_right[0], bl_up_left[1]:bl_b_left[1]]

            return """