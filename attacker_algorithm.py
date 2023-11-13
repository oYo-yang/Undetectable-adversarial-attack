import numpy as np

def attacker(func, numPix, bounds, success_fun, img, maxiter = 400):

    cilia_size0 = 0.5
    cilia_size = cilia_size0
    c = 1.0   # step_size/cilia_size
    decay_rho = 1.2
    decay_eta = 0.7

    nDim = len(bounds)

    bounds_ = np.array(bounds)
    lb = bounds_[:,0]
    ub = bounds_[:,1]
    bounds_diff = ub - lb - 1

    def bound(x_in, imgs):
        x_in = np.clip(x_in, 0, 1)

        xs = (x_in * bounds_diff + lb).astype(np.int)

        tile = [len(xs)] + [1] * (xs.ndim + 1)
        img = np.tile(imgs, tile)

        x_in2 = []
        for x, imgs in zip(xs, img):
            pixels = np.split(x, len(x) // 5)
            for pixel in pixels:

                x_pos, y_pos, *rgb = pixel
                rgb2 = np.array(rgb)
                rgb1 =imgs[x_pos, y_pos]
                for i in range(3):
                    if abs(rgb1[i] - rgb2[i]) > 10:
                        if rgb1[i] >= rgb2[i]:
                            rgb2[i] = rgb1[i] - 10
                        else:
                            rgb2[i] = rgb1[i] + 10

                x_in = x_pos, y_pos, *rgb2
                x_in1 = np.array(x_in)

                x_in2 = np.append(x_in2, x_in1)

        x_in2 = x_in2.reshape((1, 5120))
        x_in = (x_in2 - lb)/bounds_diff.astype(np.float)


        return x_in

    def scale(x_in):
        return (x_in*bounds_diff + lb).astype(np.int)

    x_best = np.zeros((numPix, nDim))
    f_best = np.ones((numPix, 1))

    x = np.random.random((1, nDim))
    x = bound(x, img)
    x_scale = scale(x)
    f = func(x_scale)

    x_best[0, :] = x
    f_best[0, :] = f
    # print('Confidence:', f_best)

    # x_store = []
    # f_store = []
    global  i
    for i in range(maxiter):
        dir = np.random.random((1, nDim))-0.5
        dir = dir / np.linalg.norm(dir)

        x_left = x + cilia_size*dir
        x_left = bound(x_left, img)
        x_left_scale = scale(x_left)
        f_left = func(x_left_scale)

        x_right = x - cilia_size*dir
        x_right = bound(x_right, img)
        x_right_scale = scale(x_right)
        f_right = func(x_right_scale)

        step_size = c*cilia_size
        x = x - step_size*np.sign(f_left - f_right)*dir
        x = bound(x, img)
        x_scale = scale(x)
        f = func(x_scale)

        if np.any(f < f_best):
            ind_to_remove = np.argmax(f_best)
            f_best[ind_to_remove, :] = f
            x_best[ind_to_remove, :] = x
            # print('iter:', i, 'Confidence:', f_best)
            if success_fun(scale(x_best).reshape((1,-1)), 0):
                return x_scale
        else:
            best_ind = np.argmin(f_best)
            f = f_best[best_ind, :]
            x = x_best[best_ind, :]
        
        # x_store += [x]
        # f_store += [f]

        # cilia_size *= decay_factor
        # cilia_size = cilia_size0/np.power(1 + decay_rho*i/10, decay_eta)

    return x_scale