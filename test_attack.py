import pickle
import numpy as np
import pandas as pd
import matplotlib
from keras.datasets import cifar10
from keras import backend as K
from matplotlib import pyplot as plt

import attacker_algorithm
# Custom Networks
from networks.lenet import LeNet
from networks.resnet import ResNet
matplotlib.style.use('ggplot')
np.random.seed(100)
import helper
import time

time_start = time.time()  # 记录开始时间
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
lenet = LeNet()
resnet = ResNet()
models = [lenet, resnet]


def attacker(func, numPix, bounds, success_fun, maxiter=400):
    cilia_size0 = 0.5
    cilia_size = cilia_size0
    c = 1.0  # step_size/cilia_size
    decay_rho = 1.2
    decay_eta = 0.7

    nDim = len(bounds)  #bounds的大小

    bounds_ = np.array(bounds)  #将bounds转换成数组进行运算
    lb = bounds_[:, 0]   #lb:[0 0 0 0 0 ] 取所有集合的第n个数据
    ub = bounds_[:, 1]   #ub:[32 32 256 256 256]
    bounds_diff = ub - lb - 1

    #r,c,rgb在[0,1]之间
    def bound(x_in):
        return np.clip(x_in, 0, 1)

    #将[0,1]之间的r,c,rgb还原成图像中的表述
    def scale(x_in):
        return (x_in * bounds_diff + lb).astype(np.int)

    x_best = np.zeros((numPix, nDim))
    f_best = np.ones((numPix, 1))

    x = np.random.random((1, nDim))  #生成1行nDim列的浮点数，浮点数从0-1中随机
    x = bound(x)
    x_scale = scale(x)
    f = func(x_scale)

    x_best[0, :] = x #第0个集合的所有数据为x
    f_best[0, :] = f
    # print('Confidence:', f_best)

    # x_store = []
    # f_store = []
    global i
    for i in range(maxiter):
        dir = np.random.random((1, nDim)) - 0.5
        dir = dir / np.linalg.norm(dir) #linalg.norm计算l2范数

        x_left = x + cilia_size * dir
        x_left = bound(x_left)
        x_left_scale = scale(x_left)
        f_left = func(x_left_scale)

        x_right = x - cilia_size * dir
        x_right = bound(x_right)
        x_right_scale = scale(x_right)
        f_right = func(x_right_scale)

        step_size = c * cilia_size
        x = x - step_size * np.sign(f_left - f_right) * dir
        x = bound(x)
        x_scale = scale(x)
        f = func(x_scale)

        if np.any(f < f_best):
            ind_to_remove = np.argmax(f_best) #f_best最大值对应的索引
            f_best[ind_to_remove, :] = f
            x_best[ind_to_remove, :] = x
            # print('iter:', i, 'Confidence:', f_best)
            if success_fun(scale(x_best[ind_to_remove, :]).reshape((1, -1)), 0):
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

def plot_image(image, label_true=None, class_names=None, label_pred=None):
    if image.ndim == 4 and image.shape[0] == 1:
        image = image[0]

    print("Iterations:" + str(i))

    time_end = time.time()  # 记录结束时间
    time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
    print('totally time is ' + str('%.2f'%time_sum) + 's')

    plt.grid()
    true_class = y_test[image_id, 0]
    prior_confidence = model.predict_one(x_test[image_id])[true_class]
    labels_pred_name = class_names[label_pred]
    print('Pre-attack:'+class_names[label_true] + "(" + str(
                round(prior_confidence*100, 1)) + "%)")
    print('Post-attack:' + labels_pred_name + "(" + str(
                round(post_confidence*100, 1)) + " %)")
    plt.axis('off')
    plt.margins(0,0)
    plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)
    plt.imshow(image.astype(np.uint8))

    # # Show true and predicted classes
    # if label_true is not None and class_names is not None:
    #     labels_true_name = class_names[label_true]
    #     if label_pred is None:
    #         xlabel = "True: " + labels_true_name
    #     else:
    #         # Name of the predicted class
    #         labels_pred_name = class_names[label_pred]
    #         # xlabel = "True: " + labels_true_name + "\nPredicted: " + labels_pred_name
    #
    #         true_class = y_test[image_id, 0]
    #         prior_confidence = model.predict_one(x_test[image_id])[true_class]
    #         xlabel = "Pre-attack: " + str(labels_true_name) + "(" + str(
    #             round(prior_confidence*100, 1)) + "%)" + "\nPost-attack: " + str(labels_pred_name) + "(" + str(
    #             round(post_confidence*100, 1)) + " %)" + "\nIterations: " + str(i)
    #     # Show the class on the x-axis
    #     plt.xlabel(xlabel)

    plt.xticks([])  # Remove ticks from the plot
    plt.yticks([])
    plt.show()  # Show the plo

#产生扰动图像
def perturb_image(xs, img):
    # If this function is passed just one perturbation vector,
    # pack it in a list to keep the computation the same
    if xs.ndim < 2:
        xs = np.array([xs])

    # Copy the image n == len(xs) times so that we can
    # create n new perturbed images

    tile = [len(xs)] + [1] * (xs.ndim + 1)
    imgs = np.tile(img, tile) #复制img数组,变成四维

    # Make sure to floor the members of xs as int types
    xs = xs.astype(int)

    for x, img in zip(xs, imgs):
        # Split x into an array of 5-tuples (perturbation pixels)
        # i.e., [[x,y,r,g,b], ...]
        pixels = np.split(x, len(x) // 5)
        for pixel in pixels:
            # At each pixel's x,y position, assign its rgb value
            x_pos, y_pos, *rgb = pixel
            img[x_pos, y_pos] = rgb

    return imgs

#返回图像的置信度
def predict_classes(xs, img, target_class, model, minimize=True):
    # Perturb the image with the given pixel(s) x and get the prediction of the model
    imgs_perturbed = perturb_image(xs, img)
    predictions = model.predict(imgs_perturbed)[:,target_class]
    # This function should always be minimized, so return its complement if needed
    return predictions if minimize else 1 - predictions

def attack_success(x, img, target_class, model, targeted_attack=False, verbose=False):
    # Perturb the image with the given pixel(s) and get the prediction of the model
    attack_image = perturb_image(x, img)

    confidence = model.predict(attack_image)[0]
    predicted_class = np.argmax(confidence)

    # If the prediction is what we want (misclassification or
    # targeted classification), return True
    if (verbose):
        print('Confidence:', confidence[target_class])
    if ((targeted_attack and predicted_class == target_class) or
            (not targeted_attack and predicted_class != target_class)):
        return True

def attack(img_id, model, target=None, pixel_count=1, verbose=False, plot=False):
    targeted_attack = target is not None      #判断是否为目标攻击，若为目标攻击则为True;非目标攻击则为False
    target_class = target if targeted_attack else y_test[img_id, 0]    #如果是目标攻击，则返回目标类；非目标攻击，返回图片属于的类

    bounds = [(0,32), (0,32), (0,256), (0,256), (0,256)] * pixel_count

    predict_fn = lambda xs: predict_classes(
        xs, x_test[img_id], target_class, model, target is None)
    callback_fn = lambda x, convergence: attack_success(
        x, x_test[img_id], target_class, model, targeted_attack, verbose)

    attack_result = attacker(predict_fn, pixel_count, bounds, callback_fn) #返回要修改像素的r,c,rgb

    attack_image = perturb_image(attack_result, x_test[img_id])[0] #攻击后的图像
    prior_probs = model.predict_one(x_test[img_id])   #攻击前的置信度
    predicted_probs = model.predict_one(attack_image) #攻击后的置信度
    predicted_class = np.argmax(predicted_probs) #索引攻击后最大的置信度
    actual_class = y_test[img_id, 0]
    global post_confidence
    post_confidence = predicted_probs[predicted_class]

    success = predicted_class != actual_class
    # cdiff = prior_probs[actual_class] - predicted_probs[actual_class]

    plot_image(attack_image, actual_class, class_names, predicted_class)

    return attack_result

image_id = 8242
pixels = 1
model = resnet

_ = attack(image_id, model, pixel_count=pixels, verbose=True)






