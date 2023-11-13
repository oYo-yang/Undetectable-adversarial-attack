# some part of codes used from https://github.com/Hyperparticle/one-pixel-attack-keras

import numpy as np
import tensorflow as tf
import pickle
import pandas as pd

# Custom Networks
from networks.lenet import LeNet
from networks.resnet import ResNet

# Helper functions
import attack
import helper

np.random.seed(100)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
class_names = ['airplane', 'automobile', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# uncomment the network to attack it
model = [LeNet()]
# model = [ResNet()]

_, correct_imgs = helper.evaluate_models(model, x_test, y_test)
correct_imgs = pd.DataFrame(correct_imgs, columns=[
                            'name', 'img', 'label', 'confidence', 'pred'])

attacker = attack.PixelAttacker(model, (x_test, y_test), class_names)

def attack_all(models, samples=2, pixels=1024, targeted=False, verbose=False):
    results = []
    for model in models:
        model_results = []
        valid_imgs = correct_imgs[correct_imgs.name == model.name].img
        img_samples = np.random.choice(valid_imgs, samples, replace=False)

        for pixel_count in pixels:
            for i, img in enumerate(img_samples):
                print(model.name, '- image', img,
                      '-', i+1, '/', len(img_samples))
                targets = [None] if not targeted else [np.random.randint(0,10)]

                for target in targets:
                    result = []
                    for i in range(5):
                        result_ = attacker.attack(img, model, target, pixel_count, verbose=verbose)
                        result.append(result_)
                    success = True if np.sum(result) > 1 else False
                    model_results.append([img, pixel_count, result, success])
                
                print('pixel_count', pixel_count, 'success ratio:', np.sum(np.array(model_results)[:,3])/len(model_results))

        results += model_results

    return results

num_samples = 2 # select the number of random images to attack
num_pixels = [1024] # select the number of pixels to use during the attack
is_targeted = False # select whether attack is targeted or untareted
results = attack_all(model, samples=num_samples, pixels=num_pixels, targeted=is_targeted)
with open('targeted_attack_bas_lenet135.pkl', 'wb') as file:
    pickle.dump(results, file)
