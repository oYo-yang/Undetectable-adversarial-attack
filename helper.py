# functions used from https://github.com/Hyperparticle/one-pixel-attack-keras

import numpy as np

def perturb_image(xs, img):
    # If this function is passed just one perturbation vector,
    # pack it in a list to keep the computation the same
    if xs.ndim < 2:
        xs = np.array([xs])
    
    # Copy the image n == len(xs) times so that we can 
    # create n new perturbed images
    tile = [len(xs)] + [1]*(xs.ndim+1)
    imgs = np.tile(img, tile)
    
    # Make sure to floor the members of xs as int types
    xs = xs.astype(int)
    
    for x,img in zip(xs, imgs):
        # Split x into an array of 5-tuples (perturbation pixels)
        # i.e., [[x,y,r,g,b], ...]
        pixels = np.split(x, len(x) // 5)
        for pixel in pixels:
            # At each pixel's x,y position, assign its rgb value
            x_pos, y_pos, *rgb = pixel
            img[x_pos, y_pos] = rgb
    
    return imgs

def evaluate_models(models, x_test, y_test):
    correct_imgs = []
    network_stats = []
    for model in models:
        print('Evaluating', model.name)

        predictions = model.predict(x_test)

        correct = [[model.name, i, label, np.max(pred), pred]
                   for i, (label, pred)
                   in enumerate(zip(y_test[:, 0], predictions))
                   if label == np.argmax(pred)]
        accuracy = len(correct) / len(x_test)

        correct_imgs += correct
        network_stats += [[model.name, accuracy]]
    return network_stats, correct_imgs
