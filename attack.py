import numpy as np

# Helper functions
import helper
import attacker_algorithm
from keras.datasets import cifar10
# code used from https://github.com/Hyperparticle/one-pixel-attack-keras
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
class PixelAttacker:
    def __init__(self, models, data, class_names, dimensions=(32, 32)):
        # Load data and model
        self.models = models
        self.x_test, self.y_test = data
        self.class_names = class_names
        self.dimensions = dimensions

        # network_stats, correct_imgs = helper.evaluate_models(self.models, self.x_test, self.y_test)
        # self.correct_imgs = pd.DataFrame(correct_imgs, columns=['name', 'img', 'label', 'confidence', 'pred'])
        # self.network_stats = pd.DataFrame(network_stats, columns=['name', 'accuracy', 'param_count'])

    def predict_classes(self, xs, img, target_class, model, minimize=True):
        # Perturb the image with the given pixel(s) x and get the prediction of the model
        imgs_perturbed = helper.perturb_image(xs, img)
        predictions = model.predict(imgs_perturbed)[:,target_class]
        # This function should always be minimized, so return its complement if needed
        return predictions if minimize else 1 - predictions

    def attack_success(self, x, img, target_class, model, targeted_attack=False, verbose=False):
        # Perturb the image with the given pixel(s) and get the prediction of the model
        attack_image = helper.perturb_image(x, img)

        confidence = model.predict(attack_image)[0]
        predicted_class = np.argmax(confidence)
        
        # If the prediction is what we want (misclassification or 
        # targeted classification), return True
        if (verbose):
            print('Confidence:', confidence[target_class])
        if ((targeted_attack and predicted_class == target_class) or
            (not targeted_attack and predicted_class != target_class)):
            return True

    def attack(self, img, model, target=None, pixel_count=1024, verbose=False, plot=False):
        targeted_attack = target is not None
        target_class = target if targeted_attack else self.y_test[img,0]
        
        dim_x, dim_y = self.dimensions

        bounds = [(0, dim_x), (0, dim_y), (0, 256), (0, 256), (0, 256)]* pixel_count
        
        predict_fn = lambda xs: self.predict_classes(
            xs, self.x_test[img], target_class, model, target is None)
        callback_fn = lambda x, convergence: self.attack_success(
            x, self.x_test[img], target_class, model, targeted_attack, verbose)

        attack_result = attacker_algorithm.attacker(predict_fn, pixel_count, bounds, callback_fn, x_test[img])

        return attack_result
