#!/usr/local/bin/python3

import numpy as np
import pickle

with open('targeted_attack_bas_resnet11.pkl', 'rb') as file:
    data = pickle.load(file)

with open('targeted_attack_bas_resnet12.pkl', 'rb') as file:
    data += pickle.load(file)

with open('targeted_attack_bas_resnet13.pkl', 'rb') as file:
    data += pickle.load(file)

with open('targeted_attack_bas_resnet31.pkl', 'rb') as file:
    data += pickle.load(file)

with open('targeted_attack_bas_resnet32.pkl', 'rb') as file:
    data += pickle.load(file)

with open('targeted_attack_bas_resnet33.pkl', 'rb') as file:
    data += pickle.load(file)

with open('targeted_attack_bas_resnet51.pkl', 'rb') as file:
    data += pickle.load(file)

with open('targeted_attack_bas_resnet52.pkl', 'rb') as file:
    data += pickle.load(file)

with open('targeted_attack_bas_resnet53.pkl', 'rb') as file:
    data += pickle.load(file)

with open('targeted_attack_bas_resnet135.pkl', 'wb') as file:
    pickle.dump(data, file)