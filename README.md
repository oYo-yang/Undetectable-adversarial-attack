The code demonstrate the algorithm, inspired from simple behaviour of a unicellular microorganism, to fool Convolutional Neural Networks (CNNs). The current code will collect statistical results by attack a bunch of random images. 

The files contain:
    1. attack_script.py: The main script to run the attack on the selected CNN model with the specified settings. Read the comments to see how to change the target network and other settings of the fooling attack algorithm.
    2. attacker_algorithm.py: The core fooling attack algorithm.
    2. attack.py: Some helper functions for the attacking algorithm.
    3. helper.py: Some other helper functions.
    4. Networks: Folder containing the LeNet-5 and ResNet models developed in tensorflow.