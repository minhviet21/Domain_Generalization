from hyperparameter import Hyperparameter
from algorithms import ERM
input_shape = (3, 224, 224)  # Example input shape for an RGB image with size 224x224
num_classes = 10  # Example number of classes
num_domains = 3  # Example number of domains
hp = Hyperparameter()
erm_model = ERM(input_shape, num_classes, num_domains, hp)