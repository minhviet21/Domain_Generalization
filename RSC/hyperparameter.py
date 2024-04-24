class Hyperparameter():
    def __init__(self):
        self.model = "ResNet18"
        self.resnet_dropout = 0.1
        self.nonlinear_classifier = False
        self.lr = 0.001
        self.weight_decay = 0.0
        self.rsc_f_drop_factor = 1/3
        self.rsc_b_drop_factor = 1/3
        self.batch_size = 16

