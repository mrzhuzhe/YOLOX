import os

from yolox.exp import Exp as MyExp

content = "../reef-data/out"

class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.67
        self.width = 0.75
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # Define yourself dataset path
        self.data_dir = content + "/all"
        self.train_ann = "annotations_train_2.json"
        self.val_ann = "annotations_valid_2.json"

        self.num_classes = 1

        self.max_epoch = 20
        self.data_num_workers = 2
        self.eval_interval = 1    
        
        self.mosaic_prob = 1.0
        self.mixup_prob = 1.0
        self.hsv_prob = 1.0
        self.flip_prob = 0.5
        self.no_aug_epochs = 2
        
        """
        self.degrees = 10.0
        self.translate = 0.1
        self.scale = (0.1, 2)
        self.mosaic_scale = (0.8, 1.6)
        self.shear = 2.0
        self.perspective = 0.0
        self.enable_mixup = True
        """


        self.input_size = (960, 960)
        self.mosaic_scale = (0.5, 1.5)
        self.random_size = (10, 20)
        self.test_size = (960, 960)
