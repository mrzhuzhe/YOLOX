import os

from yolox.exp import Exp as MyExp

content = "../reef-data/out"

class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.50
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # Define yourself dataset path
        self.data_dir = content + "/all"
        self.train_ann = "annotations_train_2.json"
        self.val_ann = "annotations_valid_2.json"

        self.num_classes = 1

        self.max_epoch = 15
        self.data_num_workers = 4
        self.eval_interval = 1    
        
        self.mosaic_prob = 1.0
        self.mixup_prob = 1.0
        self.hsv_prob = 1.0
        self.flip_prob = 0.5
        self.no_aug_epochs = 2
        
        self.mosaic_prob = 1.0
        self.mixup_prob = 1.0
        self.hsv_prob = 1.0
        self.flip_prob = 0.5
        self.degrees = 90.0
        self.translate = 0.1
        self.mosaic_scale = (0.1, 1.2)
        self.mixup_scale = (0.1, 1.2)
        self.shear = 2.0
        self.enable_mixup = True

        #self.input_size = (720, 1280)
        self.input_size = (736, 1280)
        self.random_size = (15, 25)
        self.test_size = (736, 1280)
        self.test_conf = 0.1
