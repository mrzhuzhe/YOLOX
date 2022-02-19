import os

from yolox.exp import Exp as MyExp

content = "../reef-data/out"

class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        
        self.depth = 1.33
        self.width = 1.25

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
        
        #self.input_size = (720, 1280)
        self.input_size = (800, 1280)
        self.mosaic_scale = (0.5, 1.5)
        self.random_size = (35, 45)
        self.test_size = (800, 1280)
