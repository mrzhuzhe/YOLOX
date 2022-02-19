import os

from yolox.exp import Exp as MyExp

content = "../reef-data/out"

class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 1.0
        self.width = 1.0
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # Define yourself dataset path
        self.data_dir = content + "/all"
        self.train_ann = "annotations_train_2.json"
        self.val_ann = "annotations_valid_2.json"

        self.num_classes = 1

        self.warmup_epochs = 5
        self.max_epoch = 25
        self.warmup_lr = 0
        self.basic_lr_per_img = 0.01 / 64.0
        self.scheduler = "yoloxwarmcos"
        self.no_aug_epochs = 2
        self.min_lr_ratio = 0.05
        self.ema = True
        self.multiscale_range = 10

        self.weight_decay = 5e-4
        self.momentum = 0.9
        self.print_interval = 10
        self.eval_interval = 1

        self.data_num_workers = 4
            
        self.mosaic_prob = 1.0
        self.mixup_prob = 1.0
        self.hsv_prob = 1.0
        self.flip_prob = 0.5
        self.degrees = 10.0
        self.translate = 0.1
        #self.mosaic_scale = (0.1, 2)
        self.mosaic_scale = (0.5, 1.5)
        self.mixup_scale = (0.5, 1.5)
        #self.mixup_scale = (0.1, 2)
        self.shear = 2.0
        self.enable_mixup = True


        #self.input_size = (720, 1280)
        self.input_size = (736, 1280)
        #self.random_size = (35, 45)
        #self.random_size = (10, 20) # ????


        #self.test_size = (736, 1280)
        #self.test_size = (640, 1120)
        self.test_size = (1120, 1920)
        #self.test_size = (1472, 2560)

        self.test_conf = 0.1
        self.nmsthre = 0.65
