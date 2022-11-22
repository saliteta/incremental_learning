
# training overall setting
LOAD_MODEL_PATH = 'pretrained_model/resnet50_places365.pth.tar'
BATCHSIZE = 256
EVAL_BACHSIZE = 128
EPOCHS = 10
DATASET = '/data/dataset/SUN_RGBD'
BACKBONE_TRAIN = False
CLASSIFIER_TRAIN = True

# stored model path
# backbone model
BACKBONE_STORE_PATH = './model/backbone.tar'
# classifier model
CLASSIFIER_STORE_PATH = './model/classifier.tar'



# optimizer setting
LEARNING_RATE = 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4

# hardware setting
WORKERS_NUMBER = 16

# for utils
IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

class LR_Params():
    def __init__(self):
        # -----------------------------------------------------------------------------
        # Training settings
        # -----------------------------------------------------------------------------
        self.START_EPOCH = 0
        self.EPOCHS = 40
        self.WARMUP_EPOCHS = 20
        self.WEIGHT_DECAY = 0.3
        self.BASE_LR = 2e-2
        self.WARMUP_LR = 5e-6
        self.MIN_LR = 5e-5
        # Clip gradient norm
        self.CLIP_GRAD = 5.0
        # Auto resume from latest checkpoint
        self.AUTO_RESUME = True
        # Gradient accumulation steps
        # could be overwritten by command line argument
        self.ACCUMULATION_STEPS = 1
        # Whether to use gradient checkpointing to save memory
        # could be overwritten by command line argument
        self.USE_CHECKPOINT = False

        # LR scheduler
        self.LR_SCHEDULER_NAME = 'cosine'
        # Epoch interval to decay LR, used in StepLRScheduler
        self.LR_SCHEDULER_DECAY_EPOCHS = 30
        # LR decay rate, used in StepLRScheduler
        self.LR_SCHEDULER_DECAY_RATE = 0.1

