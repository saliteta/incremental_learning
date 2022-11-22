# incremental_learning
Adding new knowledge without accessing the old dataset

- Folder structure
```
├── config.py
├── display.py
├── elastic_weight_consolidation.py
├── ewc_result
│   ├── backbone.tar
│   ├── classifier.tar
│   └── cm
│       ├── 0.npy
│       ├── 1.npy
│       ├── 2.npy
│       ├── 3.npy
│       ├── 4.npy
│       ├── 5.npy
│       ├── 6.npy
│       ├── 7.npy
│       ├── 8.npy
│       └── 9.npy
├── learning_without_forgeting.py
├── log
│   └── log.txt
├── logit_test
│   ├── confusion_matrix.png
│   ├── logit.npy
│   ├── out_logit.csv
│   └── target.npy
├── loss_function.py
├── lwf_result
│   ├── backbone.tar
│   ├── classifier.tar
│   ├── cm
│   │   ├── 0.npy
│   │   ├── 1.npy
│   │   ├── 2.npy
│   │   ├── 3.npy
│   │   ├── 4.npy
│   │   ├── 5.npy
│   │   ├── 6.npy
│   │   ├── 7.npy
│   │   ├── 8.npy
│   │   └── 9.npy
│   ├── logit.npy
│   └── target.npy
├── model
│   ├── 10
│   │   ├── backbone.tar
│   │   ├── classifier.tar
│   │   ├── description.txt
│   │   ├── logit.npy
│   │   └── target.npy
│   ├── 11
│   │   ├── backbone.tar
│   │   ├── classifier.tar
│   │   ├── description.txt
│   │   ├── logit.npy
│   │   └── target.npy
│   ├── 12
│   │   ├── backbone.tar
│   │   ├── classifier.tar
│   │   ├── description.txt
│   │   ├── logit.npy
│   │   └── target.npy
│   ├── 13
│   │   ├── backbone.tar
│   │   ├── classifier.tar
│   │   ├── description.txt
│   │   ├── logit.npy
│   │   └── target.npy
│   ├── 19
│   │   ├── backbone.tar
│   │   └── classifier.tar
│   ├── 4
│   │   ├── backbone.tar
│   │   ├── classifier.tar
│   │   ├── description.txt
│   │   ├── logit.npy
│   │   └── target.npy
│   ├── 5
│   │   ├── backbone.tar
│   │   ├── classifier.tar
│   │   ├── description.txt
│   │   ├── logit.npy
│   │   └── target.npy
│   ├── 6
│   │   ├── backbone.tar
│   │   ├── classifier.tar
│   │   ├── description.txt
│   │   ├── logit.npy
│   │   └── target.npy
│   ├── 7
│   │   ├── backbone.tar
│   │   ├── classifier.tar
│   │   ├── description.txt
│   │   ├── logit.npy
│   │   └── target.npy
│   ├── 8
│   │   ├── backbone.tar
│   │   ├── classifier.tar
│   │   ├── description.txt
│   │   ├── logit.npy
│   │   └── target.npy
│   ├── 9
│   │   ├── backbone.tar
│   │   ├── classifier.tar
│   │   ├── description.txt
│   │   ├── logit.npy
│   │   └── target.npy
│   ├── backbone.tar
│   ├── classifier.tar
│   └── model_path_name_standard.md
├── model.py
├── model_test.py
├── no_incremental_learning_result
│   ├── backbone.tar
│   ├── classifier.tar
│   ├── dict.txt
│   ├── logits
│   │   ├── epochs0.npy
│   │   ├── epochs1.npy
│   │   └── epochs2.npy
│   └── targets
│       ├── epochs0.npy
│       ├── epochs1.npy
│       └── epochs2.npy
├── pretrained_model
│   └── resnet50_places365.pth.tar
├── __pycache__
│   ├── config.cpython-37.pyc
│   ├── display.cpython-37.pyc
│   ├── loss_function.cpython-37.pyc
│   ├── test.cpython-37.pyc
│   ├── train.cpython-37.pyc
│   └── utils.cpython-37.pyc
├── run.py
├── test.py
├── train.py
├── train_with_no_incremental.py
└── utils.py

23 directories, 114 files
```
