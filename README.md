# incremental_learning
Adding new knowledge without accessing the old dataset

- Folder structure
I have not uploaded all the content to github due to the size limitation. However, one can check out the folder structure in the file folder_tree.txt

# Introduction
- Here is a pytorch implementation to two important incremental learning methods. 
- LWF(learning without forgetting https://ieeexplore.ieee.org/document/8107520)
- EWC(Elastic Weight Consolidation (EWC) Overcoming catastrophic forgetting in neural networks https://arxiv.org/abs/1612.00796)

# LWF
- Using distillation to transform the knowledge of old model together with the new dataset's label
- Trainning and test code is in learning_without_forgetting.py

# EWC
- Using Fisher Information Matrix to determine which parameters are important to the old classes so that we don't alter those parameters in new training 
- Training and test cide us in elastic_weight_consolidation.py

# without incremental learning technique
- This will lead to catastrophic forgetting
- code is in train_with_no_incremental.py

# run code
- one needs to modify the config.py file carefully
- one needs to check out the "subset"'s meaning carefully

# result display
- The result will be store as a npy file whose name represent the epochs
- If one needs to display those matrix, one may need to modify python scripts in display_incremental_learning folder

# dataset
- we use the SUN_RGBD as test dataset, we use places365 pretrained model to train our new network
- we start to train the model from 4 classes only and add another two new classes in experiments
- LWF get a good result while for EWC, it does not converge
- The confusion matrices of each epochs are shown below
