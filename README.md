# HandAugment
The winner method of [HANDS19 Challenge: Task 1 - Depth-Based 3D Hand Pose Estimation](https://sites.google.com/view/hands2019/challenge#h.p_Y9eLuCuXYN7U)

The code for paper: "HandAugment: A Simple Data Augmentation Method for Depth-Based 3D Hand Pose Estimation"

https://arxiv.org/abs/2001.00702


## Updates !!!!

**(2020-June-30)** upload test script and pretrained model.

## Required libraries

Python 3.6  
Numpy 1.17.2  
PyTorch 1.0.1  
OpenCV 4.1


## Usage
1. Clone this repo
    ```
    git clone https://github.com/wozhangzhaohui/HandAugment.git
    cd HandAugment
    ```
2. Download Hands19 dataset from [HANDS19 website](https://sites.google.com/view/hands2019/challenge).
Replace spaces in file path with underscores "_"
and link HANDS19 folder by ``` ln -s your-hands19-folder-path dataset/HANDS19_Challenge/```
3. Run the test script by command: ```bash run_test.sh```, the result is saved in output folder "output/stage1/result.txt".
4. The result file in "output/stage1/result.zip" can be submitted directly to [Hands19Task1](https://competitions.codalab.org/competitions/20913)


## Pre-trained model
We provide two stages pre-trained model for Hands19Task1 dataset.

Intermediate score at stage0 can reach 14.06

Final score at stage1 can reach 12.99


## HandAugment Architecture
![system_overview](resources/system_overview.png)

![augmented_patch1](resources/augmented_patch1.png)

![augmented_patch](resources/augmented_patch.png)

![data_synthesis](resources/data_synthesis.png)


## Quantitative Comparison
### HANDS19 Task1 Dataset
![hands19_result1](resources/hands19_result1.png)

![hands19_result](resources/hands19_result.png)

###NYU Dataset
![nyu_result](resources/nyu_result.png)


## Citation
```
@article{zhang2020handaugment,
  title={HandAugment: A Simple Data Augmentation Method for Depth-Based 3D Hand Pose Estimation},
  author={Zhang, Zhaohui and Xie, Shipeng and Chen, Mingxiu and Zhu, Haichao},
  journal={arXiv},
  pages={arXiv--2001},
  year={2020}
}
```
