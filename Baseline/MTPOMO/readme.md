------

This code develops a multi-task learning neural combinatorial optimization (NCO) model for cross-problem generalization for routing problems. 

Please cite the paper https://arxiv.org/abs/2402.16891, if you find the code helpful, 

@article{liu2024multi, \
    title={Multi-Task Learning for Routing Problem with Cross-Problem Zero-Shot Generalization}, \
    author={Liu, Fei and Lin, Xi and Zhang, Qingfu and Tong, Xialiang and Yuan, Mingxuan}, \
    journal={International Conference on Knowledge Discovery and Data Mining (KDD)}, \
    year={2024} \
}

## Files in MTNCO

+ MTPOMO: the implementation of multi-task learning with attribute composition based on POMO.
+ Trained_models: the pre-trained unified models with problem sizes 50 and 100 
+ Test_instances: test instances of 11 VRPs
+ utils: utils

## Train & Test

cd MTPOMO/POMO/

**Train:**  python train_n50.py

**Test:**  python test_n50.py

