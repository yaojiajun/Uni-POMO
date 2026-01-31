<h1 align="center"> MVMoE: Multi-Task Vehicle Routing Solver with Mixture-of-Experts </h1>

## How to Run

<details>
    <summary><strong>Train</strong></summary>


```shell
# Default: --problem_size=100 --pomo_size=100 --gpu_id=0
# 0. POMO
python train.py --problem={PROBLEM} --model_type=SINGLE

# 1. POMO-MTL
python train.py --problem=Train_ALL --model_type=MTL

# 2. MVMoE/4E 
python train.py --problem=Train_ALL --model_type=MOE --num_experts=4 --routing_level=node --routing_method=input_choice

# 3. MVMoE/4E-L
python train.py --problem=Train_ALL --model_type=MOE_LIGHT --num_experts=4 --routing_level=node --routing_method=input_choice
```

</details>

<details>
    <summary><strong>Evaluation</strong></summary>


```shell
# 0. POMO
python test.py --problem={PROBLEM} --model_type=SINGLE --checkpoint={MODEL_PATH}

# 1. POMO-MTL
python test.py --problem=ALL --model_type=MTL --checkpoint={MODEL_PATH}

# 2. MVMoE/4E
python test.py --problem=ALL --model_type=MOE --num_experts=4 --routing_level=node --routing_method=input_choice --checkpoint={MODEL_PATH}

# 3. MVMoE/4E-L
python test.py --problem=ALL --model_type=MOE_LIGHT --num_experts=4 --routing_level=node --routing_method=input_choice --checkpoint={MODEL_PATH}

# 4. Evaluation on CVRPLIB
python test.py --problem=CVRP --model_type={MODEL_TYPE} --checkpoint={MODEL_PATH} --test_set_path=../data/CVRP-LIB
```

</details>

<details>
    <summary><strong>Baseline</strong></summary>


```shell
# 0. LKH3 - Support for ["CVRP", "OVRP", "VRPL", "VRPTW"]
python LKH_baseline.py --problem={PROBLEM} --datasets={DATASET_PATH} -n=1000 --cpus=32 -runs=1 -max_trials=10000

# 1. HGS - Support for ["CVRP", "VRPTW"]
python HGS_baseline.py --problem={PROBLEM} --datasets={DATASET_PATH} -n=1000 --cpus=32 -max_iteration=20000

# 2. OR-Tools - Support for all 16 VRP variants
python OR-Tools_baseline.py --problem={PROBLEM} --datasets={DATASET_PATH} -n=1000 --cpus=32 -timelimit=20
```

</details>


## Citation

```tex
@inproceedings{zhou2024mvmoe,
title       ={MVMoE: Multi-Task Vehicle Routing Solver with Mixture-of-Experts},
author      ={Jianan Zhou and Zhiguang Cao and Yaoxin Wu and Wen Song and Yining Ma and Jie Zhang and Chi Xu},
booktitle   ={International Conference on Machine Learning},
year        ={2024}
}
```
