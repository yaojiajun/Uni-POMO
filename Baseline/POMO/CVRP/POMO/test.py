##########################################################################################
# Machine Environment Config
import time
import random
import numpy as np
import torch  # <--- 导入 torch 用于设置种子

DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE
CUDA_DEVICE_NUM = 0


# --- 1. 定义随机种子初始化函数 ---
def seed_everything(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
    torch.backends.cudnn.deterministic = True  # 保证卷积等操作的确定性
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to: {seed}")


##########################################################################################
# Path Config

import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for problem_def
sys.path.insert(0, "../..")  # for utils

##########################################################################################
# import

import logging
from utils.utils import create_logger, copy_all_src

from VRPTester import VRPTester as Tester

##########################################################################################
# parameters

env_params = {
    'problem_type': 'CVRP',
    'problem_size': 100,
    'pomo_size': 100,
}

model_params = {
    'embedding_dim': 128,
    'sqrt_embedding_dim': 128 ** (1 / 2),
    'encoder_layer_num': 6,
    'qkv_dim': 16,
    'head_num': 8,
    'logit_clipping': 10,
    'ff_hidden_dim': 512,
    'eval_type': 'argmax',
}

tester_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'model_load': {
        'path': './result/100',
        'epoch': 'checkpoint-10000POMOhy+M+N100',
    },
    'test_episodes': 5000,
    'test_batch_size': 1,
    'augmentation_enable': True,
    'aug_factor': 1,
    'aug_batch_size': 1,
    'test_data_load': {
        'enable': True,
        'filename': '../../../Test_instances/data_' + env_params['problem_type'] + '_' + str(env_params['problem_size']) \
                    + '_' + str(5000) + '.pt'
    },
}
if tester_params['augmentation_enable']:
    tester_params['test_batch_size'] = tester_params['aug_batch_size']

logger_params = {
    'log_file': {
        'desc': 'test_cvrp100',
        'filename': 'log.txt'
    }
}


##########################################################################################
# main

def main():
    # --- 2. 初始化随机种子 ---
    # seed_everything(2026)

    if DEBUG_MODE:
        _set_debug_mode()

    create_logger(**logger_params)
    _print_config()

    tester = Tester(env_params=env_params,
                    model_params=model_params,
                    tester_params=tester_params)

    copy_all_src(tester.result_folder)

    # --- 3. 开始计时 ---
    start_time = time.time()
    tester.run()
    end_time = time.time()

    # --- 4. 计算并打印时间 ---
    total_time = end_time - start_time
    print(f"Total Inference Time: {total_time:.4f}s")


def _set_debug_mode():
    global tester_params
    tester_params['test_episodes'] = 10


def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]


##########################################################################################

if __name__ == "__main__":
    main()