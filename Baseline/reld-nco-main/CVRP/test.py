##########################################################################################
# Machine Environment Config

DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE

##########################################################################################
# Path Config

import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for problem_def
sys.path.insert(0, "../..")  # for utils

##########################################################################################

import logging
from utils import create_logger, copy_all_src
import numpy as np
import random
import torch, yaml
from CVRPTester import CVRPTester as Tester


##########################################################################################

def main():
    # ========================== 设置随机种子 1234 ==========================
    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # =====================================================================

    with open('config.yml', 'r', encoding='utf-8') as config_file:
        config = yaml.load(config_file.read(), Loader=yaml.FullLoader)

    create_logger(**config['logger_params'])
    _print_config()

    if config['test_params']['test_data_load']['enable']:
        config['test_params']['aug_batch_size'] = config['test_params']['test_batch_size']

        score, aug_score, gap = test(config)

        print(
            "CVRP{}: score: {:.4f}, aug score: {:.4f}, gap: {:.4f}%".format(config['env_params']['problem_size'], score,
                                                                            aug_score, gap * 100))
    else:
        test_data_setting = {
            # [Data path, Number of instances, test batch size, forcing_first_step: (True) Uniform random select the first step or (False) Get diverse first step with top probability]
            # 100: ['vrp100_test_lkh.txt', 10000, 1, False],
            # 200: ['vrp200_test_lkh.txt', 128, 1, False],
            # 500: ['vrp500_test_lkh.txt', 128, 1, False],
            # 1000: ['vrp1000_test_lkh.txt', 128, 1, False],
        }

        result = []
        for k, v in test_data_setting.items():
            k = int(k)
            config['env_params']['problem_size'] = k
            config['env_params']['pomo_size'] = min(k, config['test_params']['pomo_size'])

            config['test_params']['test_episodes'] = v[1]

            config['test_params']['aug_batch_size'] = v[2]
            config['test_params']['test_batch_size'] = v[2]
            config['model_params']['force_first_step'] = v[3]
            config['test_params']['test_data_load']['filename'] = 'data/' + v[0]

            score, aug_score, gap = test(config)

            print("Test done for CVRP{} ".format(k) + " ################################################")

            result.append((k, (score, aug_score, gap)))

        for res in result:
            print("CVRP{}: score: {:.4f}, aug score: {:.4f}, gap: {:.4f}%".format(res[0], res[1][0], res[1][1],
                                                                                  res[1][2] * 100))


def test(config):
    # 此处导入已被移至文件顶部，但保留函数逻辑
    if DEBUG_MODE:
        _set_debug_mode()

    tester = Tester(config)

    return tester.run()


def _set_debug_mode():
    global tester_params
    tester_params['test_episodes'] = 10


def _print_config():
    logger = logging.getLogger('root')
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]


if __name__ == "__main__":
    main()