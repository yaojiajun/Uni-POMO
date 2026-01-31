import vrplib
import numpy as np
import torch
import yaml
import json
import time
import os
from CVRPModel import CVRPModel
from CVRPEnv import CVRPEnv
from utils import rollout, check_feasible
import random
import random

def seed_everything(seed=1234):
    """
    设置全局随机种子以保证结果可复现
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 确保针对卷积等操作的算法选择是确定的
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"随机种子已设置为: {seed}")
def rollout(model, env, eval_type='greedy'):
    env.reset()
    actions = []
    probs = []
    reward = None
    state, reward, done = env.pre_step()

    while not done:
        cur_dist,current_node = env.get_cur_feature() #                cur_dist,current_node = self.env.get_cur_feature()
        selected, one_step_prob = model.one_step_rollout(state, cur_dist,current_node , eval_type=eval_type)
        # selected, one_step_prob = model(state)
        state, reward, done = env.step(selected)
        actions.append(selected)
        probs.append(one_step_prob)

    actions = torch.stack(actions, 1)
    if eval_type == 'greedy':
        probs = None
    else:
        probs = torch.stack(probs, 1)

    return torch.transpose(actions, 1, 2), probs, reward,actions


class VRPLib_Tester:

    def __init__(self, config):
        self.config = config
        model_params = config['model_params']
        load_checkpoint = config['load_checkpoint']
        self.multiple_width = config['test_params']['pomo_size']

        # cuda
        USE_CUDA = config['use_cuda']
        if USE_CUDA:
            cuda_device_num = config['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            self.device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            self.device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')

        # load trained model
        self.model = CVRPModel(**model_params)
        checkpoint = torch.load(load_checkpoint, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        self.vrplib_path = 'data/VRPLib/Vrp-Set-X' if config['vrplib_set'] == 'X' else "data/VRPLib/Vrp-Set-XXL"
        self.repeat_times = 1
        self.aug_factor = config['test_params']['aug_factor']
        print("AUG_FACTOR: ", self.aug_factor)
        self.vrplib_results = None

    def test_on_vrplib(self):
        files = sorted(os.listdir(self.vrplib_path))
        vrplib_results = []
        total_time = 0.

        # 定义保存结果的文件名，仅包含 cost
        cost_log_file = f"costs_{self.config['vrplib_set']}_{time.strftime('%Y%m%d_%H%M%S')}.txt"

        # 打开文件准备写入
        with open(cost_log_file, 'w', encoding='utf-8') as f_out:
            for t in range(self.repeat_times):
                for name in files:
                    if '.sol' in name or not name.endswith('.vrp'):
                        continue

                    instance_name = name[:-4]
                    instance_file = os.path.join(self.vrplib_path, name)
                    solution_file = os.path.join(self.vrplib_path, instance_name + '.sol')

                    if not os.path.exists(solution_file):
                        continue

                    # 读取最优值仅为了计算打印时的 Gap，不影响保存内容
                    solution = vrplib.read_solution(solution_file)
                    optimal = solution['cost']

                    result_dict = {}
                    result_dict['run_idx'] = t

                    # 执行测试
                    self.test_on_one_ins(name=instance_name, result_dict=result_dict, instance=instance_file,
                                         solution=solution_file)

                    # --- 核心修改：仅保存 cost，且换行 ---
                    f_out.write(f"{result_dict['best_cost']}\n")
                    f_out.flush()  # 确保数据实时写入磁盘

                    total_time += result_dict['runtime']
                    # 这里将 gap 存入 vrplib_results 以便后续计算平均值
                    vrplib_results.append({'instance': instance_name, 'optimal': optimal, 'gap': result_dict['gap']})

                    print(f"Instance {instance_name}: Cost {result_dict['best_cost']}, Gap {result_dict['gap']:.4f}")

        # ============================== 新增统计逻辑 ==============================
        if len(vrplib_results) > 0:
            all_gaps = [res['gap'] for res in vrplib_results]
            avg_gap = sum(all_gaps) / len(all_gaps)
            print("\n" + "=" * 50)
            print(f"测试集规模: {len(vrplib_results)}")
            print(f"所有实例的平均 Gap: {avg_gap * 100:.4f}%")
            print("=" * 50)
        # =========================================================================

        print(f"\n>>> 所有 Cost 已保存至: {cost_log_file}")


    def test_on_one_ins(self, name, result_dict, instance, solution):
        start_time = time.time()
        instance = vrplib.read_instance(instance)
        solution = vrplib.read_solution(solution)
        optimal = solution['cost']
        problem_size = instance['node_coord'].shape[0] - 1
        multiple_width = min(problem_size, self.multiple_width)
        # multiple_width = problem_size

        # Initialize CVRP state
        env = CVRPEnv(self.multiple_width, self.device)

        aug_reward = None
        sep_augmentation = False
        if sep_augmentation:
            # compute only one augmented version each time to save gpu memory, repeat 8 times for each instance
            for idx in range(8):
                env.load_vrplib_problem(instance, aug_factor=self.aug_factor, aug_idx=idx)

                reset_state, reward, done = env.reset()
                self.model.eval()
                self.model.requires_grad_(False)
                self.model.pre_forward(reset_state)

                with torch.no_grad():
                    policy_solutions, policy_prob, rewards = rollout(self.model, env, 'greedy')
                # Return

                if aug_reward is not None:
                    aug_reward = rewards.reshape(self.aug_factor, 1, env.multi_width)
                    # shape: (augmentation, batch, multi)
                else:
                    aug_reward = rewards.reshape(1, 1, env.multi_width)

        else:
            env.load_vrplib_problem(instance, aug_factor=self.aug_factor, aug_idx=-1)

            reset_state, reward, done = env.reset()
            self.model.eval()
            self.model.requires_grad_(False)
            self.model.pre_forward(reset_state)

            with torch.no_grad():
                policy_solutions, policy_prob, rewards, actions = rollout(self.model, env, 'greedy')

            aug_reward = rewards.reshape(self.aug_factor, 1, env.multi_width)
            # shape: (augmentation, batch, multi)

        # 2. 获取 POMO 维度的最大值及其索引 (即哪个起始点最好)
        max_pomo_reward, pomo_idx = aug_reward.max(dim=2)
        # max_pomo_reward shape: (augmentation, 1)
        # pomo_idx shape: (augmentation, 1) -> 每个增强维度下的最佳起始点索引

        # 3. 获取 Augmentation 维度的最大值及其索引 (即哪种旋转/翻转最好)
        max_aug_pomo_reward, aug_idx = max_pomo_reward.max(dim=0)
        # max_aug_pomo_reward shape: (1,)
        aug_dim, seq_len, pomo_dim = actions.shape
        index_pomo = pomo_idx.unsqueeze(1).expand(-1, seq_len, -1)
        best_pomo_actions = torch.gather(actions, dim=2, index=index_pomo)

        index_aug = aug_idx.view(1, 1, 1).expand(-1, seq_len, 1)
        final_best_actions = torch.gather(best_pomo_actions, dim=0, index=index_aug).squeeze()

        # 2. 仅打印路径 (转换为 list 以确保不显示省略号 ...)
        print(final_best_actions.cpu().tolist())
        # shape: (batch,)
        aug_cost = -max_aug_pomo_reward.float()  # negative sign to make positive value

        best_cost = aug_cost
        end_time = time.time()
        elapsed_time = end_time - start_time
        if result_dict is not None:
            result_dict['best_cost'] = best_cost.cpu().numpy().tolist()[0]
            result_dict['scale'] = problem_size
            result_dict['gap'] = (result_dict['best_cost'] - optimal) / optimal
            result_dict['runtime'] = elapsed_time
            print(
                f"Instance {name}: Time {elapsed_time:.4f}s, Cost {result_dict['best_cost']}, Gap {result_dict['gap']:.4f}")



if __name__ == "__main__":
    seed_everything(1234)
    with open('config.yml', 'r', encoding='utf-8') as config_file:
        config = yaml.load(config_file.read(), Loader=yaml.FullLoader)

    tester = VRPLib_Tester(config=config)
    tester.test_on_vrplib()