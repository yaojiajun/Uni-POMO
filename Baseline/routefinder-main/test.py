import argparse
import os
import pickle
import time
import warnings
import random
import numpy as np

import torch
from rl4co.data.transforms import StateAugmentation
from rl4co.utils.ops import gather_by_index, unbatchify
from tqdm.auto import tqdm

from routefinder.data.utils import get_dataloader
from routefinder.envs import MTVRPEnv
from routefinder.models import RouteFinderBase, RouteFinderMoE
from routefinder.models.baselines.mtpomo import MTPOMO
from routefinder.models.baselines.mvmoe import MVMoE

# Tricks for faster inference
try:
    torch._C._jit_set_profiling_executor(False)
    torch._C._jit_set_profiling_mode(False)
except AttributeError:
    pass

torch.set_float32_matmul_precision("medium")
import torch
# 将报错中提到的类加入安全白名单
from routefinder.envs.mtvrp.env import MTVRPEnv
torch.serialization.add_safe_globals([MTVRPEnv])

def test(
        policy,
        td,
        env,
        num_augment=8,
        augment_fn="dihedral8",
        num_starts=None,
        device="cuda",
):
    costs_bks = td.get("costs_bks", None)

    with torch.inference_mode():
        with (
                torch.amp.autocast("cuda")
                if "cuda" in str(device)
                else torch.inference_mode()
        ):
            n_start = env.get_num_starts(td) if num_starts is None else num_starts

            if num_augment > 1:
                td = StateAugmentation(num_augment=num_augment, augment_fn=augment_fn)(td)

            out = policy(td, env, phase="test", num_starts=n_start, return_actions=True)
            reward = unbatchify(out["reward"], (num_augment, n_start))

            if n_start > 1:
                max_reward, max_idxs = reward.max(dim=-1)
                instance_reward = reward[0:1]

                # 2. 对第 1 维（8 个 Augmentation）取最大值，并保持维度
                # 此时维度变为 (1, 1, 100)
                max_reward1, _ = instance_reward.max(dim=1, keepdim=True)

                # 3. 打印结果
                print("最终维度:", max_reward1.shape)  # torch.Size([1, 1, 100])

                # 4. 打印这 100 个数据的值
                # 使用 .view(-1) 将其展平为一维方便打印
                values = max_reward1.view(-1).cpu().numpy()
                print("100个起始点的奖励值：")
                print(values)
                out.update({"max_reward": max_reward})

                if out.get("actions", None) is not None:
                    actions = unbatchify(out["actions"], (num_augment, n_start))
                    out.update(
                        {
                            "best_multistart_actions": gather_by_index(
                                actions, max_idxs, dim=max_idxs.dim()
                            )
                        }
                    )
                    out["actions"] = actions

            if num_augment > 1:
                reward_ = max_reward if n_start > 1 else reward
                max_aug_reward, max_idxs = reward_.max(dim=1)
                out.update({"max_aug_reward": max_aug_reward})

                if costs_bks is not None:
                    gap_to_bks = (
                            100
                            * (-max_aug_reward - torch.abs(costs_bks))
                            / torch.abs(costs_bks)
                    )
                    out.update({"gap_to_bks": gap_to_bks})

                if out.get("actions", None) is not None:
                    actions_ = (
                        out["best_multistart_actions"] if n_start > 1 else out["actions"]
                    )
                    out.update({"best_aug_actions": gather_by_index(actions_, max_idxs)})

            if out.get("gap_to_bks", None) is None:
                out.update({"gap_to_bks": torch.tensor(69420.0)})

            return out


if __name__ == "__main__":
    seed = 69420
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--problem", type=str, default="all")
    parser.add_argument("--size", type=int, default=100)
    parser.add_argument("--datasets", default=None)
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--remove-mixed-backhaul", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-results", action=argparse.BooleanOptionalAction, default=True)

    warnings.filterwarnings("ignore", message=".*weights_only.*", category=FutureWarning)
    opts = parser.parse_args()

    device = torch.device("cuda:0" if "cuda" in opts.device and torch.cuda.is_available() else "cpu")

    # 1. 扫描所有可用数据集
    all_available_paths = []
    if opts.datasets is not None:
        all_available_paths = opts.datasets.split(",")
    else:
        for root, _, files in os.walk("data16"):
            for file in files:
                if "test" not in root: continue
                if file.endswith(".npz") and str(opts.size) in file:
                    if opts.remove_mixed_backhaul and "m" in root: continue
                    if file in ["50.npz", "100.npz"]:
                        all_available_paths.append(os.path.join(root, file))

    # 2. 按照用户指定的顺序进行重排
    target_order = [
        'CVRP', 'VRPTW', 'OVRP', 'VRPB', 'VRPL', 'OVRPB', 'OVRPL', 'OVRPTW',
        'VRPBL', 'OVRPBL', 'VRPBTW', 'OVRPBTW', 'VRPLTW', 'OVRPLTW', 'VRPBLTW', 'OVRPBLTW'
    ]

    data_paths = []
    for problem_type in target_order:
        for path in all_available_paths:
            # 路径通常为 data/cvrp/test/100.npz, split("/")[-3] 得到 'cvrp'
            path_parts = path.replace("\\", "/").split("/")
            if len(path_parts) >= 3:
                problem_name_in_path = path_parts[-3].upper()
                if problem_name_in_path == problem_type:
                    data_paths.append(path)
                    break  # 找到该类型的匹配路径后跳出，处理下一个类型

    assert len(data_paths) > 0, "No datasets found. Check data directory or target_order names."
    print(f"Total datasets to test: {len(data_paths)}")
    print(f"Order: {[p.replace('\\', '/').split('/')[-3].upper() for p in data_paths]}")

    # 3. 加载模型
    print("Loading checkpoint from ", opts.checkpoint)
    if "mvmoe" in opts.checkpoint.lower():
        BaseLitModule = MVMoE
    elif "mtpomo" in opts.checkpoint.lower():
        BaseLitModule = MTPOMO
    elif "moe" in opts.checkpoint.lower():
        BaseLitModule = RouteFinderMoE
    else:
        BaseLitModule = RouteFinderBase
    # 在 test.py 第 162 行之前
    import torch

    # ckpt = torch.load(opts.checkpoint, map_location="cpu")
    # print("Hyperparameters in checkpoint:", ckpt.get("hyper_parameters", {}).keys())
    model = BaseLitModule.load_from_checkpoint(opts.checkpoint, map_location="cpu", strict=False)
    env = MTVRPEnv()
    policy = model.policy.to(device).eval()

    # 4. 循环测试
    results = {}
    for dataset in tqdm(data_paths, desc="Evaluating Datasets"):
        dataset_name = dataset.replace("\\", "/").split("/")[-3].upper()
        # print(f"\n>>> Running evaluation for: {dataset_name}")

        td_test = env.load_data(dataset)
        dataloader = get_dataloader(td_test, batch_size=opts.batch_size)

        start = time.time()
        res = []
        for batch in dataloader:
            td_batch = env.reset(batch).to(device)
            o = test(policy, td_batch, env, device=device)
            res.append({
                "max_aug_reward": o["max_aug_reward"].cpu(),
                "gap_to_bks": o["gap_to_bks"].cpu() if "gap_to_bks" in o else None
            })

        inference_time = time.time() - start

        # 聚合结果
        all_rewards = torch.cat([r["max_aug_reward"] for r in res])
        avg_cost = -all_rewards.mean().item()

        # 尝试获取真实的 Gap
        avg_gap = 0
        # gaps = [r["gap_to_bks"] for r in res if r["gap_to_bks"] is not None]
        # if len(gaps) > 0:
        #     avg_gap = torch.cat(gaps).mean().item()

        print(f"Result: {dataset_name} | Cost: {avg_cost:.3f} | Gap: 0 | Time: {inference_time:.3f}s")

        results[dataset_name] = {
            "cost": avg_cost,
            # "gap": avg_gap,
            "inference_time": inference_time
        }

    # 5. 保存结果
    if opts.save_results:
        checkpoint_name = os.path.basename(opts.checkpoint).split(".")[0]
        savedir = f"results/main/{opts.size}/"
        os.makedirs(savedir, exist_ok=True)
        save_path = os.path.join(savedir, f"{checkpoint_name}.pkl")
        # with open(save_path, "wb") as f:
        #     pickle.dump(results, f)
        # print(f"\nResults saved to {save_path}")