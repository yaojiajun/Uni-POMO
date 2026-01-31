import torch

import os
from logging import getLogger

from VRPEnv import VRPEnv as Env
from VRPModel import VRPModel as Model

from utils.utils import *


class VRPTester:
    def __init__(self,
                 env_params,
                 model_params,
                 tester_params):

        # save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.tester_params = tester_params

        # result folder, logger
        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()

        # cuda
        USE_CUDA = self.tester_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.tester_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')
        self.device = device

        # ENV and MODEL
        self.env = Env(**self.env_params)
        self.model = Model(**self.model_params)

        # Restore
        model_load = tester_params['model_load']
        checkpoint_fullname = '{path}/{epoch}.pt'.format(**model_load)
        checkpoint = torch.load(checkpoint_fullname, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # utility
        self.time_estimator = TimeEstimator()

    def run(self):
        self.time_estimator.reset()

        score_AM = AverageMeter()
        aug_score_AM = AverageMeter()

        if self.tester_params['test_data_load']['enable']:
            self.env.use_saved_problems(self.tester_params['test_data_load']['filename'], self.device)

        test_num_episode = self.tester_params['test_episodes']
        episode = 0

        while episode < test_num_episode:

            remaining = test_num_episode - episode
            batch_size = min(self.tester_params['test_batch_size'], remaining)

            score, aug_score = self._test_one_batch(batch_size)

            score_AM.update(score, batch_size)
            aug_score_AM.update(aug_score, batch_size)

            episode += batch_size

            ############################
            # Logs
            ############################
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(episode, test_num_episode)
            self.logger.info("episode {:3d}/{:3d}, Elapsed[{}], Remain[{}], score:{:.3f}, aug_score:{:.3f}".format(
                episode, test_num_episode, elapsed_time_str, remain_time_str, score, aug_score))

            all_done = (episode == test_num_episode)

            if all_done:
                self.logger.info(" *** Test Done *** ")
                self.logger.info(" NO-AUG SCORE: {:.4f} ".format(score_AM.avg))
                self.logger.info(" AUGMENTATION SCORE: {:.4f} ".format(aug_score_AM.avg))
        return aug_score_AM.avg

    def _test_one_batch(self, batch_size):

        # Augmentation
        ###############################################
        if self.tester_params['augmentation_enable']:
            aug_factor = self.tester_params['aug_factor']
        else:
            aug_factor = 1

        # Ready
        ###############################################
        self.model.eval()
        with torch.no_grad():
            self.env.load_problems(batch_size, aug_factor)
            reset_state, _, _ = self.env.reset()
            self.model.pre_forward(reset_state)

        # POMO Rollout
        ###############################################
        state, reward, done = self.env.pre_step()
        while not done:
            selected, _ = self.model(state)
            # shape: (batch, pomo)
            state, reward, done, actions = self.env.step(selected)

        # Output Result
        ###############################################
        aug_reward = reward.reshape(aug_factor, batch_size, self.env.pomo_size)
        # shape: (aug_factor, batch, pomo_size)

        # Step 1: Find the best reward among the pomo_size starting points
        max_pomo_reward, pomo_idx = aug_reward.max(dim=2)
        # max_pomo_reward: (aug_factor, batch), pomo_idx: (aug_factor, batch)

        # Step 2: Find the best reward among the augmentation dimensions
        max_aug_pomo_reward, aug_idx = max_pomo_reward.max(dim=0)
        # max_aug_pomo_reward: (batch,), aug_idx: (batch,)

        # 2. Actions Extraction Logic
        # Assuming actions dimension is (aug_factor, pomo_size, seq_len)
        max_pomo_reward, pomo_idx = aug_reward.max(dim=2)

        # aug_idx shape: (batch,) -> index of the best augmentation
        max_aug_pomo_reward, aug_idx = max_pomo_reward.max(dim=0)

        # 2. Extract Best Actions
        # Indexing actions with shape (aug_factor, pomo_size, seq_len)
        seq_len = actions.shape[2]

        # Step A: Extract the best path under each of the 8 augmentations (Extract from dim=1)
        # Expand pomo_idx from (aug_factor, batch) to (aug_factor, batch, seq_len)
        index_pomo = pomo_idx.unsqueeze(-1).expand(-1, -1, seq_len)
        best_pomo_actions = torch.gather(actions, dim=1, index=index_pomo)
        # Resulting shape: (aug_factor, batch, seq_len)

        # Step B: Extract the globally best path among augmentations (Extract from dim=0)
        index_aug = aug_idx.view(1, -1, 1).expand(-1, -1, seq_len)
        final_best_actions = torch.gather(best_pomo_actions, dim=0, index=index_aug).squeeze()

        # 3. Print final_best_actions only (Converted to list to ensure full print)
        print(final_best_actions.cpu().tolist())

        # -----------------------------------------------
        # Original Return Logic
        no_aug_score = -max_pomo_reward[0, :].float().mean()
        aug_score = -max_aug_pomo_reward.float().mean()

        return no_aug_score.item(), aug_score.item()

    def _plot_TSP(self, nodesCoordinate):

        # print("nodesCoordinate = ",nodesCoordinate)

        plt.plot(nodesCoordinate[:, 0], nodesCoordinate[:, 1])
        plt.show()

    def _plot_CVRP(self, nodesCoordinate, depotCoordinate, demands, result):

        # print("nodesCoordinate = ",nodesCoordinate)

        plt.scatter(depotCoordinate[0], depotCoordinate[1], marker='*', s=160, c='r')

        for i in range(len(result) - 1):
            if (self.env.problem_type == "OVRP" or self.env.problem_type == "OVRPTW"):
                xlist = nodesCoordinate[int(result[i] - 1):int(result[i + 1] - 1), 0]
                ylist = nodesCoordinate[int(result[i] - 1):int(result[i + 1] - 1), 1]
                demandlist = demands[int(result[i] - 1):int(result[i + 1] - 1)]

            else:
                xlist = nodesCoordinate[int(result[i] - 1):int(result[i + 1]), 0]
                ylist = nodesCoordinate[int(result[i] - 1):int(result[i + 1]), 1]
                demandlist = demands[int(result[i] - 1):int(result[i + 1])]

            plt.plot(xlist, ylist, marker='o')

        if (self.env.problem_type == "OVRP" or self.env.problem_type == "OVRPTW"):
            xlist = nodesCoordinate[int(result[len(result) - 1] - 1):self.env.problem_size - 1, 0]
            ylist = nodesCoordinate[int(result[len(result) - 1] - 1):self.env.problem_size - 1, 1]
        else:
            xlist = nodesCoordinate[int(result[len(result) - 1] - 1):self.env.problem_size - 1, 0]
            ylist = nodesCoordinate[int(result[len(result) - 1] - 1):self.env.problem_size - 1, 1]

        plt.plot(xlist, ylist, marker='d')
        plt.axis('off')
        plt.show()