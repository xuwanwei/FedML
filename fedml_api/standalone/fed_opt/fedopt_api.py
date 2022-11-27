import copy
import logging
import pulp as pl

import numpy as np
import wandb
import torch

import random

from fedml_api.standalone.fed_opt.utils.client import Client


def _init_client_bid(client):
    client.update_bid(training_intensity=np.random.randint(1, 10), cost=np.random.randint(2, 10),
                      truth_ratio=1, computation_coefficient=np.random.rand() * 0.8,
                      communication_time=np.random.randint(5, 10))


class FedOptAPI(object):
    def __init__(self, device, args, dataset=None, model_trainer=None):
        self.device = device
        self.args = args
        logging.info("inside of fed_opt init, client num:" + str(self.args.client_num_in_total))

        self.client_list = []
        self.candidates = []
        self.candidate_selected = []
        self.t_max = 0
        self.mx_training_intensity = 0

        [train_data_num, test_data_num, train_data_global, test_data_global,
         train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num] = dataset
        self.train_global = train_data_global
        self.test_global = test_data_global
        self.val_global = None
        self.train_data_num_in_total = train_data_num
        self.test_data_num_in_total = test_data_num

        self.train_data_local_num_dict = train_data_local_num_dict
        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict

        self.model_trainer = model_trainer
        self._setup_clients(train_data_local_num_dict, train_data_local_dict, test_data_local_dict, model_trainer)

    def _setup_clients(self, train_data_local_num_dict, train_data_local_dict, test_data_local_dict, model_trainer):
        logging.info("############setup_clients (START)#############")
        logging.info("client_num_in_total:" + str(self.args.client_num_in_total))
        for client_idx in range(self.args.client_num_in_total):
            c = Client(client_idx=client_idx, args=self.args, device=self.device, communication_time=0,
                       computation_coefficient=0, cost=1, training_intensity=0,
                       local_training_data=train_data_local_dict[client_idx],
                       local_test_data=test_data_local_dict[client_idx],
                       local_sample_number=train_data_local_num_dict[client_idx],
                       model_trainer=model_trainer)
            self.client_list.append(c)
        logging.info("number of clients in client_list:" + str(len(self.client_list)))
        logging.info("############setup_clients (END)#############")

    def _get_winners(self):
        '''
        :return: winners: List(int), min_utility: int
        '''
        mx_utility = 0
        winners = []
        opt_time = 0
        for client in self.client_list:
            t_max = client.get_time()
            self.t_max = t_max
            logging.info("------set t_max:{}---------".format(self.t_max))

            # DFS
            # temp_winners = self._winners_determination_dfs()
            # LP
            temp_winners = self._winners_determination()
            temp_winners_utility = self._get_utility(temp_winners)

            if temp_winners_utility > mx_utility:
                winners = copy.deepcopy(temp_winners)
                mx_utility = temp_winners_utility
                opt_time = self.t_max
        self.t_max = opt_time

        return winners, mx_utility

    def train(self):
        w_global = self.model_trainer.get_model_params()

        accuracy_list = []
        loss_list = []
        time_list = []
        ti_sum_list = []
        round_list = []

        for round_idx in range(self.args.comm_round):
            np.random.seed(self.args.seed * round_idx)
            logging.info("################Communication round : {}".format(round_idx))
            w_locals = []
            t_max = 0
            ti_sum = 0

            # bids init
            for client in self.client_list:
                _init_client_bid(client)

            client_indexes, _ = self._get_winners()
            logging.info("train: client selected:{}".format(client_indexes))

            if len(client_indexes) == 0:
                continue

            # train on winners
            for idx, client_idx in enumerate(client_indexes):
                # update dataset
                client = self.client_list[int(client_idx)]
                client.update_local_dataset(client_idx, self.train_data_local_dict[client_idx],
                                            self.test_data_local_dict[client_idx],
                                            self.train_data_local_num_dict[client_idx])

                # train on new dataset
                w = client.train(copy.deepcopy(w_global))
                w_locals.append((client.get_sample_number(), copy.deepcopy(w)))
                t_max = max(t_max, client.get_time())
                ti_sum += client.get_training_intensity()

                # update global weights
            w_global = self._aggregate(w_locals)
            self.model_trainer.set_model_params(w_global)

            # time
            time_list.append(t_max)
            ti_sum_list.append(ti_sum)
            logging.info("train: ti sum:{}".format(ti_sum))

            # test results
            # at last round
            if round_idx == self.args.comm_round - 1:
                acc, loss, m_round_idx = self._local_test_on_all_clients(round_idx)
                accuracy_list.append(acc)
                loss_list.append(loss)
                round_list.append(m_round_idx)
            # per {frequency_of_the_test} round
            elif round_idx % self.args.frequency_of_the_test == 0:
                if self.args.dataset.startswith("stackoverflow"):
                    acc, loss, m_round_idx = self._local_test_on_validation_set(round_idx)
                else:
                    acc, loss, m_round_idx = self._local_test_on_all_clients(round_idx)
                accuracy_list.append(acc)
                loss_list.append(loss)
                round_list.append(m_round_idx)
        return accuracy_list, loss_list, time_list, ti_sum_list, round_list

    def _dfs(self, candidate_selected, bid_idx, sum_ti, sum_p):
        if bid_idx >= len(self.candidates):
            return
        bid = self.candidates[bid_idx]
        self._dfs(candidate_selected, bid_idx + 1, sum_ti, sum_p)
        if sum_p + bid.get_bidding_price() < self.args.budget_per_round:
            candidate_selected[bid_idx] = 1
            if sum_ti + bid.get_training_intensity() > self.mx_training_intensity:
                self.mx_training_intensity = sum_ti + bid.get_training_intensity()
                self.candidate_selected = copy.deepcopy(candidate_selected)
            self._dfs(candidate_selected, bid_idx + 1, sum_ti + bid.get_training_intensity(),
                      sum_p + bid.get_bidding_price())
            candidate_selected[bid_idx] = 0

    def _winners_determination_dfs(self, m_client_list=None):
        """
        :param T_max: int
        :param m_client_list: List(Client)
        :return winners_index: List(int)
        """
        if m_client_list is None:
            m_client_list = self.client_list

        # winners index is the index in the list of self.client_list
        winners_indexes = []
        # candidates' bid
        candidates = []
        for client in m_client_list:
            if client.get_time() <= self.t_max:
                candidates.append(client.bid)
        self.candidates = candidates

        # DFS
        candidates_selected = np.zeros(len(self.candidates))
        self.candidate_selected = np.zeros(len(self.candidates))
        self.mx_training_intensity = 0
        self.t_max = 0

        self._dfs(candidates_selected, 0, 0, 0)
        for bid_idx, bid_val in enumerate(self.candidate_selected):
            if bid_val == 1:
                winners_indexes.append(self.candidates[bid_idx].client_idx)
        logging.info("DFS: winners:{}".format(winners_indexes))

        return winners_indexes

    def _winners_determination(self, m_client_list=None):
        """
        :param T_max: int
        :param m_client_list: List(Client)
        :return winners_index: List(int)
        """
        if m_client_list is None:
            m_client_list = self.client_list

        # winners index is the index in the list of self.client_list
        winners_indexes = []
        # candidates' bid
        candidates = []
        for client in m_client_list:
            if client.get_time() <= self.t_max:
                candidates.append(client.bid)

        self.t_max = 0

        # LP
        model = pl.LpProblem(name="LP_winners", sense=pl.LpMaximize)

        x = [pl.LpVariable(name=f"x{i}", lowBound=0, upBound=1, cat=pl.LpInteger) for i in range(0, len(candidates))]
        # training intensity list
        ti_a = [bid.get_training_intensity() for bid in candidates]
        # payment list, bidding price list
        payment_a = [bid.get_bidding_price() for bid in candidates]
        # model goal
        model += pl.lpDot(ti_a, x)
        # constraint
        model += (pl.lpDot(payment_a, x) <= self.args.budget_per_round)
        model.solve()
        print("budget:{}".format(self.args.budget_per_round))
        for idx, var in enumerate(model.variables()):
            print("idx:{}, client_idx:{}, var:{}, b:{}, b_a:{}, ti:{}, ti_a:{}".format(idx, candidates[idx].client_idx,
                                                                                       var.value(),
                                                                                       candidates[idx].get_bidding_price(),
                                                                                       payment_a[idx],
                                                                                       candidates[
                                                                                           idx].get_training_intensity(),
                                                                                       ti_a[idx]))
            if var.value() == 1:
                winners_indexes.append(candidates[idx].client_idx)

        logging.info("LP winners:{}".format(winners_indexes))
        logging.info("LP ti sum:{}".format(model.objective.value()))
        for name, constraint in model.constraints.items():
            print(f"{name}: {constraint.value()}")

        return winners_indexes

    def _get_cost(self, winners):
        '''
        :param winners: List(int)
        :return: int
        '''
        t_max = 0
        tot_training_intensity = 0
        for index in winners:
            client = self.client_list[index]
            t_max = max(t_max, client.get_time())
            tot_training_intensity += client.get_training_intensity()
        return t_max - tot_training_intensity

    def _get_utility(self, winners):
        '''
        :param winners: List(int)
        :return: int
        '''
        logging.info("getting utility for {}".format(winners))
        if len(winners) == 0:
            return 0
        t_max = 0
        tot_training_intensity = 0
        for index in winners:
            client = self.client_list[index]
            t_max = max(t_max, client.get_time())
            tot_training_intensity += client.get_training_intensity()
            logging.info("index:{}, ti:{}".format(index, client.get_training_intensity()))
        if len(winners) == 0:
            logging.info("utility: 0")
            return 0
        logging.info("ti sum:{}, t_max:{}, utility:{}".format(tot_training_intensity, t_max,
                                                              1.0 * tot_training_intensity / t_max))
        return 1.0 * tot_training_intensity / t_max

    def _aggregate(self, w_locals):
        training_num = 0
        for idx in range(len(w_locals)):
            (sample_num, averaged_params) = w_locals[idx]
            training_num += sample_num

        (sample_num, averaged_params) = w_locals[0]
        for k in averaged_params.keys():
            for i in range(0, len(w_locals)):
                local_sample_number, local_model_params = w_locals[i]
                w = local_sample_number / training_num
                if i == 0:
                    averaged_params[k] = local_model_params[k] * w
                else:
                    averaged_params[k] += local_model_params[k] * w
        return averaged_params

    def _generate_validation_set(self, num_samples=10000):
        test_data_num = len(self.test_global.dataset)
        sample_indices = random.sample(range(test_data_num), min(num_samples, test_data_num))
        subset = torch.utils.data.Subset(self.test_global.dataset, sample_indices)
        sample_testset = torch.utils.data.DataLoader(subset, batch_size=self.args.batch_size)
        self.val_global = sample_testset

    def _local_test_on_all_clients(self, round_idx):

        logging.info("################local_test_on_all_clients : {}".format(round_idx))

        train_metrics = {
            'num_samples': [],
            'num_correct': [],
            'losses': []
        }

        test_metrics = {
            'num_samples': [],
            'num_correct': [],
            'losses': []
        }

        client = self.client_list[0]

        for client_idx in range(self.args.client_num_in_total):
            """
            Note: for datasets like "fed_CIFAR100" and "fed_shakespheare",
            the training client number is larger than the testing client number
            """
            if self.test_data_local_dict[client_idx] is None:
                continue
            client.update_local_dataset(0, self.train_data_local_dict[client_idx],
                                        self.test_data_local_dict[client_idx],
                                        self.train_data_local_num_dict[client_idx])
            # train data
            train_local_metrics = client.local_test(False)
            train_metrics['num_samples'].append(copy.deepcopy(train_local_metrics['test_total']))
            train_metrics['num_correct'].append(copy.deepcopy(train_local_metrics['test_correct']))
            train_metrics['losses'].append(copy.deepcopy(train_local_metrics['test_loss']))

            # test data
            test_local_metrics = client.local_test(True)
            test_metrics['num_samples'].append(copy.deepcopy(test_local_metrics['test_total']))
            test_metrics['num_correct'].append(copy.deepcopy(test_local_metrics['test_correct']))
            test_metrics['losses'].append(copy.deepcopy(test_local_metrics['test_loss']))

            """
            Note: CI environment is CPU-based computing. 
            The training speed for RNN training is to slow in this setting, so we only test a client to make sure there is no programming error.
            """
            if self.args.ci == 1:
                break

        # test on training dataset
        train_acc = sum(train_metrics['num_correct']) / sum(train_metrics['num_samples'])
        train_loss = sum(train_metrics['losses']) / sum(train_metrics['num_samples'])

        # test on test dataset
        test_acc = sum(test_metrics['num_correct']) / sum(test_metrics['num_samples'])
        test_loss = sum(test_metrics['losses']) / sum(test_metrics['num_samples'])

        stats = {'training_acc': train_acc, 'training_loss': train_loss}
        wandb.log({"Train/Acc": train_acc, "round": round_idx})
        wandb.log({"Train/Loss": train_loss, "round": round_idx})
        logging.info(stats)

        stats = {'test_acc': test_acc, 'test_loss': test_loss}
        wandb.log({"Test/Acc": test_acc, "round": round_idx})
        wandb.log({"Test/Loss": test_loss, "round": round_idx})
        logging.info(stats)

        return test_acc, test_loss, round_idx

    def _local_test_on_validation_set(self, round_idx):

        logging.info("################local_test_on_validation_set : {}".format(round_idx))

        if self.val_global is None:
            self._generate_validation_set()

        client = self.client_list[0]
        client.update_local_dataset(0, None, self.val_global, None)
        # test data
        test_metrics = client.local_test(True)

        if self.args.dataset == "stackoverflow_nwp":
            test_acc = test_metrics['test_correct'] / test_metrics['test_total']
            test_loss = test_metrics['test_loss'] / test_metrics['test_total']
            stats = {'test_acc': test_acc, 'test_loss': test_loss}
            wandb.log({"Test/Acc": test_acc, "round": round_idx})
            wandb.log({"Test/Loss": test_loss, "round": round_idx})
        elif self.args.dataset == "stackoverflow_lr":
            test_acc = test_metrics['test_correct'] / test_metrics['test_total']
            test_pre = test_metrics['test_precision'] / test_metrics['test_total']
            test_rec = test_metrics['test_recall'] / test_metrics['test_total']
            test_loss = test_metrics['test_loss'] / test_metrics['test_total']
            stats = {'test_acc': test_acc, 'test_pre': test_pre, 'test_rec': test_rec, 'test_loss': test_loss}
            wandb.log({"Test/Acc": test_acc, "round": round_idx})
            wandb.log({"Test/Pre": test_pre, "round": round_idx})
            wandb.log({"Test/Rec": test_rec, "round": round_idx})
            wandb.log({"Test/Loss": test_loss, "round": round_idx})
        else:
            raise Exception("Unknown format to log metrics for dataset {}!" % self.args.dataset)

        logging.info(stats)

        return test_acc, test_loss, round_idx
