import copy
import logging
import operator
import random

import numpy as np
import wandb
import torch

from fedml_api.standalone.fed_3.utils.client import Client
from fedml_api.utils.testInfo import TestInfo
from fedml_api.standalone.fed_3.utils.utils_func import *


class Fed3API(object):
    def __init__(self, device, args, dataset=None, model_trainer=None):
        self.device = device
        self.args = args
        logging.info("inside of fed_3 init, client num:" + str(self.args.client_num_in_total))

        self.client_list = []
        self.t_max = 0

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
        payment = []
        mx_utility = 0
        winners = []
        opt_time = 0
        critical_client = Client()
        for client in self.client_list:
            t_max = client.get_time()
            self.t_max = t_max
            logging.info("------set t_max:{}---------".format(self.t_max))

            temp_winners_utility = 0
            prob = np.random.random()
            logging.info("time:{}, prob:{}".format(self.t_max, prob))
            if prob <= 1.0 / 3.0:
                winner = 0
                mx_v = 0
                for client_i in self.client_list:
                    if client_i.get_time() > self.t_max:
                        continue
                    if client_i.get_training_intensity() > mx_v:
                        mx_v = client_i.get_training_intensity()
                        winner = client_i.client_idx
                        temp_winners_utility = 1.0 * client_i.get_training_intensity() / self.t_max
                temp_winners = [winner]
                temp_payment = [self.args.budget_per_round]
            else:
                temp_winners, critical_client = self._winners_determination()
                temp_winners_utility = self._get_utility(temp_winners)
                temp_payment = self._get_payment(temp_winners, critical_client)

            if temp_winners_utility > mx_utility:
                winners = temp_winners
                mx_utility = temp_winners_utility
                opt_time = self.t_max
                payment = temp_payment
        self.t_max = opt_time

        return winners, mx_utility, payment

    def test_properties(self):
        np.random.seed(self.args.seed)

        # bids init
        for client in self.client_list:
            client.update_bid(training_intensity=np.random.randint(5, 100), cost=np.random.random() * 5.0 + 2.0,
                              truth_ratio=1, computation_coefficient=np.random.rand() * 0.2,
                              communication_time=np.random.randint(10, 15))

        client_indexes, mn_cost, payment = self._get_winners()
        logging.info('winners:{}'.format(client_indexes))
        logging.info('payment:{}'.format(payment))

        tot_payment = 0
        for idx, client_idx in enumerate(client_indexes):
            client = self.client_list[int(client_idx)]
            # distribute payment
            client.receive_payment(payment[idx])
            tot_payment += payment[idx]

        t_idx = np.random.randint(0, len(client_indexes))
        client = self.client_list[client_indexes[t_idx]]
        real_cost = client.get_cost()
        client_payment = client.get_payment()

        return TestInfo(tot_payment=tot_payment, true_cost=real_cost, payment=client_payment)

    def train(self):
        np.random.seed(self.args.seed)
        w_global = self.model_trainer.get_model_params()

        accuracy_list = []
        loss_list = []
        time_list = []
        ti_sum_list = []
        round_list = []

        for round_idx in range(self.args.comm_round):
            logging.info("################Communication round : {}".format(round_idx))
            t_max = 0
            ti_sum = 0
            w_locals = []

            # bids init
            for client in self.client_list:
                client.update_bid(training_intensity=np.random.randint(5, 100), cost=np.random.random() * 5.0 + 2.0,
                                  truth_ratio=1, computation_coefficient=np.random.rand() * 0.2,
                                  communication_time=np.random.randint(10, 15))

            client_indexes, _, payment = self._get_winners()

            logging.info("client selected:{}".format(client_indexes))

            # train on winners
            for idx, client_idx in enumerate(client_indexes):
                # update dataset
                client = self.client_list[int(client_idx)]
                client.update_local_dataset(client_idx, self.train_data_local_dict[client_idx],
                                            self.test_data_local_dict[client_idx],
                                            self.train_data_local_num_dict[client_idx])
                # distribute payment
                client.receive_payment(payment[idx])
                t_max = max(t_max, client.get_time())
                ti_sum += client.get_training_intensity()

                # train on new dataset
                w = client.train(copy.deepcopy(w_global))
                # self.logger.info("local weights = " + str(w))
                w_locals.append((client.get_sample_number(), copy.deepcopy(w)))

            # update global weights
            w_global = self._aggregate(w_locals)
            self.model_trainer.set_model_params(w_global)

            # time, sum training intensity results
            time_list.append(t_max)
            ti_sum_list.append(ti_sum)

            # test results
            # at last round
            acc = 0
            loss = 0
            m_round_idx = round_idx
            if round_idx == self.args.comm_round - 1:
                acc, loss, m_round_idx = self._local_test_on_all_clients(round_idx)
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

    # used to test truthfulness
    def train_for_truthfulness(self, truth_ratio):
        np.random.seed(self.args.seed)

        # bids init
        for client in self.client_list:
            client.update_bid(training_intensity=np.random.randint(1, 5), cost=np.random.random() * 5 + 2,
                              truth_ratio=1, computation_coefficient=np.random.rand() * 0.2,
                              communication_time=np.random.randint(10, 15))

        # choose one bid in one particular round to test truthfulness
        truth_index = np.random.randint(0, len(self.client_list))
        self.client_list[truth_index].update_bidding_price_with_ratio(truth_ratio)
        logging.info(
            "truth_index" + str(truth_index) + ", true cost: " + str(
                self.client_list[truth_index].get_cost()) + ", bidding price: " + str(
                self.client_list[truth_index].get_bidding_price()) + ", time: " + str(
                self.client_list[truth_index].get_time()))

        client_indexes, mn_cost, payment = self._get_winners()
        logging.info('winners:{}'.format(client_indexes))
        logging.info('payment:{}'.format(payment))

        # train on winners
        for idx, client_idx in enumerate(client_indexes):
            if client_idx == truth_index:
                client = self.client_list[int(client_idx)]
                # distribute payment
                client.receive_payment(payment[idx])

        client_truth = self.client_list[truth_index]
        logging.info('id:{}, cost:{} bidding_price:{}, payment:{}, utility:{}'.format(truth_index,
                                                                                      client_truth.get_cost(),
                                                                                      client_truth.get_bidding_price(),
                                                                                      client_truth.get_payment(),
                                                                                      client_truth.get_utility()))
        # get utility for truthfulness test
        return self.client_list[truth_index].get_utility(), self.client_list[truth_index].get_bidding_price()

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
        # winners list is the list of selected clients
        winners_list = []
        # candidates' bid
        candidates = []
        for client in m_client_list:
            if client.get_time() <= self.t_max:
                candidates.append(client.bid)

        t_max = 0

        for bid in candidates:
            bid.update_avg_cost()

        # sort candidates according to average cost
        # argmax \tau / c
        cmp = operator.attrgetter('avg_cost')
        candidates.sort(key=cmp)

        while len(candidates):
            winner_idx = candidates[-1].client_idx
            winner_client = self.client_list[winner_idx]
            candidates.pop()
            # f(W \cup s_i)
            winners_client_ti = get_total_training_intensity(winners_list + [winner_client])
            budget_limit = self.args.budget_per_round * winner_client.get_training_intensity() / winners_client_ti
            if winner_client.get_bidding_price() > budget_limit:
                break
            winners_indexes.append(winner_idx)
            winners_list.append(winner_client)
            t_max = max(t_max, winner_client.get_time())

        critical_client = None
        if len(candidates):
            critical_idx = candidates[-1].client_idx
            critical_client = self.client_list[critical_idx]
        return winners_indexes, critical_client

    def _get_payment(self, winners_index, critical_client):
        '''
        :param winners_index: List(int)
        :param critical_client: Client, k+1 th client.
        :return:
        '''
        payment = np.zeros(len(winners_index))
        tot_training_intensity = 0
        # compute total trianing intensity
        for client_index in winners_index:
            client = self.client_list[client_index]
            tot_training_intensity += client.get_training_intensity()

        for i, client_i_index in enumerate(winners_index):
            # logging.info("getting payment for {}".format(client_i_index))
            client_i = self.client_list[client_i_index]
            payment_1 = self.args.budget_per_round * client_i.get_training_intensity() / tot_training_intensity
            if critical_client is None:
                payment[i] = payment_1
            else:
                payment_2 = client_i.get_training_intensity() * critical_client.get_bidding_price() / critical_client.get_training_intensity()
                payment[i] = min(payment_1, payment_2)
        # logging.info("payment list" + str(payment))
        return payment

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
        t_max = 0
        tot_training_intensity = 0
        for index in winners:
            client = self.client_list[index]
            t_max = max(t_max, client.get_time())
            tot_training_intensity += client.get_training_intensity()
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
