import copy
import logging
import operator
import random

import numpy as np
import torch
import wandb

from fedml_api.utils.client import Client
from fedml_api.utils.testInfo import TestInfo


class FedTiAPI(object):
    def __init__(self, dataset, device, args, model_trainer):
        self.device = device
        self.args = args
        [train_data_num, test_data_num, train_data_global, test_data_global,
         train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num] = dataset
        logging.info("inside of fedti init, client num:" + str(self.args.client_num_in_total))
        self.train_global = train_data_global
        self.test_global = test_data_global
        self.val_global = None
        self.train_data_num_in_total = train_data_num
        self.test_data_num_in_total = test_data_num

        self.client_list = []
        self.train_data_local_num_dict = train_data_local_num_dict
        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict

        self.model_trainer = model_trainer
        self._setup_clients(train_data_local_num_dict, train_data_local_dict, test_data_local_dict, model_trainer)

    def _setup_clients(self, train_data_local_num_dict, train_data_local_dict, test_data_local_dict, model_trainer):
        logging.info("############setup_clients (START)#############")
        logging.info("client_num_in_total:" + str(self.args.client_num_in_total))
        for client_idx in range(self.args.client_num_in_total):
            c = Client(client_idx, train_data_local_dict[client_idx], test_data_local_dict[client_idx],
                       train_data_local_num_dict[client_idx], self.args, self.device, model_trainer, 0, 0, 0, 0)
            self.client_list.append(c)
        logging.info("number of clients in client_list:" + str(len(self.client_list)))
        logging.info("type of item in client_list:" + str(type(self.client_list[0])))
        logging.info("############setup_clients (END)#############")

    def train(self, show_info):
        return self.train_for_truthfulness(truth_ratio=1, truth_index=0, show_info=show_info, test_truthfulness=False)

    # used to test truthfulness
    def train_for_truthfulness(self, truth_ratio, truth_index, show_info, test_truthfulness):
        w_global = self.model_trainer.get_model_params()
        np.random.seed(self.args.comm_round)

        payment_list = []
        bidding_price_list = []
        running_time_list = []
        client_utility_list = []
        social_cost_list = []

        for round_idx in range(self.args.comm_round):

            logging.info("################Communication round : {}".format(round_idx))

            w_locals = []

            """
            for scalability: following the original FedAvg algorithm, we uniformly sample a fraction of clients in each round.
            Instead of changing the 'Client' instances, our implementation keeps the 'Client' instances and then updates their local dataset 
            """

            # bids init
            for client in self.client_list:
                client.update_bid(training_intensity=np.random.randint(50, 100), cost=np.random.randint(2, 5),
                                  truth_ratio=1, computation_coefficient=np.random.rand() * 0.2,
                                  communication_time=np.random.randint(10, 15))

            # choose one bid in one particular round to test truthfulness
            if test_truthfulness:
                self.client_list[truth_index].update_bid(training_intensity=90, cost=2, truth_ratio=truth_ratio,
                                                         computation_coefficient=0.025, communication_time=10)

            # WDP and Payment
            # version 1
            # client_indexes, payment = self._winners_determination()
            # version 2
            # client_indexes, payment = self._winners_determination_2()
            # version 3
            client_indexes, payment = self._winners_determination_3()
            logging.info("winners_client_indexes = " + str(client_indexes))

            t_max = 0
            cost_tot = 0

            # train on winners
            for idx, client_idx in enumerate(client_indexes):
                client = self.client_list[int(client_idx)]
                client.update_local_dataset(client_idx, self.train_data_local_dict[client_idx],
                                            self.test_data_local_dict[client_idx],
                                            self.train_data_local_num_dict[client_idx])
                # train on new dataset
                w = client.train(copy.deepcopy(w_global))
                w_locals.append((client.get_sample_number(), copy.deepcopy(w)))
                t_max = max(t_max, client.get_time())
                cost_tot += client.get_cost()
                # distribute payment
                client.receive_payment(payment[idx])

            # update global weights
            w_global = self._aggregate(w_locals)
            self.model_trainer.set_model_params(w_global)

            running_time_list.append(t_max)
            social_cost_list.append(t_max + cost_tot)
            # get utility for truthfulness test
            if test_truthfulness:
                client_utility_list.append(self.client_list[truth_index].get_utility())

            # test results at last round
            if show_info:
                if round_idx == self.args.comm_round - 1:
                    self._local_test_on_all_clients(round_idx)
                # per {frequency_of_the_test} round
                elif round_idx % self.args.frequency_of_the_test == 0:
                    if self.args.dataset.startswith("stackoverflow"):
                        self._local_test_on_validation_set(round_idx)
                    else:
                        self._local_test_on_all_clients(round_idx)

                    # sample test IR
                    client_test_id = np.random.randint(0, len(client_indexes))
                    client_test_index = client_indexes[client_test_id]
                    payment_test = payment[client_test_id]
                    client_test = self.client_list[client_test_index]
                    # add to plot list
                    payment_list.append(payment_test * client_test.get_training_intensity())
                    bidding_price_list.append(client_test.get_bidding_price())

                    # wandb visualize
                    wandb.log({"number of winning clients": len(client_indexes)})
                    wandb.log({"running time in every round": t_max})

                    # plot IR chart
                    # wandb.log({"Performance on individual rationality": wandb.plot.line_series(
                    #     xs=[i for i in range(self.args.comm_round)],
                    #     ys=[[i for i in payment_list], [i for i in bidding_price_list]],
                    #     keys=['final_payment', 'bidding_price'],
                    #     title="Performance on individual rationality"
                    # )})
        return TestInfo(np.mean(running_time_list), np.mean(client_utility_list), np.mean(social_cost_list))

    def _generate_validation_set(self, num_samples=10000):
        test_data_num = len(self.test_global.dataset)
        sample_indices = random.sample(range(test_data_num), min(num_samples, test_data_num))
        subset = torch.utils.data.Subset(self.test_global.dataset, sample_indices)
        sample_testset = torch.utils.data.DataLoader(subset, batch_size=self.args.batch_size)
        self.val_global = sample_testset

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

    # version 1
    def _winners_determination(self):
        winners_indexes = []
        winners_payment = []
        candidates = []
        for client in self.client_list:
            candidates.append(client.bid)

        training_intensity_tot = 0
        # sort candidates according to the average cost
        cmp = operator.attrgetter('avg_cost')
        candidates.sort(key=cmp)
        candidate_idx = 0

        t_max = 0
        while training_intensity_tot < self.args.training_intensity_per_round:
            if candidate_idx + 1 >= len(candidates):
                break
            idx = candidates[candidate_idx].client_idx
            training_intensity_tot += self.client_list[idx].get_training_intensity()

            second_idx = candidates[candidate_idx + 1].client_idx
            payment = self._get_payment(idx, second_idx)

            candidate_idx += 1
            winners_indexes.append(idx)
            winners_payment.append(payment)
            t_max = max(t_max, self.client_list[idx].get_time())
            # logging.info('winner {} time: {}'.format(idx, self.client_list[idx].get_time()))

        logging.info("winners: " + str(winners_indexes))
        logging.info('winner t_max:{}'.format(t_max))
        return winners_indexes, winners_payment

    # version 2
    def _winners_determination_2(self):
        winners_indexes = []
        winners_payment = []
        candidates = []
        for client in self.client_list:
            candidates.append(client.bid)

        training_intensity_tot = 0

        t_max = 0

        while training_intensity_tot < self.args.training_intensity_per_round:
            if len(candidates) <= 1:
                logging.info("Not enough client to fulfill training intensity guarantee")
                break

            # update avg_cost
            # find the smallest two client
            winner_idx = candidates[0].client_idx
            winner_second_idx = -1
            candidates[0].update_average_cost_from_time(t_max)
            winner_cost = candidates[0].get_average_cost()
            winner_second_cost = 0
            for bid in candidates[1:]:
                bid.update_average_cost_from_time(t_max)
                if winner_cost > bid.get_average_cost():
                    winner_second_idx = winner_idx
                    winner_second_cost = winner_cost
                    winner_idx = bid.client_idx
                    winner_cost = bid.get_average_cost()
                elif winner_second_idx == -1:
                    winner_second_idx = bid.client_idx
                    winner_second_cost = bid.get_average_cost()
                elif winner_second_cost > bid.get_average_cost():
                    winner_second_cost = bid.get_average_cost()
                    winner_second_idx = bid.client_idx

            training_intensity_tot += self.client_list[winner_idx].get_training_intensity()

            payment = self._get_payment_2(winner_idx, winner_second_idx, t_max)

            winners_indexes.append(winner_idx)
            winners_payment.append(payment)
            candidates.remove(self.client_list[winner_idx].bid)
            t_max = max(t_max, self.client_list[winner_idx].get_time())
            # logging.info('winner {} time:{}'.format(winner_idx, self.client_list[winner_idx].get_time()))

        logging.info("winners: " + str(winners_indexes))
        logging.info('winner t_max:{}'.format(t_max))
        return winners_indexes, winners_payment

    # version 3
    def _winners_determination_3(self):
        winners_indexes = []
        winners_payment = []
        candidates = []
        for client in self.client_list:
            candidates.append(client.bid)

        training_intensity_tot = 0

        t_max = 0

        while training_intensity_tot < self.args.training_intensity_per_round:
            if len(candidates) <= 1:
                logging.info("Not enough client to fulfill training intensity guarantee")
                break

            # update avg_cost
            # find the smallest two client
            winner_idx = candidates[0].client_idx
            winner_second_idx = -1
            candidates[0].update_average_cost_from_time(t_max)
            winner_cost = candidates[0].get_average_cost()
            winner_second_cost = 0
            for bid in candidates[1:]:
                bid.update_average_cost_from_time(t_max)
                if winner_cost > bid.get_average_cost():
                    winner_second_idx = winner_idx
                    winner_second_cost = winner_cost
                    winner_idx = bid.client_idx
                    winner_cost = bid.get_average_cost()
                elif winner_second_idx == -1:
                    winner_second_idx = bid.client_idx
                    winner_second_cost = bid.get_average_cost()
                elif winner_second_cost > bid.get_average_cost():
                    winner_second_cost = bid.get_average_cost()
                    winner_second_idx = bid.client_idx

            training_intensity_tot += self.client_list[winner_idx].get_training_intensity()

            payment = self._get_payment_3(winner_idx, winner_second_idx, t_max)

            winners_indexes.append(winner_idx)
            winners_payment.append(payment)
            candidates.remove(self.client_list[winner_idx].bid)
            t_max = max(t_max, self.client_list[winner_idx].get_time())

        logging.info("winners: " + str(winners_indexes))
        logging.info('winner t_max:{}'.format(t_max))
        return winners_indexes, winners_payment

    def _get_payment(self, opt_index, second_index):
        client_second_winner = self.client_list[second_index]
        client_winner = self.client_list[opt_index]
        # version 1
        payment = client_second_winner.get_average_cost() - client_winner.get_time() / client_winner.get_training_intensity()
        return payment

    def _get_payment_2(self, opt_index, second_index, t_max):
        client_second_winner = self.client_list[second_index]
        client_winner = self.client_list[opt_index]
        # version 2
        r_t = max(0, client_winner.get_time() - t_max)
        payment = client_second_winner.get_average_cost() - r_t / client_winner.get_training_intensity()
        return payment

    def _get_payment_3(self, opt_index, second_index, t_max):
        client_second_winner = self.client_list[second_index]
        client_winner = self.client_list[opt_index]
        # version 2
        # r_t = max(0, client_winner.get_time() - t_max)
        # version 3
        r_t = max(client_winner.get_time(), t_max)
        payment = client_second_winner.get_average_cost() - r_t / client_winner.get_training_intensity()
        return payment
