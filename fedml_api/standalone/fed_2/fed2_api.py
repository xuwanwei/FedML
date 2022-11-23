import logging
import operator

import numpy as np
import wandb

from fedml_api.standalone.fed_2.utils.client import Client
from fedml_api.utils.testInfo import TestInfo
from fedml_api.standalone.fed_2.utils.utils_func import *


class Fed2API(object):
    def __init__(self, device, args):
        self.device = device
        self.args = args
        logging.info("inside of fed_2 init, client num:" + str(self.args.client_num_in_total))

        self.client_list = []
        self.t_max = 0

        self._setup_clients()

    def _setup_clients(self):
        logging.info("############setup_clients (START)#############")
        logging.info("client_num_in_total:" + str(self.args.client_num_in_total))
        for client_idx in range(self.args.client_num_in_total):
            c = Client(client_idx, self.args, self.device, 0, 0, 0, 0)
            self.client_list.append(c)
        logging.info("number of clients in client_list:" + str(len(self.client_list)))
        logging.info("############setup_clients (END)#############")

    def _train(self):
        '''
        :return: winners: List(int), min_utility: int
        '''
        min_utility = 0
        winners = []
        opt_time = 0
        for client in self.client_list:
            t_max = client.get_time()
            self.t_max = t_max
            logging.info("------set t_max:{}---------".format(self.t_max))
            temp_winners = self._winners_determination()
            temp_winners_costs = self._get_cost(temp_winners)
            if temp_winners_costs > min_utility or min_utility == -1:
                winners = temp_winners
                min_utility = temp_winners_costs
                opt_time = self.t_max
        self.t_max = opt_time
        return winners, min_utility

    def train(self):
        np.random.seed(self.args.seed)

        # bids init
        for client in self.client_list:
            client.update_bid(training_intensity=np.random.randint(50, 100), cost=np.random.random() * 3.0 + 2.0,
                              truth_ratio=1, computation_coefficient=np.random.rand() * 0.2,
                              communication_time=np.random.randint(10, 15))

        client_indexes, mn_cost = self._train()
        logging.info('winners:{}'.format(client_indexes))
        payment = self._get_payment(client_indexes)
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

    # used to test truthfulness
    def train_for_truthfulness(self, truth_ratio):
        np.random.seed(self.args.seed)

        # bids init
        for client in self.client_list:
            client.update_bid(training_intensity=np.random.randint(50, 100), cost=np.random.random() * 3.0 + 2.0,
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

        client_indexes, mn_cost = self._train()
        logging.info('winners:{}'.format(client_indexes))
        payment = self._get_payment(client_indexes)
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
        # argmax {delta f} / {c} = \tau / c
        cmp = operator.attrgetter('avg_cost')
        candidates.sort(key=cmp)

        # logging.info("candidates len:{}".format(len(candidates)))
        while len(candidates):
            # logging.info("last 5 of {} candidats:{}".format(len(candidates), [x.client_idx for x in candidates[-5:]]))
            winner_idx = candidates[-1].client_idx
            winner_client = self.client_list[winner_idx]
            candidates.pop()
            # f(W \cup s_i)
            winners_client_ti = get_total_training_intensity(winners_list + [winner_client])
            # B/2 * tau_i / f(W+[winner])
            # logging.info(
            #     'tau_i:{}, f(W+{}):{}'.format(winner_client.get_training_intensity(), winner_idx, winners_client_ti))
            # budget_limit = self.args.budget_per_round / 2.0 * winner_client.get_training_intensity() / winners_client_ti
            budget_limit = self.args.budget_per_round * winner_client.get_training_intensity() / winners_client_ti
            # logging.info('candidates num:{}, budget_limit:{}, bidding_price:{}'.format(len(candidates), budget_limit,
            #                                                                            winner_client.get_bidding_price()))
            if winner_client.get_bidding_price() > budget_limit:
                break
            winners_indexes.append(winner_idx)
            winners_list.append(winner_client)
            t_max = max(t_max, winner_client.get_time())

        # logging.info("winners: " + str(winners_indexes))
        # logging.info('winner t_max:{}'.format(t_max))
        return winners_indexes

    def _get_payment(self, winners_index):
        payment = np.zeros(len(winners_index))
        for i, client_i_index in enumerate(winners_index):
            logging.info("getting payment for {}".format(client_i_index))
            # client_list_exc_i : S_{-i}
            client_i = self.client_list[client_i_index]
            client_list_exc_i = self.client_list.copy()
            client_list_exc_i.remove(client_i)
            winner_exc_i = self._winners_determination(client_list_exc_i)
            logging.info("winners exc {}:{}".format(client_i_index, winner_exc_i))
            # logging.info(
            #     'tau_i:{}, tot_tau:{}, selected_client_num:{}'.format(client_i.get_training_intensity(),
            #                                                           get_total_training_intensity(
            #                                                               winner_truncated + [client_i]),
            #                                                           len(winner_truncated) + 1))
            max_p = 0
            for j, client_j in enumerate(winner_exc_i):
                client_j = self.client_list[client_j]
                set_truncated = winner_exc_i[:j]
                winner_truncated = get_client_list(set_truncated, self.client_list)
                payment_1 = client_i.get_training_intensity() * client_j.get_bidding_price() / client_j.get_training_intensity()
                # B/2 * tau_i/ tot_tau
                # payment_2 = self.args.budget_per_round / 2.0 * client_i.get_training_intensity() / get_total_training_intensity(
                #     winner_truncated + [client_i])
                payment_2 = self.args.budget_per_round * client_i.get_training_intensity() / get_total_training_intensity(
                    winner_truncated + [client_i])
                mn_payment = min(payment_1, payment_2)
                if mn_payment > max_p:
                    max_p = mn_payment
                    payment[i] = max_p
                # logging.info(
                #     "i:{}, client j:{}, p_1:{}, p_2:{}, mn:{}".format(client_i_index, client_j.client_idx, payment_1,
                #                                                       payment_2, mn_payment))
        logging.info("payment list" + str(payment))
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
