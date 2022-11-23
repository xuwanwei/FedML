import argparse
import csv
import logging
import os
import random
import sys
import time

import numpy as np
import torch
import wandb

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))
from fedml_api.standalone.fed_2.fed2_api import Fed2API
from fedml_api.utils.draw import draw_IC
from fedml_api.utils.draw import draw_individual_rationality
from fedml_api.utils.draw import draw_budget_balance
DATA_PATH = "../../../OutputData/fed_2"


def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    parser.add_argument('--model', type=str, default='resnet56', metavar='N',
                        help='neural network used in training')

    parser.add_argument('--dataset', type=str, default='cifar10', metavar='N',
                        help='dataset used for training')

    parser.add_argument('--epochs', type=int, default=5, metavar='EP',
                        help='how many epochs will be trained locally')

    parser.add_argument('--client_num_in_total', type=int, default=10, metavar='NN',
                        help='number of workers in a distributed cluster')

    parser.add_argument('--client_num_per_round', type=int, default=10, metavar='NN',
                        help='number of workers')

    parser.add_argument('--comm_round', type=int, default=10,
                        help='how many round of communications we shoud use')

    parser.add_argument('--frequency_of_the_test', type=int, default=5,
                        help='the frequency of the algorithms')

    parser.add_argument('--gpu', type=int, default=0,
                        help='gpu')

    parser.add_argument('--ci', type=int, default=0,
                        help='CI')

    parser.add_argument('--budget_per_round', type=float, default=70,
                        help='the budget of the server in a round')

    parser.add_argument('--seed', type=int, default=4, help='numpy random seed')

    parser.add_argument('--draw', type=bool, default=False, help='draw pic')
    return parser


def test_truthfulness(device, args):
    fed_2API = Fed2API(device, args)
    truth_ratio_list = []
    utility_list = []
    logging.info("####################Truthfulness#####################")
    for truth_ratio in np.arange(0.2, 2, 0.2):
        logging.info("Ratio:" + str(truth_ratio))
        client_utility, client_bidding_price = fed_2API.train_for_truthfulness(truth_ratio=truth_ratio)
        truth_ratio_list.append(truth_ratio)
        utility_list.append(client_utility)

    logging.info("####################End##############################")
    logging.info("utility list:" + str(utility_list))
    truth_data = [[round(x, 2), y] for (x, y) in zip(truth_ratio_list, utility_list)]
    truth_table = wandb.Table(data=truth_data, columns=["The ratio of the submitted bid to the truthful cost",
                                                        "The utility of a single buyers"])
    wandb.log(
        {"Performance on truthfulness": wandb.plot.line(truth_table,
                                                        "The ratio of the submitted bid to the truthful cost",
                                                        "The utility of a single buyers",
                                                        title="Performance on truthfulness")})

    timestamp = time.time()
    datatime = time.strftime("%Y-%m-%d-%H-%M", time.localtime(timestamp))
    file_name = 'fed2-{}-IC-{}'.format(args.seed, datatime)
    print("writing {}/{}.csv".format(DATA_PATH, file_name))

    with open('{}/{}.csv'.format(DATA_PATH, file_name), mode="w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(truth_data)

    if args.draw:
        draw_IC(file_name)


def test_one_round(device, args):
    args.client_num_in_total = 150
    fed_2API = Fed2API(device, args)
    res = fed_2API.train()
    logging.info("total payment:{}".format(res.tot_payment))


def test_budget_balance_with_client_num(device, args):
    tot_payment_list = []
    client_num_list = []
    budget_list = []
    logging.info("####################Budget Balance#####################")
    for client_num in np.arange(10, 200, 10):
        args.client_num_in_total = client_num
        fed_2API = Fed2API(device, args)
        res = fed_2API.train()
        tot_payment_list.append(res.tot_payment)
        budget_list.append(args.budget_per_round)
        client_num_list.append(client_num)

    logging.info("####################End##############################")
    truth_data = [[x, y, z] for (x, y, z) in zip(client_num_list, tot_payment_list, budget_list)]

    timestamp = time.time()
    datatime = time.strftime("%Y-%m-%d-%H-%M", time.localtime(timestamp))
    file_name = 'fed2-{}-BB-{}'.format(args.seed, datatime)
    print("writing {}".format(file_name))
    with open('{}/{}.csv'.format(DATA_PATH, file_name), mode="w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(truth_data)

    if args.draw:
        draw_budget_balance(file_name)


def test_individual_rationality(device, args):
    payment_list = []
    true_cost_list = []
    client_num_list = []
    logging.info("####################IR#####################")
    for client_num in np.arange(10, 200, 10):
        args.client_num_in_total = client_num
        fed_2API = Fed2API(device, args)
        res = fed_2API.train()
        payment_list.append(res.payment)
        true_cost_list.append(res.true_cost)
        client_num_list.append(client_num)

    logging.info("####################End##############################")
    truth_data = [[x, y, z] for (x, y, z) in zip(client_num_list, true_cost_list, payment_list)]

    timestamp = time.time()
    datatime = time.strftime("%Y-%m-%d-%H-%M", time.localtime(timestamp))
    file_name = 'fed2-{}-IR-{}'.format(args.seed, datatime)
    print("writing {}".format(file_name))
    with open('{}/{}.csv'.format(DATA_PATH, file_name), mode="w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(truth_data)

    if args.draw:
        draw_individual_rationality(file_name)


if __name__ == "__main__":
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    parser = add_args(argparse.ArgumentParser(description='fed_2-standalone'))
    args = parser.parse_args()
    logger.info(args)
    device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
    logger.info(device)

    wandb.init(
        project="fedml",
        name="Fed2-r" + str(args.comm_round) + "-e" + str(args.epochs),
        config=args
    )

    # Set the random seed. The np.random seed determines the dataset partition.
    # The torch_manual_seed determines the initial weight.
    # We fix these two, so that we can reproduce the result.
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True

    test_truthfulness(device, args)
    test_budget_balance_with_client_num(device, args)
    test_individual_rationality(device, args)

    # load data
    dataset = load_data(args, args.dataset)

    # create model.
    # Note if the model is DNN (e.g., ResNet), the training will be very slow.
    # In this case, please use our FedML distributed version (./fedml_experiments/distributed_fedavg)
    model = create_model(args, model_name=args.model, output_dim=dataset[7])
    model_trainer = custom_model_trainer(args, model)
    logging.info(model)

    test_one_round(device, args)

