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

DATA_PATH = "../../../OutputData/fed_opt"

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))

from fedml_api.data_preprocessing.cifar10.data_loader import load_partition_data_cifar10
from fedml_api.data_preprocessing.cifar100.data_loader import load_partition_data_cifar100
from fedml_api.data_preprocessing.cinic10.data_loader import load_partition_data_cinic10
from fedml_api.data_preprocessing.fed_cifar100.data_loader import load_partition_data_federated_cifar100
from fedml_api.data_preprocessing.shakespeare.data_loader import load_partition_data_shakespeare
from fedml_api.data_preprocessing.fed_shakespeare.data_loader import load_partition_data_federated_shakespeare
from fedml_api.data_preprocessing.stackoverflow_lr.data_loader import load_partition_data_federated_stackoverflow_lr
from fedml_api.data_preprocessing.stackoverflow_nwp.data_loader import load_partition_data_federated_stackoverflow_nwp
from fedml_api.data_preprocessing.ImageNet.data_loader import load_partition_data_ImageNet
from fedml_api.data_preprocessing.Landmarks.data_loader import load_partition_data_landmarks
from fedml_api.model.cv.mobilenet import mobilenet
from fedml_api.model.cv.resnet import resnet56
from fedml_api.model.cv.cnn import CNN_DropOut
from fedml_api.data_preprocessing.FederatedEMNIST.data_loader import load_partition_data_federated_emnist
from fedml_api.model.nlp.rnn import RNN_OriginalFedAvg, RNN_StackOverFlow

from fedml_api.data_preprocessing.MNIST.data_loader import load_partition_data_mnist
from fedml_api.model.linear.lr import LogisticRegression
from fedml_api.model.cv.resnet_gn import resnet18

from fedml_api.standalone.fed_3.my_model_trainer_classification import MyModelTrainer as MyModelTrainerCLS
from fedml_api.standalone.fed_3.my_model_trainer_nwp import MyModelTrainer as MyModelTrainerNWP
from fedml_api.standalone.fed_3.my_model_trainer_tag_prediction import MyModelTrainer as MyModelTrainerTAG

from fedml_api.standalone.fed_opt.fedopt_api import FedOptAPI
from fedml_api.utils.draw import draw_IC
from fedml_api.utils.draw import draw_individual_rationality
from fedml_api.utils.draw import draw_budget_balance
from fedml_api.utils.draw import draw_accuracy
from fedml_api.utils.draw import draw_loss
from fedml_api.utils.draw import draw_time


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

    parser.add_argument('--data_dir', type=str, default='./../../../data/cifar10',
                        help='data directory')

    parser.add_argument('--epochs', type=int, default=5, metavar='EP',
                        help='how many epochs will be trained locally')

    parser.add_argument('--client_num_in_total', type=int, default=100, metavar='NN',
                        help='number of workers in a distributed cluster')

    parser.add_argument('--partition_method', type=str, default='hetero', metavar='N',
                        help='how to partition the dataset on local workers')

    parser.add_argument('--partition_alpha', type=float, default=0.5, metavar='PA',
                        help='partition alpha (default: 0.5)')

    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')

    parser.add_argument('--client_optimizer', type=str, default='adam',
                        help='SGD with momentum; adam')

    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')

    parser.add_argument('--wd', help='weight decay parameter;', type=float, default=0.001)

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


def load_data(args, dataset_name):
    # check if the centralized training is enabled
    centralized = True if args.client_num_in_total == 1 else False

    # check if the full-batch training is enabled
    args_batch_size = args.batch_size
    if args.batch_size <= 0:
        full_batch = True
        args.batch_size = 128  # temporary batch size
    else:
        full_batch = False

    if dataset_name == "mnist":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        logging.info("inside of load data client num:{}".format(args.client_num_in_total))
        client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_mnist(args.batch_size, args.client_num_in_total)
        """
        For shallow NN or linear models, 
        we uniformly sample a fraction of clients each round (as the original FedAvg paper)
        """
        args.client_num_in_total = client_num

    elif dataset_name == "femnist":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_federated_emnist(args.dataset, args.data_dir)
        args.client_num_in_total = client_num

    elif dataset_name == "shakespeare":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_shakespeare(args.batch_size)
        args.client_num_in_total = client_num

    elif dataset_name == "fed_shakespeare":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_federated_shakespeare(args.dataset, args.data_dir)
        args.client_num_in_total = client_num

    elif dataset_name == "fed_cifar100":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_federated_cifar100(args.dataset, args.data_dir)
        args.client_num_in_total = client_num
    elif dataset_name == "stackoverflow_lr":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_federated_stackoverflow_lr(args.dataset, args.data_dir)
        args.client_num_in_total = client_num
    elif dataset_name == "stackoverflow_nwp":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_federated_stackoverflow_nwp(args.dataset, args.data_dir)
        args.client_num_in_total = client_num

    elif dataset_name == "ILSVRC2012":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_ImageNet(dataset=dataset_name, data_dir=args.data_dir,
                                                 partition_method=None, partition_alpha=None,
                                                 client_number=args.client_num_in_total, batch_size=args.batch_size)

    elif dataset_name == "gld23k":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        args.client_num_in_total = 233
        fed_train_map_file = os.path.join(args.data_dir, 'mini_gld_train_split.csv')
        fed_test_map_file = os.path.join(args.data_dir, 'mini_gld_test.csv')

        train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_landmarks(dataset=dataset_name, data_dir=args.data_dir,
                                                  fed_train_map_file=fed_train_map_file,
                                                  fed_test_map_file=fed_test_map_file,
                                                  partition_method=None, partition_alpha=None,
                                                  client_number=args.client_num_in_total, batch_size=args.batch_size)

    elif dataset_name == "gld160k":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        args.client_num_in_total = 1262
        fed_train_map_file = os.path.join(args.data_dir, 'federated_train.csv')
        fed_test_map_file = os.path.join(args.data_dir, 'test.csv')

        train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_landmarks(dataset=dataset_name, data_dir=args.data_dir,
                                                  fed_train_map_file=fed_train_map_file,
                                                  fed_test_map_file=fed_test_map_file,
                                                  partition_method=None, partition_alpha=None,
                                                  client_number=args.client_num_in_total, batch_size=args.batch_size)

    else:
        if dataset_name == "cifar10":
            data_loader = load_partition_data_cifar10
        elif dataset_name == "cifar100":
            data_loader = load_partition_data_cifar100
        elif dataset_name == "cinic10":
            data_loader = load_partition_data_cinic10
        else:
            data_loader = load_partition_data_cifar10
        train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = data_loader(args.dataset, args.data_dir, args.partition_method,
                                args.partition_alpha, args.client_num_in_total, args.batch_size)

    if centralized:
        train_data_local_num_dict = {
            0: sum(user_train_data_num for user_train_data_num in train_data_local_num_dict.values())}
        train_data_local_dict = {
            0: [batch for cid in sorted(train_data_local_dict.keys()) for batch in train_data_local_dict[cid]]}
        test_data_local_dict = {
            0: [batch for cid in sorted(test_data_local_dict.keys()) for batch in test_data_local_dict[cid]]}
        args.client_num_in_total = 1

    if full_batch:
        train_data_global = combine_batches(train_data_global)
        test_data_global = combine_batches(test_data_global)
        train_data_local_dict = {cid: combine_batches(train_data_local_dict[cid]) for cid in
                                 train_data_local_dict.keys()}
        test_data_local_dict = {cid: combine_batches(test_data_local_dict[cid]) for cid in test_data_local_dict.keys()}
        args.batch_size = args_batch_size

    dataset = [train_data_num, test_data_num, train_data_global, test_data_global,
               train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num]
    return dataset


def combine_batches(batches):
    full_x = torch.from_numpy(np.asarray([])).float()
    full_y = torch.from_numpy(np.asarray([])).long()
    for (batched_x, batched_y) in batches:
        full_x = torch.cat((full_x, batched_x), 0)
        full_y = torch.cat((full_y, batched_y), 0)
    return [(full_x, full_y)]


def create_model(args, model_name, output_dim):
    logging.info("create_model. model_name = %s, output_dim = %s" % (model_name, output_dim))
    model = None
    if model_name == "lr" and args.dataset == "mnist":
        logging.info("LogisticRegression + MNIST")
        model = LogisticRegression(28 * 28, output_dim)
    elif model_name == "cnn" and args.dataset == "femnist":
        logging.info("CNN + FederatedEMNIST")
        model = CNN_DropOut(False)
    elif model_name == "resnet18_gn" and args.dataset == "fed_cifar100":
        logging.info("ResNet18_GN + Federated_CIFAR100")
        model = resnet18()
    elif model_name == "rnn" and args.dataset == "shakespeare":
        logging.info("RNN + shakespeare")
        model = RNN_OriginalFedAvg()
    elif model_name == "rnn" and args.dataset == "fed_shakespeare":
        logging.info("RNN + fed_shakespeare")
        model = RNN_OriginalFedAvg()
    elif model_name == "lr" and args.dataset == "stackoverflow_lr":
        logging.info("lr + stackoverflow_lr")
        model = LogisticRegression(10000, output_dim)
    elif model_name == "rnn" and args.dataset == "stackoverflow_nwp":
        logging.info("RNN + stackoverflow_nwp")
        model = RNN_StackOverFlow()
    elif model_name == "resnet56":
        model = resnet56(class_num=output_dim)
    elif model_name == "mobilenet":
        model = mobilenet(class_num=output_dim)
    return model


def custom_model_trainer(args, model):
    if args.dataset == "stackoverflow_lr":
        return MyModelTrainerTAG(model)
    elif args.dataset in ["fed_shakespeare", "stackoverflow_nwp"]:
        return MyModelTrainerNWP(model)
    else:  # default model trainer is for classification problem
        return MyModelTrainerCLS(model)


def test_truthfulness(device, args, dataset, model_trainer):
    fed_optAPI = FedOptAPI(device, args, dataset=dataset, model_trainer=model_trainer)
    truth_ratio_list = []
    utility_list = []
    logging.info("####################Truthfulness#####################")
    for truth_ratio in np.arange(0.2, 2, 0.2):
        logging.info("Ratio:" + str(truth_ratio))
        client_utility, client_bidding_price = fed_optAPI.train_for_truthfulness(truth_ratio=truth_ratio)
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
    file_name = 'fedopt-{}-IC-{}'.format(args.seed, datatime)
    print("writing {}".format(file_name))
    with open('{}/{}.csv'.format(DATA_PATH, file_name), mode="w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(truth_data)

    if args.draw:
        draw_IC(file_name)


def test_budget_balance_with_client_num(device, args, dataset, model_trainer):
    tot_payment_list = []
    client_num_list = []
    budget_list = []
    logging.info("####################Budget Balance#####################")
    for client_num in np.arange(10, 100, 10):
        args.client_num_in_total = client_num
        fed_optAPI = FedOptAPI(device, args, dataset, model_trainer)
        res = fed_optAPI.test_properties()
        tot_payment_list.append(res.tot_payment)
        budget_list.append(args.budget_per_round)
        client_num_list.append(client_num)

    logging.info("####################End##############################")
    truth_data = [[x, y, z] for (x, y, z) in zip(client_num_list, tot_payment_list, budget_list)]

    timestamp = time.time()
    datatime = time.strftime("%Y-%m-%d-%H-%M", time.localtime(timestamp))
    file_name = 'fedopt-{}-BB-{}'.format(args.seed, datatime)
    print("writing {}".format(file_name))
    with open('{}/{}.csv'.format(DATA_PATH, file_name), mode="w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(truth_data)

    if args.draw:
        draw_budget_balance(file_name)


def test_individual_rationality(device, args, dataset, model_trainer):
    payment_list = []
    true_cost_list = []
    client_num_list = []
    logging.info("####################IR#####################")
    for client_num in np.arange(10, 100, 10):
        args.client_num_in_total = client_num
        fed_optAPI = FedOptAPI(device, args, dataset, model_trainer)
        res = fed_optAPI.test_properties()
        payment_list.append(res.payment)
        true_cost_list.append(res.true_cost)
        client_num_list.append(client_num)

    logging.info("####################End##############################")
    truth_data = [[x, y, z] for (x, y, z) in zip(client_num_list, true_cost_list, payment_list)]

    timestamp = time.time()
    datatime = time.strftime("%Y-%m-%d-%H-%M", time.localtime(timestamp))
    file_name = 'fedopt-{}-IR-{}'.format(args.seed, datatime)
    print("writing {}".format(file_name))
    with open('{}/{}.csv'.format(DATA_PATH, file_name), mode="w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(truth_data)

    if args.draw:
        draw_individual_rationality(file_name)


def test_with_rounds(dataset, device, args, model_trainer):
    fed_optAPI = FedOptAPI(device=device, args=args, dataset=dataset, model_trainer=model_trainer)
    acc_list, loss_list, time_list, ti_sum_list, round_list = fed_optAPI.train()
    goal_list = []
    for idx, ti_val in enumerate(ti_sum_list):
        goal_list.append(float(ti_val) / float(time_list[idx]))

    data_table = [[r, acc, loss, t, ti_sum, goal] for (r, acc, loss, t, ti_sum, goal) in
                  zip(round_list, acc_list, loss_list, time_list, ti_sum_list, goal_list)]

    # writing data to file
    timestamp = time.time()
    datatime = time.strftime("%Y-%m-%d-%H-%M", time.localtime(timestamp))
    file_name = 'fedopt-{}-INFO-{}'.format(args.seed, datatime)
    print("writing {}".format(file_name))
    with open('{}/{}.csv'.format(DATA_PATH, file_name), mode="w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(data_table)

    if args.draw:
        draw_accuracy(file_name)
        draw_loss(file_name)
        draw_time(file_name)


if __name__ == "__main__":
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    parser = add_args(argparse.ArgumentParser(description='fed_opt-standalone'))
    args = parser.parse_args()
    logger.info(args)
    device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
    logger.info(device)

    wandb.init(
        project="fedml",
        name="FedOpt-r" + str(args.comm_round) + "-e" + str(args.epochs),
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

    # load data
    dataset = load_data(args, args.dataset)
    logging.info("after load data:{}".format(args.client_num_in_total))

    # create model.
    # Note if the model is DNN (e.g., ResNet), the training will be very slow.
    # In this case, please use our FedML distributed version (./fedml_experiments/distributed_fedavg)
    model = create_model(args, model_name=args.model, output_dim=dataset[7])
    model_trainer = custom_model_trainer(args, model)
    logging.info(model)

    # Test economic properties
    # test_truthfulness(device, args, dataset, model_trainer)
    # test_budget_balance_with_client_num(device, args, dataset, model_trainer)
    # test_individual_rationality(device, args, dataset, model_trainer)
    # test_with_budget(device, args)

    # Test Accuracy and Time
    test_with_rounds(dataset, device, args, model_trainer)
