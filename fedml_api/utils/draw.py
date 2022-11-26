import csv
import logging
import time

from matplotlib import pyplot as plt

DATA_PATH_PRE = "../../../OutputData"
IMG_PATH_PRE = "../../../OutputImage"

DATA_PATH = "../../../OutputData/fed_opt"
IMG_PATH = "../../../OutputImage/fed_opt"

in_filename = "fed3-1104-INFO-2022-11-25-07-51"
filename_3 = "fed3-1104-INFO-2022-11-26-14-31"
filename_bf = "fedbf-1104-INFO-2022-11-26-15-39"
filename_opt = "fedopt-1104-INFO-2022-11-26-20-46"
DATA_PATH_3 = "../../../OutputData/fed_3"
DATA_PATH_OPT = "../../../OutputData/fed_opt"
DATA_PATH_BF = "../../../OutputData/fed_bf"
timestamp = time.time()
datatime = time.strftime("%Y-%m-%d-%H-%M", time.localtime(timestamp))
out_file_name = 'fed3-{}-{}'.format(1104, datatime)
OUTPUT_FILENAME = "fed3-opt-bf-{}".format(datatime)


def draw_IC(file_name=None):
    if file_name is None:
        file_name = in_filename
    ratio = []
    utility = []
    print("drawing:{}/{}.csv".format(DATA_PATH, file_name))
    with open("{}/{}.csv".format(DATA_PATH, file_name), mode="r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for row in reader:
            print("ratio:{}, utility:{}".format(row[0], row[1]))
            ratio.append(float(row[0]))
            utility.append(float(row[1]))

    plt.plot(ratio, utility, marker='o')
    plt.title("Incentive compatibility")
    plt.ylabel("Utility of Clients")
    plt.xlabel("Ratio")
    plt.savefig("{}/IC-{}.png".format(IMG_PATH, file_name))
    plt.show()


def draw_budget_balance(file_name=None):
    if file_name is None:
        file_name = in_filename
    with open("{}/{}.csv".format(DATA_PATH, file_name), mode="r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        budget_list = []
        client_num_list = []
        tot_payment_list = []
        for row in reader:
            print("number of clients:{}, tot_payment:{}, budget:{}".format(row[0], row[1], row[2]))
            budget_list.append(float(row[2]))
            client_num_list.append(int(row[0]))
            tot_payment_list.append(float(row[1]))
    ind = [i for i, _ in enumerate(client_num_list)]
    plt.bar(ind, budget_list, label='Budget')
    plt.bar(ind, tot_payment_list, label='Total Payment')

    plt.xticks(ind, client_num_list)
    plt.ylabel("The total payment and budget")
    plt.xlabel("the number of clients")
    plt.title("Budget Balance")
    plt.legend(loc="upper right")
    plt.savefig("{}/BB-{}.png".format(IMG_PATH, file_name))
    plt.show()


# TODO: draw IR
def draw_individual_rationality(file_name=None):
    if file_name is None:
        file_name = in_filename
    with open("{}/{}.csv".format(DATA_PATH, file_name), mode="r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        client_num_list = []
        payment_list = []
        cost_list = []
        for row in reader:
            print("number of clients:{}, true_cost:{}, payment:{}".format(row[0], row[1], row[2]))
            client_num_list.append(int(row[0]))
            cost_list.append(float(row[1]))
            payment_list.append(float(row[2]))

    ind = [i for i, _ in enumerate(client_num_list)]
    plt.bar(ind, payment_list, label='payment')
    plt.bar(ind, cost_list, label='real cost')

    plt.xticks(ind, client_num_list)
    plt.ylabel("The real cost and payment")
    plt.xlabel("the number of clients")
    plt.title("Individual Rationality")
    plt.legend(loc="upper right")
    plt.savefig("{}/IR-{}.png".format(IMG_PATH, file_name))
    plt.show()


def draw_accuracy(file_name=None):
    if file_name is None:
        file_name = in_filename
    round_list = []
    acc_list = []
    print("drawing:{}/{}.csv".format(DATA_PATH, file_name))
    with open("{}/{}.csv".format(DATA_PATH, file_name), mode="r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for row in reader:
            print("round:{}, acc:{}".format(row[0], row[1]))
            round_list.append(int(row[0]))
            acc_list.append(float(row[1]))

    plt.plot(round_list, acc_list)
    plt.title("Tested Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Rounds")
    plt.savefig("{}/ACC-{}.png".format(IMG_PATH, file_name))
    plt.show()


def draw_loss(file_name=None):
    if file_name is None:
        file_name = in_filename
    round_list = []
    loss_list = []
    print("drawing:{}/{}.csv".format(DATA_PATH, file_name))
    with open("{}/{}.csv".format(DATA_PATH, file_name), mode="r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for row in reader:
            print("round:{}, loss:{}".format(row[0], row[1]))
            round_list.append(int(row[0]))
            loss_list.append(float(row[2]))

    print("round_list:{}".format(round_list))
    plt.plot(round_list, loss_list)
    plt.title("Loss")
    plt.ylabel("Loss")
    plt.xlabel("Rounds")
    plt.savefig("{}/LOSS-{}.png".format(IMG_PATH, file_name))
    plt.show()


def draw_time(file_name=None):
    if file_name is None:
        file_name = in_filename
    round_list = []
    loss_list = []
    print("drawing:{}/{}.csv".format(DATA_PATH, file_name))
    with open("{}/{}.csv".format(DATA_PATH, file_name), mode="r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for row in reader:
            print("round:{}, time:{}".format(row[0], row[1]))
            round_list.append(int(row[0]))
            loss_list.append(float(row[3]))

    print("round_list:{}".format(round_list))
    plt.plot(round_list, loss_list)
    plt.title("Time")
    plt.ylabel("Time")
    plt.xlabel("Rounds")
    plt.savefig("{}/TIME-{}.png".format(IMG_PATH, file_name))
    plt.show()


def draw_accuracy_cmp():
    round_list = []
    acc_list_1 = []
    acc_list_2 = []
    acc_list_3 = []
    with open("{}/{}.csv".format(DATA_PATH_3, filename_3), mode="r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for row in reader:
            print("round:{}, acc:{}".format(row[0], row[1]))
            round_list.append(int(row[0]))
            acc_list_1.append(float(row[1]))

    with open("{}/{}.csv".format(DATA_PATH_OPT, filename_opt), mode="r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for row in reader:
            print("round:{}, acc:{}".format(row[0], row[1]))
            acc_list_2.append(float(row[1]))

    with open("{}/{}.csv".format(DATA_PATH_BF, filename_bf), mode="r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for row in reader:
            print("round:{}, acc:{}".format(row[0], row[1]))
            acc_list_3.append(float(row[1]))

    plt.plot(round_list, acc_list_1)
    plt.plot(round_list, acc_list_2)
    plt.plot(round_list, acc_list_3)
    plt.title("Tested Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Rounds")
    plt.legend(['ours', 'optimal', 'bid price first'])
    output_filename = "ACC-{}".format(OUTPUT_FILENAME)
    plt.savefig("{}/cmp/{}.png".format(IMG_PATH_PRE, output_filename))
    plt.show()


def draw_loss_cmp():
    round_list = []
    loss_list_1 = []
    loss_list_2 = []
    loss_list_3 = []
    with open("{}/{}.csv".format(DATA_PATH_3, filename_3), mode="r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for row in reader:
            print("round:{}, acc:{}".format(row[0], row[2]))
            round_list.append(int(row[0]))
            loss_list_1.append(float(row[2]))

    with open("{}/{}.csv".format(DATA_PATH_OPT, filename_opt), mode="r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for row in reader:
            print("round:{}, acc:{}".format(row[0], row[2]))
            loss_list_2.append(float(row[2]))

    with open("{}/{}.csv".format(DATA_PATH_BF, filename_bf), mode="r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for row in reader:
            print("round:{}, acc:{}".format(row[0], row[2]))
            loss_list_3.append(float(row[2]))

    plt.plot(round_list, loss_list_1)
    plt.plot(round_list, loss_list_2)
    plt.plot(round_list, loss_list_3)

    plt.title("Tested Loss")
    plt.ylabel("Loss")
    plt.xlabel("Rounds")
    plt.legend(['ours', 'optimal', 'bid price first'])
    output_filename = "LOSS-{}".format(OUTPUT_FILENAME)
    plt.savefig("{}/cmp/{}.png".format(IMG_PATH_PRE, output_filename))
    plt.show()


def draw_time_cmp():
    round_list = []
    time_list_1 = []
    time_list_2 = []
    time_list_3 = []
    with open("{}/{}.csv".format(DATA_PATH_3, filename_3), mode="r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for row in reader:
            print("round:{}, acc:{}".format(row[0], row[3]))
            round_list.append(int(row[0]))
            time_list_1.append(float(row[3]))

    with open("{}/{}.csv".format(DATA_PATH_OPT, filename_opt), mode="r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for row in reader:
            print("round:{}, acc:{}".format(row[0], row[3]))
            time_list_2.append(float(row[3]))

    with open("{}/{}.csv".format(DATA_PATH_BF, filename_bf), mode="r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for row in reader:
            print("round:{}, acc:{}".format(row[0], row[3]))
            time_list_3.append(float(row[3]))

    logging.info("round_list:{}".format(round_list))
    plt.plot(round_list, time_list_1)
    plt.plot(round_list, time_list_2)
    plt.plot(round_list, time_list_3)

    plt.title("Time")
    plt.ylabel("Time")
    plt.xlabel("Rounds")
    plt.legend(['ours', 'optimal', 'bid price first'])
    output_filename = "TIME-{}".format(OUTPUT_FILENAME)
    plt.savefig("{}/cmp/{}.png".format(IMG_PATH_PRE, output_filename))
    plt.show()


def draw_training_intensity_sum_cmp():
    round_list = []
    time_list_1 = []
    time_list_2 = []
    time_list_3 = []

    with open("{}/{}.csv".format(DATA_PATH_3, filename_3), mode="r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for row in reader:
            print("round:{}, acc:{}".format(row[0], row[4]))
            round_list.append(int(row[0]))
            time_list_1.append(float(row[4]))

    with open("{}/{}.csv".format(DATA_PATH_OPT, filename_opt), mode="r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for row in reader:
            print("round:{}, acc:{}".format(row[0], row[4]))
            time_list_2.append(float(row[4]))

    with open("{}/{}.csv".format(DATA_PATH_BF, filename_bf), mode="r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for row in reader:
            print("round:{}, acc:{}".format(row[0], row[4]))
            time_list_3.append(float(row[4]))

    logging.info("round_list:{}".format(round_list))
    plt.plot(round_list, time_list_1)
    plt.plot(round_list, time_list_2)
    plt.plot(round_list, time_list_3)

    plt.title("Total Training Intensity")
    plt.ylabel("Total Training Intensity")
    plt.xlabel("Rounds")
    plt.legend(['ours', 'optimal', 'bid price first'])
    output_filename = "TI-{}".format(OUTPUT_FILENAME)
    plt.savefig("{}/cmp/{}.png".format(IMG_PATH_PRE, output_filename))
    plt.show()


def draw_goal_cmp():
    round_list = []
    goal_list_1 = []
    goal_list_2 = []
    goal_list_3 = []

    with open("{}/{}.csv".format(DATA_PATH_3, filename_3), mode="r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for row in reader:
            print("round:{}, goal:{}".format(row[0], float(row[4]) / float(row[3])))
            round_list.append(int(row[0]))
            goal_list_1.append(float(row[4]) / float(row[3]))

    with open("{}/{}.csv".format(DATA_PATH_OPT, filename_opt), mode="r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for row in reader:
            print("round:{}, goal:{}".format(row[0], float(row[4]) / float(row[3])))
            goal_list_2.append(float(row[4]) / float(row[3]))

    with open("{}/{}.csv".format(DATA_PATH_BF, filename_bf), mode="r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for row in reader:
            print("round:{}, goal:{}".format(row[0], float(row[4]) / float(row[3])))
            goal_list_3.append(float(row[4]) / float(row[3]))

    logging.info("round_list:{}".format(round_list))
    plt.plot(round_list, goal_list_1)
    plt.plot(round_list, goal_list_2)
    plt.plot(round_list, goal_list_3)

    plt.title("Goal")
    plt.ylabel("Goal")
    plt.xlabel("Rounds")
    plt.legend(['ours', 'optimal', 'bid price first'])
    output_filename = "GOAL-{}".format(OUTPUT_FILENAME)
    plt.savefig("{}/cmp/{}.png".format(IMG_PATH_PRE, output_filename))
    plt.show()


def draw_accuracy_budget(file_name=None):
    if file_name is None:
        file_name = in_filename
    budget_list = []
    acc_list = []
    print("drawing:{}/{}.csv".format(DATA_PATH, file_name))
    with open("{}/{}.csv".format(DATA_PATH, file_name), mode="r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for row in reader:
            print("budget:{}, acc:{}".format(row[0], row[1]))
            budget_list.append(int(row[0]))
            acc_list.append(float(row[1]))

    plt.plot(budget_list, acc_list)
    plt.title("Tested Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Budget")
    plt.savefig("{}/B-ACC-{}.png".format(IMG_PATH, file_name))
    plt.show()


def draw_loss_budget(file_name):
    if file_name is None:
        file_name = in_filename
    budget_list = []
    loss_list = []
    print("drawing:{}/{}.csv".format(DATA_PATH, file_name))
    with open("{}/{}.csv".format(DATA_PATH, file_name), mode="r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for row in reader:
            print("budget:{}, loss:{}".format(row[0], row[1]))
            budget_list.append(int(row[0]))
            loss_list.append(float(row[2]))

    plt.plot(budget_list, loss_list)
    plt.title("Loss")
    plt.ylabel("Loss")
    plt.xlabel("Budget")
    plt.savefig("{}/B-LOSS-{}.png".format(IMG_PATH, file_name))
    plt.show()


def draw_time_budget(file_name):
    if file_name is None:
        file_name = in_filename
    budget_list = []
    loss_list = []
    print("drawing:{}/{}.csv".format(DATA_PATH, file_name))
    with open("{}/{}.csv".format(DATA_PATH, file_name), mode="r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for row in reader:
            print("round:{}, time:{}".format(row[0], row[1]))
            budget_list.append(int(row[0]))
            loss_list.append(float(row[3]))

    print("budget_list:{}".format(budget_list))
    plt.plot(budget_list, loss_list)
    plt.title("Time")
    plt.ylabel("Time")
    plt.xlabel("Budget")
    plt.savefig("{}/B-TIME-{}.png".format(IMG_PATH, file_name))
    plt.show()


def draw_accuracy_cmp_with_time():
    time_list_1 = []
    time_list_2 = []
    time_list_3 = []

    acc_list_1 = []
    acc_list_2 = []
    acc_list_3 = []
    with open("{}/{}.csv".format(DATA_PATH_3, filename_3), mode="r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for row in reader:
            print("time:{}, acc:{}".format(row[0], row[1]))
            if len(time_list_1) == 0:
                time_list_1.append(float(row[0]))
            else:
                time_list_1.append(time_list_1[-1] + float(row[0]))
            acc_list_1.append(float(row[1]))

    with open("{}/{}.csv".format(DATA_PATH_OPT, filename_opt), mode="r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for row in reader:
            print("time:{}, acc:{}".format(row[0], row[1]))
            if len(time_list_2) == 0:
                time_list_2.append(float(row[0]))
            else:
                time_list_2.append(time_list_2[-1] + float(row[0]))
            acc_list_2.append(float(row[1]))

    with open("{}/{}.csv".format(DATA_PATH_BF, filename_bf), mode="r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for row in reader:
            print("time:{}, acc:{}".format(row[0], row[1]))
            if len(time_list_3) == 0:
                time_list_3.append(float(row[0]))
            else:
                time_list_3.append(time_list_3[-1] + float(row[0]))
            acc_list_3.append(float(row[1]))

    plt.plot(time_list_1, acc_list_1)
    plt.plot(time_list_2, acc_list_2)
    plt.plot(time_list_3, acc_list_3)
    plt.title("Tested Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Time")
    plt.legend(['ours', 'optimal', 'bid price first'])
    output_filename = "T-ACC-{}".format(OUTPUT_FILENAME)
    plt.savefig("{}/cmp/{}.png".format(IMG_PATH_PRE, output_filename))
    plt.show()


def draw_loss_cmp_with_time():
    time_list_1 = []
    time_list_2 = []
    time_list_3 = []

    loss_list_1 = []
    loss_list_2 = []
    loss_list_3 = []
    with open("{}/{}.csv".format(DATA_PATH_3, filename_3), mode="r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for row in reader:
            print("time:{}, acc:{}".format(row[0], row[2]))
            if len(time_list_1) == 0:
                time_list_1.append(float(row[0]))
            else:
                time_list_1.append(time_list_1[-1] + float(row[0]))
            loss_list_1.append(float(row[2]))

    with open("{}/{}.csv".format(DATA_PATH_OPT, filename_opt), mode="r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for row in reader:
            print("time:{}, loss:{}".format(row[0], row[2]))
            if len(time_list_2) == 0:
                time_list_2.append(float(row[0]))
            else:
                time_list_2.append(time_list_2[-1] + float(row[0]))
            loss_list_2.append(float(row[2]))

    with open("{}/{}.csv".format(DATA_PATH_BF, filename_bf), mode="r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for row in reader:
            print("time:{}, loss:{}".format(row[0], row[2]))
            if len(time_list_3) == 0:
                time_list_3.append(float(row[0]))
            else:
                time_list_3.append(time_list_3[-1] + float(row[0]))
            loss_list_3.append(float(row[2]))

    plt.plot(time_list_1, loss_list_1)
    plt.plot(time_list_2, loss_list_2)
    plt.plot(time_list_3, loss_list_3)
    plt.title("Tested Loss")
    plt.ylabel("LOSS")
    plt.xlabel("Time")
    plt.legend(['ours', 'optimal', 'bid price first'])
    output_filename = "T-LOSS-{}".format(OUTPUT_FILENAME)
    plt.savefig("{}/cmp/{}.png".format(IMG_PATH_PRE, output_filename))
    plt.show()

def draw_time_cmp_with_budget():
    budget_list = []

    time_list_1 = []
    time_list_2 = []
    time_list_3 = []
    with open("{}/{}.csv".format(DATA_PATH_3, filename_3), mode="r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for row in reader:
            print("budget:{}, acc:{}".format(row[0], row[2]))
            budget_list.append(float(row[0]))
            time_list_1.append(float(row[2]))

    with open("{}/{}.csv".format(DATA_PATH_OPT, filename_opt), mode="r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for row in reader:
            print("budget:{}, time:{}".format(row[0], row[2]))
            time_list_2.append(float(row[2]))

    with open("{}/{}.csv".format(DATA_PATH_BF, filename_bf), mode="r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for row in reader:
            print("budget:{}, time:{}".format(row[0], row[2]))
            time_list_3.append(float(row[2]))

    plt.plot(budget_list, time_list_1)
    plt.plot(budget_list, time_list_2)
    plt.plot(budget_list, time_list_3)
    plt.title("Tested Time")
    plt.ylabel("Time")
    plt.xlabel("Budget")
    plt.legend(['ours', 'optimal', 'bid price first'])
    output_filename = "B-T-{}".format(OUTPUT_FILENAME)
    plt.savefig("{}/cmp/{}.png".format(IMG_PATH_PRE, output_filename))
    plt.show()