from fedml_api.data_preprocessing.MNIST.data_loader_with_client_num import load_partition_data_mnist


def debug_load_partition_data_mnist():
    client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
    train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
    class_num = load_partition_data_mnist(32)

debug_load_partition_data_mnist()