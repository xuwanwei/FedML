import logging
from fedml_api.standalone.fedti.bid import Bid


class Client:

    def __init__(self, client_idx, local_training_data, local_test_data, local_sample_number, args, device,
                 model_trainer, training_intensity, cost, computation_coefficient, communication_time):
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number
        # logging.info("self.local_sample_number = " + str(self.local_sample_number))
        self.args = args
        self.device = device
        self.model_trainer = model_trainer
        self.bid = Bid(client_idx, training_intensity, cost, computation_coefficient, communication_time)

    def update_local_dataset(self, client_idx, local_training_data, local_test_data, local_sample_number):
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number

    def get_sample_number(self):
        return self.local_sample_number

    def train(self, w_global):
        self.model_trainer.set_model_params(w_global)
        self.model_trainer.train(self.local_training_data, self.device, self.args)
        weights = self.model_trainer.get_model_params()
        return weights

    def local_test(self, b_use_test_dataset):
        if b_use_test_dataset:
            test_data = self.local_test_data
        else:
            test_data = self.local_training_data
        metrics = self.model_trainer.test(test_data, self.device, self.args)
        return metrics

    def update_bid(self, training_intensity, cost, computation_coefficient, communication_time):
        self.bid.update_bid(training_intensity, cost, computation_coefficient, communication_time)

    def get_average_cost(self):
        return self.bid.get_average_cost()

    def get_training_intensity(self):
        return self.bid.training_intensity

    def get_time(self):
        return self.bid.get_time()

    def get_cost(self):
        return self.bid.get_cost()
