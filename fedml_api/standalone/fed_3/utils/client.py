from fedml_api.standalone.fed_3.utils.bid import Bid


class Client:

    def __init__(self, client_idx=0, args=None, device=None, training_intensity=0, cost=0, computation_coefficient=0,
                 communication_time=0, local_training_data=None, local_test_data=None, local_sample_number=0,
                 model_trainer=None):
        self.client_idx = client_idx
        self.args = args
        self.device = device
        self.bid = Bid(client_idx, training_intensity, cost, cost, computation_coefficient, communication_time)
        self.payment = 0

        # model and data
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number
        self.model_trainer = model_trainer

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

    def update_bid(self, training_intensity, cost, truth_ratio, computation_coefficient, communication_time):
        self.bid.update_bid(training_intensity, cost, truth_ratio * cost, computation_coefficient,
                            communication_time)
        # after updating bid, the payment will be set to 0
        self.payment = 0

    def get_average_cost(self):
        return self.bid.get_average_cost()

    def get_training_intensity(self):
        return self.bid.training_intensity

    def get_time(self):
        return self.bid.get_time()

    def get_cost(self):
        return self.bid.get_cost()

    def receive_payment(self, payment):
        self.payment = payment

    def get_utility(self):
        if self.payment == 0:
            return 0
        return self.payment - self.get_cost()

    def update_bidding_price_with_ratio(self, truth_ratio):
        self.bid.update_bid_with_ratio(truth_ratio)

    def get_bidding_price(self):
        return self.bid.bidding_price

    def get_payment(self):
        return self.payment
