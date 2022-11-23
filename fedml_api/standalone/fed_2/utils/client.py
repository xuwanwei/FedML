from fedml_api.standalone.fed_2.utils.bid import Bid


class Client:

    def __init__(self, client_idx, args, device, training_intensity, cost, computation_coefficient, communication_time):
        self.client_idx = client_idx
        self.args = args
        self.device = device
        self.bid = Bid(client_idx, training_intensity, cost, cost, computation_coefficient, communication_time)
        self.payment = 0

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
