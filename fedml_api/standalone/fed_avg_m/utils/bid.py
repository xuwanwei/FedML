class Bid:

    def __init__(self, client_idx, training_intensity, cost, bidding_price, computation_coefficient,
                 communication_time):
        """client_idx is the index of the client in fedb_api.client_list"""
        self.client_idx = client_idx
        self.training_intensity = training_intensity
        self.cost = cost
        self.bidding_price = bidding_price
        self.computation_coefficient = computation_coefficient
        self.communication_time = communication_time
        self.avg_cost = 0
        self.time = 0

    def update_bid(self, training_intensity, cost, bidding_price, computation_coefficient, communication_time):
        self.training_intensity = training_intensity
        self.cost = cost
        self.bidding_price = bidding_price
        self.computation_coefficient = computation_coefficient
        self.communication_time = communication_time
        self.time = self.training_intensity * self.computation_coefficient + self.communication_time
        self.avg_cost = self.training_intensity / self.bidding_price

    def get_time(self):
        return self.time

    def get_average_cost(self):
        return self.avg_cost

    def get_cost(self):
        return self.cost

    def get_training_intensity(self):
        return self.training_intensity

    def update_avg_cost(self):
        self.avg_cost = self.training_intensity / self.bidding_price

    def update_bid_with_ratio(self, truth_ratio):
        self.bidding_price = truth_ratio * self.cost


