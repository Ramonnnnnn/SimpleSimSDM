import numpy as np
import Parser

class TrafficGenerator:
    def __init__(self, XML_path, seed):
        self.XML_path = XML_path
        parser_object = Parser.XmlParser(XML_path)
        self.node_list = parser_object.get_nodes()
        self.call_info = parser_object.get_calls_info()
        self.traffic_info = parser_object.get_traffic_info()
        self.slots_bandwidth = parser_object.get_slots_bandwidth()
        self.seed = seed

        #Generate useful means
        self.max_rate = self.traffic_info["max_rate"]
        total_weight = 0
        self.mean_rate = 0
        self.mean_holding_time = 0

        iter_calls = 0
        for i in self.call_info[1:]:
            iter_calls += 1
        for i in range(iter_calls):
            total_weight += self.call_info[i]["weight"]
        for i in range(iter_calls):
            holding_time = self.call_info[i]["holding_time"]
            rate = self.call_info[i]["rate"]
            cos = self.call_info[i]["cos"] #Not used right now
            weight = self.call_info[i]["weight"]
            self.mean_rate += rate * (weight/total_weight)
            self.mean_holding_time += holding_time * (weight/total_weight)

    def generate_pairs(self, num_pairs):
        mean = len(self.node_list) / 2
        std_dev = len(self.node_list) / 3
        pairs = []

        if self.seed is not None:
            np.random.seed(self.seed)

        for _ in range(num_pairs):
            while True:
                source_index, destination_index = np.random.normal(loc=mean, scale=std_dev, size=2)
                source_index, destination_index = int(np.clip(source_index, 0, len(self.node_list) - 1)), int(
                    np.clip(destination_index, 0, len(self.node_list) - 1))
                if source_index != destination_index:
                    break
            pair = (self.node_list[source_index], self.node_list[destination_index])
            pairs.append(pair)
        return pairs

    def generate_normal_distribution_call_types(self, n):

        if self.seed is not None:
            np.random.seed(self.seed)

        # Count the number of rows in calls_info
        r = len(self.call_info)

        # Create a normal distribution with mean = (r-1)/2 and standard deviation = r/6
        mean = (r - 1) / 2
        std_dev = r / 6
        normal_distribution = np.random.normal(loc=mean, scale=std_dev, size=n)

        # Round and clip values to ensure they are within the valid range [0, r-1]
        normal_distribution = np.round(normal_distribution).clip(0, r - 1).astype(int)

        return normal_distribution

    def get_node_list(self):
        return self.node_list

    def get_call_info(self):
        return self.call_info

    def get_traffic_info(self):
        return self.traffic_info

    def get_mean_holding_time(self):
        return self.mean_holding_time

    def get_mean_rate(self):
        return self.mean_rate

    def get_max_rate(self):
        return self.max_rate

    def get_slots_bandwidth(self):
        return self.slots_bandwidth



    def generate_poisson_events(self, load_erlangs, mean_holding_time, attempt_n_calls):
        # Using offered load in E to calculate
        call_arrival_rate = load_erlangs / mean_holding_time  # IN SECONDS 1000E/1s = 1000 calls/s
        # Total (simulation) time duration will depend on the arrival rate and number of desired calls to be attempted
        total_time_duration = attempt_n_calls / call_arrival_rate  # IN SECONDS 10000calls/1000 = 10s duration
        # Exponential
        the_lambda = attempt_n_calls / total_time_duration
        inter_arrival_times = np.random.exponential(scale=1 / the_lambda, size=attempt_n_calls)
        # Calculate the cumulative sum to get the arrival times
        arrival_times = np.cumsum(inter_arrival_times)
        return attempt_n_calls, arrival_times, inter_arrival_times, total_time_duration

    def generate_call_durations(self, num_calls, mean_holding_time):
        # Exponential distribution for call durations (holding times)
        call_durations = np.random.exponential(scale=mean_holding_time, size=num_calls)
        diff_call_durations = np.diff(call_durations)
        return call_durations, diff_call_durations