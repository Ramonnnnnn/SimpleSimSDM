import numpy as np
import TopologyBuilder

class Metrics:

    def __init__(self, call_rates, slot_capacity, total_attempts):

        self.call_rates = call_rates
        self.slot_capacity = slot_capacity

        #Fragmentation
        self.topology_fragmentation_sampled_regularly = []
        self.end_of_simulation_round_fragmentation = [] #This one retains memory across confidence interval


        #For BBR use
        self.blockedcalls = 0
        self.total_attempts = total_attempts
        self.bbr_of_every_round = []

        #Time
        self.elapsed_execution_time_per_round = []
        self.elapsed_simulation_time_per_round = []

        #CPS
        self.topology_CpS_sampled_regularly = []
        self.end_of_simulation_round_CpS = []


    #Time-keeping
    def add_execution_time_of_round(self, time):
        self.elapsed_execution_time_per_round.append(time)
    def calculate_mean_execution_time_of_round(self):
        return np.mean(self.elapsed_execution_time_per_round)
    def add_simulation_time_of_round(self, time):
        self.elapsed_simulation_time_per_round.append(time)
    def calculate_mean_simulation_time_of_round(self):
        return np.mean(self.elapsed_simulation_time_per_round)

    def calculate_margin_of_error(self, sample):
        # Calculate sample mean
        sample_mean = np.mean(sample)

        # Calculate sample standard deviation
        sample_std = np.std(sample, ddof=1)  # Use ddof=1 for sample standard deviation

        # Determine sample size
        n = len(sample)

        # Compute critical value for 95% confidence interval (assuming normal distribution)
        z = 1.96  # For 95% confidence interval

        # Calculate margin of error
        margin_of_error = z * (sample_std / np.sqrt(n))

        return margin_of_error

    def calculate_end_of_simulation_round_bbr(self, blocked_calls):
        self.bbr_of_every_round.append((blocked_calls/self.total_attempts) * 100)

    def calculate_final_BBR_and_confidence_interval(self):
        final_BBR = np.mean(self.bbr_of_every_round)
        confidence = self.calculate_margin_of_error(self.bbr_of_every_round)
        return final_BBR, confidence

    #Measures fragmentation potential linearly per link
    def mean_fragmentation_per_link(self, spectrum_matrix):
        frag_lengths = []
        frag_potential = []
        for row in range(len(spectrum_matrix)):
            temp_frag_size = 0
            col = 0
            while col < len(spectrum_matrix[0]):
                temp_frag_size = 0
                if spectrum_matrix[row][col] == 0:
                    temp_frag_size += 1
                    col += 1
                    while col < len(spectrum_matrix[0]) and spectrum_matrix[row][col] == 0:
                        temp_frag_size += 1
                        col += 1
                    frag_lengths.append(temp_frag_size)
                elif spectrum_matrix[row][col] == 1:
                    col += 1
        for frag in frag_lengths:
            count = 0
            for rate in self.call_rates:
                if (rate/self.slot_capacity) >= frag:
                    count += 1
            frag_potential.append(count/len(self.call_rates))

        return np.mean(frag_potential)


    #Takes the mean fragmentation for the whole topology at a given point (usually every 100 calls)
    #for a given load (every step for a specified interval)
    def mean_frag_topology(self, matrices):
        fragmentation_for_every_link = []
        # get all edge matrices
        all_edge_matrices = matrices #receive edge_matrices during execution from topology.
        # Iterate over matrices
        for matrix in all_edge_matrices:
            fragmentation_for_every_link.append(self.mean_fragmentation_per_link(matrix))
        self.topology_fragmentation_sampled_regularly.append(np.mean(fragmentation_for_every_link))

    def calculate_end_of_simulation_round_fragmentation(self):
        self.end_of_simulation_round_fragmentation.append(np.mean(self.topology_fragmentation_sampled_regularly))

    def calculate_final_fragmentation_and_confidence_interval(self):
        final_fragmentation = np.mean(self.end_of_simulation_round_fragmentation)
        confidence_interval = self.calculate_margin_of_error(self.end_of_simulation_round_fragmentation)
        return final_fragmentation, confidence_interval

    def CpS_per_link(self, spectrum):
        if len(spectrum) == 1:
            return -1

        counter = 0
        used_slots = 0
        rows = len(spectrum)
        columns = len(spectrum[0])
        for row in range(rows):
            for column in range(columns):
                if spectrum[row][column] == 1:
                    used_slots += 1
                    if row == 0:
                        if spectrum[rows-1][column] == 1:
                            counter += 1
                        if spectrum[row+1][column] == 1:
                            counter += 1
                    elif row == rows-1:
                        if spectrum[0][column] == 1:
                            counter += 1
                        if spectrum[row-1][column] == 1:
                            counter += 1
                    else:
                        if spectrum[row-1][column] == 1:
                            counter += 1
                        if spectrum[row+1][column] == 1:
                            counter += 1

        if used_slots > 0:
            return counter/used_slots
        elif used_slots == 0:
            return 0

    def mean_CpS_topology(self, all_spectrum_matrices):
        CpS_for_every_link = []
        all_spectrum_matrices = all_spectrum_matrices
        for matrix in all_spectrum_matrices:
            CpS_for_every_link.append(self.CpS_per_link(matrix))
        self.topology_CpS_sampled_regularly.append(np.mean(CpS_for_every_link))



    def calculate_end_of_simulation_round_CpS(self):
        self.end_of_simulation_round_CpS.append(np.mean(self.topology_CpS_sampled_regularly))

    def calculate_final_CpS_and_confidence_interval(self):
        final_CpS = np.mean(self.end_of_simulation_round_CpS)
        confidence_interval = self.calculate_margin_of_error(self.end_of_simulation_round_CpS)
        return final_CpS, confidence_interval

    def reset_roundwise_stats(self):
        self.topology_fragmentation_sampled_regularly = []
        self.topology_CpS_sampled_regularly = []
        self.blockedcalls = 0





