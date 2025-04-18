import numpy as np
import TopologyBuilder
import os
import matplotlib.pyplot as plt
import numpy as np
import math
from collections import Counter


class Metrics:

    def __init__(self, call_rates, slot_capacity, total_attempts):

        self.call_rates = call_rates
        self.slot_capacity = slot_capacity

        # Fragmentation
        self.topology_fragmentation_sampled_regularly = []
        self.end_of_simulation_round_fragmentation = []  # This one retains memory across confidence interval

        # Crosstalk
        self.topology_crosstalk_sampled_regularly = []
        self.end_of_simulation_round_crosstalk = []

        # For blocked call ratio use
        self.blockedcalls = 0
        self.total_attempts = total_attempts
        self.bcr_of_every_round = []

        # For BBR use
        self.bbr_of_every_round = []
        # self.

        # Time
        self.elapsed_execution_time_per_round = []
        self.elapsed_simulation_time_per_round = []

        # CPS
        self.topology_CpS_sampled_regularly = []
        self.end_of_simulation_round_CpS = []

    # Time-keeping
    def add_execution_time_of_round(self, time):
        self.elapsed_execution_time_per_round.append(time)

    def calculate_mean_execution_time_of_round(self):
        return np.mean(self.elapsed_execution_time_per_round)

    def add_simulation_time_of_round(self, time):
        self.elapsed_simulation_time_per_round.append(time)

    def calculate_mean_simulation_time_of_round(self):
        return np.mean(self.elapsed_simulation_time_per_round)

    def save_modulation_histogram(self, array, folder_path, load, status):
        if len(array) >= 1:

            if status == 1:
                file_name = "modulation_histogram_" + str(load) + '.png'
            else:
                file_name = "blocked_modulation_histogram_" + str(load) + '.png'
            # Check if the folder exists, create if it does not
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            # Define unique colors for each bar
            colors = plt.cm.viridis(np.linspace(0, 1, len(set(array))))

            # Create the histogram
            counts, bins, patches = plt.hist(array, bins=np.arange(min(array) - 0.5, max(array) + 1.5, 1),
                                             edgecolor='black',
                                             align='mid')

            # Set colors for each bar
            for patch, color in zip(patches, colors):
                patch.set_facecolor(color)

            # Add titles and labels
            plt.title(f'Modulation Histogram - Load: {load}')
            plt.xlabel('Modulation')
            plt.ylabel('Frequency')

            # Save the plot to the specified folder
            file_path = os.path.join(folder_path, file_name)
            plt.savefig(file_path)

            # Close the plot to free up memory
            plt.close()

    def distance_mask_for_histogram(self, distance):
        aux = distance / 100

        if aux < 1:
            return 1
        else:
            return math.ceil(aux)

    def save_path_lenght_histogram(self, array, folder_path, load, status):
        if len(array) >= 1:
            if status == 1:
                file_name = "path_length_histogram_" + str(load) + '.png'
            else:
                file_name = "blocked_path_length_histogram_" + str(load) + '.png'
            # Check if the folder exists, create if it does not
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            # Define unique colors for each bar
            colors = plt.cm.viridis(np.linspace(0, 1, len(set(array))))

            # Create the histogram
            counts, bins, patches = plt.hist(array, bins=np.arange(min(array) - 0.5, max(array) + 1.5, 1),
                                             edgecolor='black',
                                             align='mid')

            # Set colors for each bar
            for patch, color in zip(patches, colors):
                patch.set_facecolor(color)

            # Add titles and labels
            plt.title(f'Path Length Histogram: {load}')
            plt.xlabel('Length(KMx100)')
            plt.ylabel('Frequency')

            # Save the plot to the specified folder
            file_path = os.path.join(folder_path, file_name)
            plt.savefig(file_path)

            # Close the plot to free up memory
            plt.close()

    @staticmethod
    def calculate_success_percentage(arr1, arr2, load):
        # Count occurrences in both arrays
        count1 = Counter(arr1)
        count2 = Counter(arr2)

        # Combine keys from both counters
        all_keys = set(count1.keys()).union(count2.keys())

        # Calculate success percentages
        success_percentage = {}
        for key in all_keys:
            total_occurrences = count1[key] + count2[key]
            if total_occurrences > 0:
                success_percentage[key] = (count1[key] / total_occurrences) * 100

        # Sort the success percentages by ID
        sorted_keys = sorted(success_percentage.keys())
        sorted_percentages = [success_percentage[key] for key in sorted_keys]

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.bar(sorted_keys, sorted_percentages, color='skyblue')
        plt.xlabel('ID')
        plt.ylabel('Success Percentage')
        plt.title(f'links_served_{load}')
        plt.xticks(sorted_keys)

        # Create directory if it doesn't exist
        directory = "served_percentage"
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Save the plot
        plt.savefig(os.path.join(directory, f'links_served_{load}.png'))
        plt.close()

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

    def calculate_end_of_simulation_round_bcr(self, blocked_calls):
        self.bcr_of_every_round.append((blocked_calls / self.total_attempts) * 100)

    def calculate_final_BCR_and_confidence_interval(self):
        final_BCR = np.mean(self.bcr_of_every_round)
        confidence = self.calculate_margin_of_error(self.bcr_of_every_round)
        return final_BCR, confidence

    def calculate_end_of_simulation_round_bbr(self, blocked_bandwidth, successful_bandwidth):
        self.bbr_of_every_round.append((blocked_bandwidth / (blocked_bandwidth + successful_bandwidth)) * 100)
        print(self.bbr_of_every_round)
        print(self.bcr_of_every_round)

    def calculate_final_BBR_and_confidence_interval(self):
        final_BBR = np.mean(self.bbr_of_every_round)
        confidence = self.calculate_margin_of_error(self.bbr_of_every_round)
        return final_BBR, confidence

    # Measures fragmentation potential linearly per link
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
                if (rate / self.slot_capacity) >= frag:
                    count += 1
            frag_potential.append(count / len(self.call_rates))

        return np.mean(frag_potential)

    # Takes the mean fragmentation for the whole topology at a given point (usually every 100 calls)
    # for a given load (every step for a specified interval)
    def mean_frag_topology(self, matrices):
        fragmentation_for_every_link = []
        # get all edge matrices
        all_edge_matrices = matrices  # receive edge_matrices during execution from topology.
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

    @staticmethod
    def figure_neighbors_seven_core_MCF(core):
        neighbors_map = {
            0: [1, 5, 6],
            1: [0, 2, 6],
            2: [1, 3, 6],
            3: [2, 4, 6],
            4: [3, 5, 6],
            5: [0, 4, 6],
            6: [0, 1, 2, 3, 4, 5]
        }
        return neighbors_map.get(core, [])

    def CpS_per_link(self, spectrum):
        # if len(spectrum) == 1:
        #     return -1
        if len(spectrum) != 7:
            raise KeyError(f"Change neighbor finding function.")

        rows = len(spectrum)
        columns = len(spectrum[0])
        aux = 0
        used_slots_aux = 0
        for row in range(rows):
            for column in range(columns):
                if spectrum[row][column]:
                    shadowed = False
                    used_slots_aux += 1
                    nei = self.figure_neighbors_seven_core_MCF(row)
                    for core in nei:
                        if spectrum[core][column]:
                            shadowed = True
                    if shadowed:
                        aux += 1
        if used_slots_aux == 0:
            return 0
        else:
            return aux/used_slots_aux


        # counter = 0
        # used_slots = 0
        # rows = len(spectrum)
        # columns = len(spectrum[0])
        # for row in range(rows):
        #     for column in range(columns):
        #         if spectrum[row][column] == 1:
        #             used_slots += 1
        #             if row == 0:
        #                 if spectrum[rows - 1][column] == 1:
        #                     counter += 1
        #                 if spectrum[row + 1][column] == 1:
        #                     counter += 1
        #             elif row == rows - 1:
        #                 if spectrum[0][column] == 1:
        #                     counter += 1
        #                 if spectrum[row - 1][column] == 1:
        #                     counter += 1
        #             else:
        #                 if spectrum[row - 1][column] == 1:
        #                     counter += 1
        #                 if spectrum[row + 1][column] == 1:
        #                     counter += 1
        #
        # if used_slots > 0:
        #     return counter / used_slots
        # elif used_slots == 0:
        #     return 0

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

    ############################################### CROSSTALK ####################################

    def crosstalk_per_link(self, noise_matrix):
        noise_matrix = np.array(noise_matrix) # python array (matrix)
        linear_noise_matrix = 10 ** (noise_matrix / 10) #np.array
        return np.mean(linear_noise_matrix) #int

    def mean_crosstalk_topology(self, all_noise_matrices):
        crosstalk_for_every_link = []
        all_noise_matrices = all_noise_matrices
        for matrix in all_noise_matrices:
            crosstalk_for_every_link.append(self.crosstalk_per_link(matrix)) # int appended to array
        self.topology_crosstalk_sampled_regularly.append(np.mean(crosstalk_for_every_link)) #mean of array appended to array

    def calculate_end_of_simulation_round_crosstalk(self):
        self.end_of_simulation_round_crosstalk.append(np.mean(self.topology_crosstalk_sampled_regularly)) #Mean of array appended to array

    def calculate_final_crosstalk_and_confidence_interval(self):
        final_crosstalk = np.mean(self.end_of_simulation_round_crosstalk) # Int
        final_db_crosstalk = 10 * np.log10(final_crosstalk) # Int
        db_end_of_simulation_round_crosstalk = 10 * np.log10(self.end_of_simulation_round_crosstalk)
        confidence_interval = self.calculate_margin_of_error(db_end_of_simulation_round_crosstalk)
        return final_db_crosstalk, confidence_interval

    ############################################### CROSSTALK ####################################

    def reset_roundwise_stats(self):
        self.topology_fragmentation_sampled_regularly = []
        self.topology_CpS_sampled_regularly = []
        self.topology_fragmentation_sampled_regularly = []
        self.blockedcalls = 0
