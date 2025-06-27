import math
import random
import gymnasium as gym
from gymnasium import spaces
import numpy as np
# printing stuff
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors



class TetrisResourceAllocation(gym.Env):
    ''' In this environment, every episode attempts to solve allocation for a unique link, i.e., there is no continuity as in
    regular packing approaches. So, episode_i will be entirely devoted to finding the best position for light-path_i to be
    placed in the spectrum for path_i.  '''
    def __init__(self, cores, slots, max_episode_length, total_timesteps):
        super(TetrisResourceAllocation, self).__init__()

        # Spectrum Dimension
        self.candidate_pivot = None
        self.demanded_slots = None
        self.current_starting_position = None
        self.slots = slots
        self.cores = cores
        # Define the action space: selecting a row and a starting column
        self.action_space = self.action_space = spaces.Discrete(4)  # up, down, left, right

        # Define the observation space: spectrum matrix and metrics
        self.observation_space = spaces.Dict({
            "spectrum": spaces.Box(low=0, high=1, shape=(cores * slots,), dtype=np.int32),  # path spectrum
            "current_placement_attempt_matrix": spaces.Box(low=0, high=4, shape=(cores * slots,), dtype=np.int32),
            "reward_matrix": spaces.Box(low=-1.0, high=4, shape=(cores * slots,), dtype=np.float64)

        })

        # Initialize internal state
        self.state = None
        self.max_episode_length = max_episode_length  # does not change
        self.total_timesteps = total_timesteps  # does not change
        self.current_timestep = 0  # the universal timestep_counter
        self.current_episode = 0  # episode = n_timesteps or until done = True
        self.this_episodes_timestep = 0  # goes to zero at every reset()
        # counter for episode end
        self.tried_this_many_allocations = 0
        # Counter to not insist too much on a mostly filled spectrum matrix
        self.last_episode_where_ratio_was_reset = 0

        # Tetris Approach
        self.spectrum = self.spectrum = np.zeros((cores, slots), dtype=np.int32)
        self.current_placement_attempt_matrix = np.zeros((cores, slots), dtype=np.int32)
        self.reward_matrix = np.zeros((cores, slots), dtype=np.float64)
        self.slots_remaining = None
        self.remaining_reward_on_matrix = None
        self.total_positive_reward_on_matrix = None
        # Structures for Exporting
        self.step_by_step_region = []
        self.dict_to_export = {}

    def reset(self, seed=None, **kwargs):
        """This version will move left or right of the earliest overlap."""
        super().reset(seed=seed)  # Call parent class's reset method to handle seeding
        if seed is not None:
            np.random.seed(seed)  # Update NumPy random state
        # Count episodes (universal)
        self.current_episode += 1
        # Counter (resets for episode)
        self.this_episodes_timestep = 0
        # Get new spectrum
        self.spectrum = self.generate_blocky_binary_matrix(self.cores, self.slots, max_fill_prob=0.5, max_attempts=300) #Generates episode's matrix
        self.spectrum[0][0] = 1 #No longer will randomly allocate successfuly when starting position [0,0]
        ratio = np.sum(self.spectrum) / self.spectrum.size
        print(f"Current Occ. Ratio: {ratio:.3f}")
        # Demanded Slots (probabilistic distribution for training)
        self.demanded_slots = self.generate_probabilistic_number()  # Changes at every reset.
        # Starting Coordinate
        self.current_starting_position = [0, 0]  # Start position for our block
        # Alternative random start
        # row, col = self.get_random_position()
        # self.current_starting_position = [row, col]
        self.candidate_pivot = None
        # Free spectrum for step 0
        self.current_placement_attempt_matrix = self.spectrum.copy()  # Independent copy of the spectrum
        # Reward Matrix for step 0 - reset every new episode
        self.build_reward_matrix(self.cores)
        # Set total positive reward for future comparison
        self.total_positive_reward_on_matrix = self.remaining_reward_on_matrix #Independent copy
        # Write candidate block to auxiliary matrix
        self.write_to_auxiliary_matrix()
        # Save_matrix to turn into image before action
        if self.current_timestep == 0:
            self.save_matrix_log(self.current_placement_attempt_matrix, "action", self.current_timestep)
        # Initialize state
        self.state = {
            "spectrum": self.spectrum.flatten(),
            "current_placement_attempt_matrix": self.current_placement_attempt_matrix.flatten(),
            "reward_matrix": self.reward_matrix.flatten()
        }
        # Return observation and an empty info dictionary
        return self.state, {}

    def step(self, action):
        """Take an action and compute the next state, reward, and done flag."""
        # Iterate counters
        self.this_episodes_timestep += 1
        self.current_timestep += 1
        # Count overlaps:
        overlaps = self.count_overlaps() #Counts from spectrum matrix
        # If no action was needed because the start_position was valid
        if overlaps == 0 and self.this_episodes_timestep == 1:
            print(f"Ended in success at step 1, at random")
            self.write_to_spectrum()
            self.state = {
                "spectrum": self.spectrum.flatten(),
                "current_placement_attempt_matrix": self.current_placement_attempt_matrix.flatten(),
                "reward_matrix": self.reward_matrix.flatten()
            }
            reward = 0
            return self.state, reward, True, True, {}

        if overlaps == 0 and self.this_episodes_timestep > 1:
            raise KeyError("Logic Problem")


        # Call action to deal with movement
        # Is action valid?
        # Candidate is in a bad position prior to allocation, but action is possible
        if action == 2:
            row, col = self.get_first_overlap()
            self.candidate_pivot = [row, col]
        if action == 3:
            row, col = self.get_last_overlap()
            self.candidate_pivot = [row, col]
        if (action < 2 and self.is_one_step_translation_possible(action)) or (action >= 2 and self.is_pivot_translation_possible(action)):
            # Dewrite from auxiliary matrix
            self.dewrite_from_auxiliary_matrix()
            # Then move
            if action >= 2:
                self.move_starting_position_across_pivot(action)
            else:
                self.move_starting_position(action)
            # Write new position to auxiliary matrix
            self.write_to_auxiliary_matrix()
            # Collect Reward & update reward matrix
            reward = self.collect_reward()
            # Count new position's overlaps/overshadows
            new_overlaps = self.count_overlaps()
            self.save_matrix_log(self.current_placement_attempt_matrix, "action", self.current_timestep)
            if new_overlaps == 0:
                print(f"Ended in success at step {self.this_episodes_timestep}")
                self.write_to_spectrum()
                self.state = {
                    "spectrum": self.spectrum.flatten(),
                    "current_placement_attempt_matrix": self.current_placement_attempt_matrix.flatten(),
                    "reward_matrix": self.reward_matrix.flatten()
                }
                reward += self.demanded_slots * ((self.max_episode_length - self.this_episodes_timestep)/self.max_episode_length)
                return self.state, reward, True, True, {}
            elif (self.this_episodes_timestep == self.max_episode_length) or (self.remaining_reward_on_matrix/self.total_positive_reward_on_matrix <= 0.05):
                print(f"Could not place in the spectrum until the end of the episode. Positive reward remaining: {(self.remaining_reward_on_matrix/self.total_positive_reward_on_matrix)*100:.3f}%")
                self.state = {
                    "spectrum": self.spectrum.flatten(),
                    "current_placement_attempt_matrix": self.current_placement_attempt_matrix.flatten(),
                    "reward_matrix": self.reward_matrix.flatten()
                }
                reward = -10
                if self.remaining_reward_on_matrix/self.total_positive_reward_on_matrix <= 0.05:
                    reward = 1
                return self.state, reward, True, True, {}
            else:
                self.state = {
                    "spectrum": self.spectrum.flatten(),
                    "current_placement_attempt_matrix": self.current_placement_attempt_matrix.flatten(),
                    "reward_matrix": self.reward_matrix.flatten()
                }
                reward = reward * ((self.max_episode_length - self.this_episodes_timestep)/self.max_episode_length)
                return self.state, reward, False, False, {}

        # Not a valid action
        else:
            if (self.this_episodes_timestep == self.max_episode_length) or (self.remaining_reward_on_matrix/self.total_positive_reward_on_matrix <= 0.05):
                print(f"Could not place in the spectrum until the end of the episode. Positive reward remaining: {(self.remaining_reward_on_matrix / self.total_positive_reward_on_matrix) * 100:.3f}%")
                self.state = {
                    "spectrum": self.spectrum.flatten(),
                    "current_placement_attempt_matrix": self.current_placement_attempt_matrix.flatten(),
                    "reward_matrix": self.reward_matrix.flatten()
                }
                reward = -10
                if self.remaining_reward_on_matrix/self.total_positive_reward_on_matrix <= 0.05:
                    reward = 0
                return self.state, reward, True, True, {}
            else:
                self.state = {
                    "spectrum": self.spectrum.flatten(),
                    "current_placement_attempt_matrix": self.current_placement_attempt_matrix.flatten(),
                    "reward_matrix": self.reward_matrix.flatten()
                }
                reward = -1
                return self.state, reward, False, False, {}

    def write_to_spectrum(self):
        for i in range(self.current_starting_position[1], (self.current_starting_position[1] + self.demanded_slots)):
            if self.spectrum[self.current_starting_position[0]][i] == 1:
                raise KeyError("Should not happen")
            else:
                self.spectrum[self.current_starting_position[0]][i] = 1

    def write_to_auxiliary_matrix(self):
        for i in range(self.current_starting_position[1], (self.current_starting_position[1] + self.demanded_slots)):
            self.current_placement_attempt_matrix[self.current_starting_position[0]][i] += 2

    def collect_reward(self):
        total_reward = 0
        total_positive_reward = 0
        for i in range(self.current_starting_position[1], (self.current_starting_position[1] + self.demanded_slots)):
            if self.reward_matrix[self.current_starting_position[0]][i] > 0:
                total_positive_reward += self.reward_matrix[self.current_starting_position[0]][i]
            total_reward += self.reward_matrix[self.current_starting_position[0]][i]
            self.reward_matrix[self.current_starting_position[0]][i] = -0.01
        self.remaining_reward_on_matrix -= total_positive_reward
        return total_reward

    def dewrite_from_auxiliary_matrix(self):
        for i in range(self.current_starting_position[1], (self.current_starting_position[1] + self.demanded_slots)):
            self.current_placement_attempt_matrix[self.current_starting_position[0]][i] -= 2

    def get_random_position(self, number_of_cores=7):
        # row = random.randint(0, self.cores - 1)
        # max_start_col = self.slots - self.demanded_slots
        # column = random.randint(0, max_start_col)
        # return row, column
        if number_of_cores == 7:
            non_adjacent_cores_set = [0, 2, 4]
        else:
            raise KeyError("Not a valid number of cores")
        row = random.choice(non_adjacent_cores_set)

        return row, 0

    def count_overlaps(self):
        overlaps = 0
        for i in range(self.current_starting_position[1], (self.current_starting_position[1] + self.demanded_slots)):
            if self.spectrum[self.current_starting_position[0]][i]:
                overlaps += 1
        return overlaps

    def get_first_overlap(self):
        for i in range(self.current_starting_position[1], (self.current_starting_position[1] + self.demanded_slots)):
            if self.spectrum[self.current_starting_position[0]][i]:
                return self.current_starting_position[0], i
        raise KeyError("Can only be used when there are overlaps!")

    def get_last_overlap(self):
        a, b = -1, -1
        for i in range(self.current_starting_position[1], (self.current_starting_position[1] + self.demanded_slots)):
            if self.spectrum[self.current_starting_position[0]][i]:
                a, b = self.current_starting_position[0], i
        if a != -1:
            return a, b
        else:
            raise KeyError("Can only be used when there are overlaps!")

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

    def move_starting_position(self, action):

        if action == 0:
            # Up
            self.current_starting_position[0] -= 1
        elif action == 1:
            # Down
            self.current_starting_position[0] += 1
        elif action == 2:
            # Left
            self.current_starting_position[1] -= 1
        elif action == 3:
            # Right
            self.current_starting_position[1] += 1

    def move_starting_position_across_pivot(self, action):
        if action == 0:
            # Up
            self.current_starting_position[0] -= 1
        elif action == 1:
            # Down
            self.current_starting_position[0] += 1
        elif action == 2:
            # Left
            self.current_starting_position[1] = self.candidate_pivot[1] - self.demanded_slots
        elif action == 3:
            self.current_starting_position[1] = self.candidate_pivot[1] + 1

    def is_one_step_translation_possible(self, action):

        if action == 0:
            # Up
            if self.current_starting_position[0] >= 1:
                return True
            else:
                return False
        elif action == 1:
            # Down
            if self.current_starting_position[0] <= self.cores - 2:
                return True
            else:
                return False
        elif action == 2:
            # Left
            if self.current_starting_position[1] >= 1:
                return True
            else:
                return False
        elif action == 3:
            # Right
            if self.current_starting_position[1] + self.demanded_slots <= self.slots - 2:
                return True
            else:
                return False

    def is_pivot_translation_possible(self, action):

        if action == 0:
            # Up
            if self.candidate_pivot[0] >= 1:
                return True
            else:
                return False
        elif action == 1:
            # Down
            if self.candidate_pivot[0] <= self.cores - 2:
                return True
            else:
                return False
        elif action == 2:
            # Left
            if self.candidate_pivot[1] - self.demanded_slots >= 0:
                return True
            else:
                return False
        elif action == 3:
            # Right
            if self.candidate_pivot[1] + self.demanded_slots < self.slots:
                return True
            else:
                return False

    def build_reward_matrix(self, number_of_cores):

        if number_of_cores == 7:
            non_adjacent_cores_set = [0, 2, 4]
        else:
            raise KeyError("Not a valid number of cores")

        # First-time rewards. Once collected, will default to a negative value.
        #Counts total positive reward
        total_positive_reward = 0

        for row in range(len(self.spectrum)):
            for col in range(len(self.spectrum[0])):
                if self.spectrum[row][col] and row in non_adjacent_cores_set:
                    self.reward_matrix[row][col] = 0
                elif not self.spectrum[row][col] and row in non_adjacent_cores_set:
                    self.reward_matrix[row][col] = 0.1
                    total_positive_reward += 0.1
                elif not self.spectrum[row][col] and row not in non_adjacent_cores_set:
                    self.reward_matrix[row][col] = 0.05
                    total_positive_reward += 0.05
                elif self.spectrum[row][col] and row not in non_adjacent_cores_set:
                    self.reward_matrix[row][col] = -0.01
                else:
                    raise KeyError("Logic error somewhere")
        #sets total positive reward
        self.remaining_reward_on_matrix = total_positive_reward

    def generate_probabilistic_number(self):
        """
        Generates a number between 1 and 80 with a probabilistic distribution.
        Lower numbers are more likely, but larger numbers still have a chance.

        Returns:
            int: A number from 1 to 80.
        """
        numbers = np.arange(5, 81)  # Possible values from 5 to 80
        probabilities = np.exp(-0.05 * (numbers - 1))  # Exponential decay distribution
        probabilities /= probabilities.sum()  # Normalize to sum to 1

        return np.random.choice(numbers, p=probabilities)

    def generate_blocky_binary_matrix(self, cores, slots, max_fill_prob, max_attempts):
        matrix = np.zeros((cores, slots), dtype=int)

        # Favor lower fill ratios using exponential decay
        fill_ratio = np.random.exponential(scale=max_fill_prob)
        fill_ratio = min(fill_ratio, 1.0)

        total_ones = int(fill_ratio * cores * slots)
        ones_placed = 0
        attempts = 0

        while ones_placed < total_ones and attempts < max_attempts:
            attempts += 1

            # Choose a random block size
            block_size = random.randint(2, min(80, slots))
            if total_ones - ones_placed < block_size:
                block_size = total_ones - ones_placed

            if block_size < 2:
                break  # Remaining ones are too few for a valid block

            # Pick a random row
            row = random.randint(0, cores - 1)

            # Find all valid placements in that row
            possible_starts = [
                start for start in range(slots - block_size + 1)
                if np.all(matrix[row, start:start + block_size] == 0)
            ]

            if not possible_starts:
                continue  # Try a different row or block size

            # Place the block
            start = random.choice(possible_starts)
            matrix[row, start:start + block_size] = 1
            ones_placed += block_size

        return matrix

    def save_matrix_log(self, matrix, filename_prefix, step):
        """
        Appends a matrix and its metadata to a text file for later rendering.

        Args:
            matrix (np.ndarray): 2D integer array.
            filename_prefix (str): Identifier for the matrix (used for image name).
            step (int): Time step or sequence index.
        """
        if 9996000 < self.current_timestep < 1000000:
            log_file_path = "gif_images/matrix_log.txt"
        elif self.current_timestep > 9996000:
            log_file_path = "gif_images_end/matrix_log_end.txt"
        else:
            return True
        with open(log_file_path, 'a') as f:
            f.write(f"# {filename_prefix}_{step:04d}.png\n")
            np.savetxt(f, matrix, fmt='%d')
            f.write("\n")
