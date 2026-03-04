
import os
import random
import gymnasium as gym
from gymnasium import spaces
import numpy as np



class TetrisResourceAllocation(gym.Env):
    ''' In this environment, we added the set in spectrum action (4)'''

    def __init__(self, cores, slots, max_episode_length, total_timesteps, reference_to_allocator):
        super(TetrisResourceAllocation, self).__init__()


        # ADD TO TEST -> Remove plotting. Only necessary in training
        self.demanded_slots_int = None # ADD TO TEST
        self.demanded_slots = None # ADD TO TEST
        self.allocator = reference_to_allocator # ADD TO TEST

        # Spectrum Dimension
        self.candidate_pivot = None
        self.demanded_slots = None
        self.current_starting_position = None
        self.slots = slots
        self.cores = cores
        # Define the action space: selecting a row and a starting column
        self.action_space = self.action_space = spaces.Discrete(2)  # move ahead, set in spectrum

        # Define the observation space: spectrum matrix and metrics
        self.observation_space = spaces.Dict({
            "current_placement_attempt_matrix": spaces.Box(low=0, high=4, shape=(cores * slots,), dtype=np.int32),
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

        # Demanded Slots (from simulator)
        self.demanded_slots = np.array([int(self.allocator.demanded_slots_current_attempt)], dtype=np.int32)  # ADD TO TEST
        self.demanded_slots_int = self.allocator.demanded_slots_current_attempt  # ADD TO TEST
        # Get new spectrum
        self.spectrum = self.allocator.spectrum_current_attempt  # ADD TO TEST
        # Starting Coordinate
        self.current_starting_position = [0, 0]  # Start position for our block
        # Alternative random start
        # row, col = self.get_random_position()
        # self.current_starting_position = [row, col]
        self.candidate_pivot = None
        # Free spectrum for step 0
        self.current_placement_attempt_matrix = self.spectrum.copy()  # Independent copy of the spectrum
        # Set total positive reward for future comparison
        self.total_positive_reward_on_matrix = self.build_reward_matrix(self.cores)
        # Set remaining positive reward
        self.remaining_reward_on_matrix = self.total_positive_reward_on_matrix
        # Write candidate block to auxiliary matrix
        self.write_to_auxiliary_matrix()
        # Initialize state
        self.state = {
            "current_placement_attempt_matrix": self.current_placement_attempt_matrix.flatten(),
        }
        # Return observation and an empty info dictionary
        return self.state, {}

    def step(self, action):
        """Take an action and compute the next state, reward, and done flag."""
        move_ahead = 0
        set_lp = 1
        # Iterate counters
        self.this_episodes_timestep += 1
        self.current_timestep += 1
        # Check if within episode timestep limit
        if self.this_episodes_timestep == self.max_episode_length:
            # print(f"Could not place in the spectrum until the end of the episode.")
            # Save to Allocator
            self.allocator.rl_list_of_regions = {} #ADD TO TEST
            self.state = {"current_placement_attempt_matrix": self.current_placement_attempt_matrix.flatten(),}
            reward = -1
            return self.state, reward, True, True, {}

        # Deal with actions

        # Placement action
        if action == set_lp and self.count_overlaps() == 0:
            # Good action
            #print(f"Ended in success at step {self.this_episodes_timestep}")
            # Updates the actual spectrum
            self.write_to_spectrum()
            # Write final valid position to auxiliary matrix (current placement matrix)
            self.success_write_to_auxiliary_matrix()
            # Matrix for visualization
            # self.save_matrix_log(self.current_placement_attempt_matrix, "action", self.current_timestep) # ADD comment from test
            # State
            self.state = {
                "current_placement_attempt_matrix": self.current_placement_attempt_matrix.flatten(),
            }
            # rew  = - remaining timesteps % + 1 + non_fragmented_bonus (max 0.1) + reward for specific position/max_reward for that core
            reward = ((-self.this_episodes_timestep/self.max_episode_length) + 1) + self.is_non_fragmented_bonus() + (self.collect_reward()/self.demanded_slots_int)
            # Save to allocator
            self.allocator.rl_list_of_regions = {self.current_episode: self.write_in_region_form()}  # ADD TO TEST
            return self.state, reward, True, True, {}
        if action == set_lp and self.count_overlaps() > 0:
            # Bad/invalid action
            #print(f"Could not set {self.current_starting_position, self.demanded_slots}, because of overlaps")
            self.state = {
                "current_placement_attempt_matrix": self.current_placement_attempt_matrix.flatten(),
            }
            reward = -1
            return self.state, reward, False, False, {}

        # Move ahead urges movement forward according to the one step or pivot rules
        # It is always possible to move 'forward', even if it requires looping back to the start
        if action == move_ahead:
            # To move past, we move past the furthest overlap
            if self.count_overlaps() > 0:
                row, col = self.get_last_overlap()
                self.candidate_pivot = [row, col]
            else:
                self.candidate_pivot = None

            # Dewrite from auxiliary matrix
            self.dewrite_from_auxiliary_matrix()
            # Then move - It moves one step at a time or pivot style, whenever possible
            self.move_ahead()
            # Write new position to auxiliary matrix (current placement matrix)
            self.write_to_auxiliary_matrix()
            # Collect Reward
            reward = 0
            # Save for visualization
            # self.save_matrix_log(self.current_placement_attempt_matrix, "action", self.current_timestep) # ADD comment from test
            # Update and return State
            self.state = {"current_placement_attempt_matrix": self.current_placement_attempt_matrix.flatten(), }
            return self.state, reward, False, False, {}



    def write_to_spectrum(self):
        for i in range(self.current_starting_position[1], (self.current_starting_position[1] + self.demanded_slots_int)):
            if self.spectrum[self.current_starting_position[0]][i] == 1:
                raise KeyError("Should not happen")
            else:
                self.spectrum[self.current_starting_position[0]][i] = 1

    def write_to_auxiliary_matrix(self):
        for i in range(self.current_starting_position[1], (self.current_starting_position[1] + self.demanded_slots_int)):
            self.current_placement_attempt_matrix[self.current_starting_position[0]][i] += 2
    def success_write_to_auxiliary_matrix(self):
        for i in range(self.current_starting_position[1], (self.current_starting_position[1] + self.demanded_slots_int)):
            self.current_placement_attempt_matrix[self.current_starting_position[0]][i] = 1
    def dewrite_from_auxiliary_matrix(self):
        for i in range(self.current_starting_position[1], (self.current_starting_position[1] + self.demanded_slots_int)):
            self.current_placement_attempt_matrix[self.current_starting_position[0]][i] -= 2

    def collect_reward(self):
        total_reward = 0
        for i in range(self.current_starting_position[1], (self.current_starting_position[1] + self.demanded_slots_int)):
            # Sums reward for every cell
            total_reward += self.reward_matrix[self.current_starting_position[0]][i]
            # zeroes collected reward
            self.reward_matrix[self.current_starting_position[0]][i] = 0
        return total_reward


    def get_random_position(self):
        if self.cores == 7:
            non_adjacent_cores_set = [0, 2, 4]
        else:
            raise KeyError("Not a valid number of cores")
        row = random.choice(non_adjacent_cores_set)

        return row, 0

    def count_overlaps(self):
        overlaps = 0
        for i in range(self.current_starting_position[1], (self.current_starting_position[1] + self.demanded_slots_int)):
            if self.spectrum[self.current_starting_position[0]][i]:
                overlaps += 1
        return overlaps

    def get_first_overlap(self):
        for i in range(self.current_starting_position[1], (self.current_starting_position[1] + self.demanded_slots_int)):
            if self.spectrum[self.current_starting_position[0]][i]:
                return self.current_starting_position[0], i
        raise KeyError("Can only be used when there are overlaps!")

    def get_last_overlap(self):
        a, b = -1, -1
        for i in range(self.current_starting_position[1], (self.current_starting_position[1] + self.demanded_slots_int)):
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


    def figure_next_core(self):
        core = self.current_starting_position[0]
        if self.cores == 7:
            if core == 6:
                return 0
            else:
                return core+1
        else:
            raise KeyError("Not a valid number of cores")



    def move_ahead(self):
        # Can only move one cell per timestep
        if self.candidate_pivot is None:
            # if there is enough space to the right, move within same core
            if (self.current_starting_position[1] + self.demanded_slots_int) < self.slots:
                self.current_starting_position[1] += 1
            # else move to start of next core
            else:
                self.current_starting_position = [self.figure_next_core(),0]
        # Moves across pivot
        else:
            # if there is enough space to the right, move within same core
            if (self.candidate_pivot[1] + self.demanded_slots_int) < self.slots:
                self.current_starting_position[1] = self.candidate_pivot[1] + 1
            # else move to start of next core
            else:
                self.current_starting_position = [self.figure_next_core(), 0]



    def build_reward_matrix(self, number_of_cores):

        if self.cores == 7:
            non_adjacent_cores_set = [0, 2, 4]
        else:
            raise KeyError("Not a valid number of cores")

        # First-time rewards. Once collected, will default to a negative value.
        #Counts total positive reward
        total_positive_reward = 0

        for row in range(len(self.spectrum)):
            for col in range(len(self.spectrum[0])):
                # occ and non-adjacent
                if self.spectrum[row][col] and row in non_adjacent_cores_set:
                    self.reward_matrix[row][col] = 0
                # free and non-adjacent
                elif not self.spectrum[row][col] and row in non_adjacent_cores_set:
                    self.reward_matrix[row][col] = 1
                    total_positive_reward += 1
                # free and adjacent
                elif not self.spectrum[row][col] and row not in non_adjacent_cores_set:
                    self.reward_matrix[row][col] = 0.5
                    total_positive_reward += 0.5
                # occ and adjacent
                elif self.spectrum[row][col] and row not in non_adjacent_cores_set:
                    self.reward_matrix[row][col] = 0
                else:
                    raise KeyError("Logic error somewhere")
        # returns total positive reward
        return total_positive_reward



    def is_non_fragmented_bonus(self):
        row, column = self.current_starting_position
        lp_end = column + self.demanded_slots_int -1
        if column == 0:
            return 1
        elif self.spectrum[row][column-1] == 1:
            return 1
        elif lp_end == self.slots - 1:
            return 1
        elif self.spectrum[row][column+1] == 1:
            return 1
        else:
            return -1

    def find_largest_region(self):
        largest_region = 0
        for row in self.spectrum:
            current_region = 0
            for slot in row:
                if slot == 0:  # free
                    current_region += 1
                    largest_region = max(largest_region, current_region)
                else:  # occupied
                    current_region = 0
        return largest_region

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
        base_dir = os.path.dirname(os.path.abspath(__file__))

        if (self.total_timesteps * 0.1) <= self.current_timestep <= (self.total_timesteps * 0.101):
            log_file_path = os.path.join(base_dir, "../gif_images/matrix_log.txt")
        elif self.current_timestep >= self.total_timesteps:
            log_file_path = os.path.join(base_dir, "../gif_images_end/matrix_log_end.txt")
        else:
            return True

        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)  # Create directory if missing

        with open(log_file_path, 'a') as f:
            f.write(f"# {filename_prefix}_{step:04d}.png\n")
            np.savetxt(f, matrix, fmt='%d')
            f.write("\n")

    def write_in_region_form(self):
        region = []
        for i in range(self.current_starting_position[1], (self.current_starting_position[1] + self.demanded_slots_int)):
            region.append((self.current_starting_position[0], i))
        return region
