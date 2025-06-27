import math
import random
import gymnasium as gym
from gymnasium import spaces
import numpy as np


class TetrisResourceAllocation(gym.Env):
    def __init__(self, cores, slots, max_episode_length, total_timesteps, reference_to_allocator):
        super(TetrisResourceAllocation, self).__init__()

        self.demanded_slots_int = None # ADD TO TEST
        self.demanded_slots = None # ADD TO TEST
        self.allocator = reference_to_allocator # ADD TO TEST
        # Spectrum Dimension
        self.current_starting_position = None
        self.slots = slots
        self.cores = cores
        # Define the action space: selecting a row and a starting column
        self.action_space = self.action_space = spaces.Discrete(4)  # up, down, left, right

        # Define the observation space: spectrum matrix and metrics
        self.observation_space = spaces.Dict({
            "spectrum": spaces.Box(low=0, high=1, shape=(cores * slots,), dtype=np.int32),  # path spectrum
            "demanded_slots": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.int32),
            "current_placement_attempt_matrix": spaces.Box(low=0, high=1, shape=(cores * slots,), dtype=np.int32),

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
        self.spectrum = np.zeros((cores, slots), dtype=np.int32)
        self.current_placement_attempt_matrix = np.ones((cores, slots), dtype=np.int32)  # All ones


    def reset(self, seed=None, **kwargs):
        """Reset the environment at the start of an episode."""
        super().reset(seed=seed)  # Call parent class's reset method to handle seeding
        if seed is not None:
            np.random.seed(seed)  # Update NumPy random state
        # Count episodes (universal)
        self.current_episode += 1
        # Counter (resets for episode)
        self.this_episodes_timestep = 0
        # Get current attempt's spectrum
        self.spectrum = self.allocator.spectrum_current_attempt  # ADD TO TEST
        # Demanded Slots (probabilistic distribution for training)
        self.demanded_slots = np.array([int(self.allocator.demanded_slots_current_attempt)], dtype=np.int32) # ADD TO TEST
        self.demanded_slots_int = self.allocator.demanded_slots_current_attempt # ADD TO TEST
        # Starting Coordinate
        r, c = self.get_random_position()
        self.current_starting_position = [r, c]  # Always a random within-bounds position for our block
        # Mirror matrix for mock placement for step 0
        self.current_placement_attempt_matrix = self.spectrum.copy()  # Independent copy of the spectrum
        # Write candidate block to auxiliary matrix
        self.write_to_auxiliary_matrix()
        # Initialize state
        self.state = {
            "spectrum": self.spectrum.flatten(),
            "demanded_slots": self.demanded_slots,
            "current_placement_attempt_matrix": self.current_placement_attempt_matrix.flatten(),
        }
        # Return observation and an empty info dictionary
        return self.state, {}

    def step(self, action):
        """Take an action and compute the next state, reward, and done flag."""
        # Iterate counters
        self.this_episodes_timestep += 1
        self.current_timestep += 1
        # Count overlaps:
        overlaps, overshadows = self.count_overlaps_and_overshadows()
        # Call action to deal with movement
        # Is action valid?
        action = action[0] # ADD TO TEST
        if self.is_translation_possible(action):
            # Dewrite from auxiliary matrix
            self.dewrite_from_auxiliary_matrix()
            # Then move
            self.move_starting_position(action)
            # Write new position to auxiliary matrix
            self.write_to_auxiliary_matrix()
            # Count new position's overlaps/overshadows
            new_overlaps, new_overshadows = self.count_overlaps_and_overshadows()
            if new_overlaps == 0:
                # print(f"Ended in success at step {self.this_episodes_timestep}")
                self.allocator.rl_list_of_regions = {self.current_episode: self.write_in_region_form()} # ADD TO TEST
                self.write_to_spectrum()
                self.state = {
                    "spectrum": self.spectrum.flatten(),
                    "demanded_slots": self.demanded_slots,
                    "current_placement_attempt_matrix": self.current_placement_attempt_matrix.flatten(),
                }
                reward = self.demanded_slots_int
                return self.state, reward, True, True, {}
            elif self.this_episodes_timestep == self.max_episode_length:
                # print("Could not place in the spectrum until the end of the episode")
                self.state = {
                    "spectrum": self.spectrum.flatten(),
                    "demanded_slots": self.demanded_slots,
                    "current_placement_attempt_matrix": self.current_placement_attempt_matrix.flatten(),
                }
                reward = 0
                return self.state, reward, True, True, {}
            else:
                self.state = {
                    "spectrum": self.spectrum.flatten(),
                    "demanded_slots": self.demanded_slots,
                    "current_placement_attempt_matrix": self.current_placement_attempt_matrix.flatten(),
                }
                reward = overlaps - new_overlaps
                return self.state, reward, False, False, {}
        # If invalid action
        else:
            if self.this_episodes_timestep == self.max_episode_length:
                # print("Could not place in the spectrum until the end of the episode")
                self.state = {
                    "spectrum": self.spectrum.flatten(),
                    "demanded_slots": self.demanded_slots,
                    "current_placement_attempt_matrix": self.current_placement_attempt_matrix.flatten(),
                }
                reward = 0
                return self.state, reward, True, True, {}
            else:
                self.state = {
                    "spectrum": self.spectrum.flatten(),
                    "demanded_slots": self.demanded_slots,
                    "current_placement_attempt_matrix": self.current_placement_attempt_matrix.flatten(),
                }
                reward = 0
                return self.state, reward, False, False, {}

    # ADD TO TEST
    def write_in_region_form(self):
        region = []
        for i in range(self.current_starting_position[1], (self.current_starting_position[1] + self.demanded_slots_int)): # ADD TO TEST (MAKE _int)
            region.append((self.current_starting_position[0], i))
        return region

    def write_to_spectrum(self):
        for i in range(self.current_starting_position[1], (self.current_starting_position[1] + self.demanded_slots_int)):
            if self.spectrum[self.current_starting_position[0]][i] == 1:
                raise KeyError("Should not happen")
            else:
                self.spectrum[self.current_starting_position[0]][i] = 1

    def write_to_auxiliary_matrix(self):
        for i in range(self.current_starting_position[1], (self.current_starting_position[1] + self.demanded_slots_int)):
            self.current_placement_attempt_matrix[self.current_starting_position[0]][i] += 1

    def dewrite_from_auxiliary_matrix(self):
        for i in range(self.current_starting_position[1], (self.current_starting_position[1] + self.demanded_slots_int)):
            self.current_placement_attempt_matrix[self.current_starting_position[0]][i] -= 1

    def get_random_position(self):
        row = random.randint(0, self.cores - 1)
        max_start_col = self.slots - self.demanded_slots_int
        column = random.randint(0, max_start_col)
        return row, column

    def count_overlaps_and_overshadows(self):
        overlaps = 0
        overshadows = 0
        for i in range(self.current_starting_position[1], (self.current_starting_position[1] + self.demanded_slots_int)):
            is_overshadowed = False
            if self.spectrum[self.current_starting_position[0]][i]:
                overlaps += 1
            for j in self.figure_neighbors_seven_core_MCF(self.current_starting_position[0]):
                if self.spectrum[j][i]:
                    is_overshadowed = True
            if is_overshadowed:
                overshadows += 1
        return overlaps, overshadows

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

    def is_translation_possible(self, action):

        if action == 0:
            # Up
            if self.current_starting_position[0] >= 1:
                #print("UP")
                return True
            else:
                #print("not_up")
                return False
        elif action == 1:
            # Down
            if self.current_starting_position[0] <= self.cores - 2:
                #print("Down")
                return True
            else:
                #print("Not Down")
                return False
        elif action == 2:
            # Left
            if self.current_starting_position[1] >= 1:
                #print("Left")
                return True
            else:
                #print("Not Left")
                return False
        elif action == 3:
            # Right
            if self.current_starting_position[1] + self.demanded_slots_int <= self.slots - 2:
                #print("Right")
                return True
            else:
                #print("Not Right")
                return False

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
