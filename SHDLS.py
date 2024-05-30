class Shadowless:
    def __init__(self, matrix):
        self.matrix = matrix
        self.rows = len(matrix)
        self.cols = len(matrix[0])
        self.row_regions = {}

    def find_connected_components(self):
        # First non-adjacent cores
        for i in range(self.rows):
            if i % 2 == 0:
                for j in range(self.cols):
                    if self.matrix[i][j] == 0:
                        self.process_zero(i, j)
        # Then adjacent cores
        for i in range(self.rows):
            if i % 2 != 0:
                for j in range(self.cols-1, -1, -1):
                    if self.matrix[i][j] == 0:
                        self.process_zero_inverse(i, j)

    def process_zero(self, row, col):
        if col == 0:
            region_id = len(self.row_regions[row]) + 1 if row in self.row_regions else 1
            if row not in self.row_regions:
                self.row_regions[row] = {}
            self.row_regions[row][region_id] = [(row, col)]
        if col > 0 and self.matrix[row][col - 1] != 0:
            # New region
            region_id = len(self.row_regions[row]) + 1 if row in self.row_regions else 1
            if row not in self.row_regions:
                self.row_regions[row] = {}
            self.row_regions[row][region_id] = [(row, col)]
        elif col > 0 and self.matrix[row][col - 1] == 0:
            for id, coordinates_list in self.row_regions[row].items():
                if (row, col - 1) in coordinates_list:
                    region_id = id
            # Belongs to the same region as the previous zero
            self.row_regions[row][region_id].append((row, col))

    def process_zero_inverse(self, row, col):
        # If first column (inverted)
        if col == self.cols - 1:
            region_id = len(self.row_regions[row]) + 1 if row in self.row_regions else 1
            if row not in self.row_regions:
                self.row_regions[row] = {}
            self.row_regions[row][region_id] = [(row, col)]
        if col < self.cols - 1 and self.matrix[row][col + 1] != 0:
            # New region
            region_id = len(self.row_regions[row]) + 1 if row in self.row_regions else 1
            if row not in self.row_regions:
                self.row_regions[row] = {}
            self.row_regions[row][region_id] = [(row, col)]
        elif col < self.cols - 1 and self.matrix[row][col + 1] == 0:
            for id, coordinates_list in self.row_regions[row].items():
                if (row, col + 1) in coordinates_list:
                    region_id = id
            # Belongs to the same region as the previous zero
            self.row_regions[row][region_id].append((row, col))

    def get_connected_components(self):
        sorted_regions = {}
        even_rows = [row for row in range(self.rows) if row % 2 == 0]
        odd_rows = [row for row in range(self.rows) if row % 2 != 0]
        region_id = 1
        for row in even_rows + odd_rows:
            if row in self.row_regions:
                sorted_row_regions = sorted(self.row_regions[row].values(), key=len)
                for region in sorted_row_regions:
                    sorted_regions[region_id] = region
                    region_id += 1
        return sorted_regions
