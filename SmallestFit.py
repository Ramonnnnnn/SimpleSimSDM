class SmallestFit:
    def __init__(self, matrix):
        self.matrix = matrix
        self.rows = len(matrix)
        self.cols = len(matrix[0])
        self.regions = {}

    def find_connected_components(self):
        for i in range(self.rows):
            for j in range(self.cols):
                if self.matrix[i][j] == 0:
                    self.process_zero(i, j)

    def process_zero(self, row, col):

        if col == 0:
            region_id = len(self.regions) + 1
            self.regions[region_id] = [(row,col)]
        if col > 0 and self.matrix[row][col - 1] != 0:
            # New region
            region_id = len(self.regions) + 1
            self.regions[region_id] = [(row, col)]

        elif col > 0 and self.matrix[row][col - 1] == 0:
            for id, coordinates_list in self.regions.items():
                if (row,col-1) in coordinates_list:
                    region_id = id

            # Belongs to the same region as the previous zero
            self.regions[region_id].append((row, col))

        # # Add current zero's coordinates to the region
        # self.regions[region_id].append((row, col))
        # # Update the matrix to mark the region
        # self.matrix[row][col] = region_id

    def get_connected_components(self):
        sorted_regions = sorted(self.regions.items(), key=lambda x: len(x[1]))
        return dict(sorted_regions)


