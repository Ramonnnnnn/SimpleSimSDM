from collections import defaultdict

class FcaRcsa:

    def __init__(self, matrix, requiredFS):
        self.matrix = matrix
        self.requiredFS = requiredFS
        self.list_of_regions = {}
        self.nextKey = 321

    def update_dict(self, row, column, size):
        self.list_of_regions[self.nextKey] = []
        for j in range(size):
            self.list_of_regions[self.nextKey].append((row, column + j))
        self.nextKey += 1  # Increment for the next set of entries

    def output_dict(self):
        return self.list_of_regions


    def calculate_fragmentation(self, line):
        largest_frag = 0
        totalFreeSlotsInCore = 0
        sizeFrag = 0

        for column in range(len(self.matrix[line])):
            while column < len(self.matrix[line]) and not self.matrix[line][column]:
                sizeFrag += 1
                column += 1
                totalFreeSlotsInCore += 1

            if sizeFrag >= largest_frag:
                largest_frag = sizeFrag
            sizeFrag = 0  # Reset the fragment size counter

        # Calculate the fragmentation
        if largest_frag <= 1:
            return 1.0
        else:
            fragmentation = largest_frag / totalFreeSlotsInCore
            return 1.0 - fragmentation

    def start_index_exact_fit(self, core):
        numCols = len(self.matrix[0])
        fitSize = 0

        for column in range(numCols):
            if not self.matrix[core][column]:
                start = column
                while column < numCols and not self.matrix[core][column]:
                    column += 1
                    fitSize += 1

                if fitSize >= self.requiredFS:
                    return start

                fitSize = 0

        return -1  # Return -1 if no exact fit is found

    def frag_coef_rcsa(self):
        numLines = len(self.matrix)
        fragmentationValues = []
        lineIndexes = list(range(numLines))

        # Calculate fragmentation for each line
        for i in range(numLines):
            fragmentationValues.append(self.calculate_fragmentation(i))

        # Rank lines by fragmentation in ascending order
        lineIndexes.sort(key=lambda i: fragmentationValues[i])

        # Follow the ranking to update the dictionary
        for line in lineIndexes:
            startIndex = self.start_index_exact_fit(line)
            if startIndex != -1:
                self.update_dict(line, startIndex, self.requiredFS)



# # Example usage
# matrix = [
#     [False, True, False, False],
#     [False, False, False, False],
#     [True, True, True, True],
#     [False, True, False, False]
# ]
# requiredFS = 2
# fca_rcsa = FcaRcsa(matrix, requiredFS)
# result = fca_rcsa.frag_coef_rcsa()
#
# print(result)
