import MF
import MMM
import SHDLS
matrix = [
    [0, 0, 1, 0, 0, 0],
    [0, 1, 1, 1, 0, 0],
    [0, 0, 0, 1, 1, 0],
    [0, 0, 1, 0, 0, 0]
]

MF = MF.MeenyFirst(matrix)
MF.find_connected_components()
list_of_allocable_regions = MF.get_connected_components()



print(list_of_allocable_regions)