



class Modulation:
    def __init__(self):
        self.distances = (10000000,2000,1000,500,250,125)


    def get_modulation_by_distance(self, givendistance):
        if givendistance == -1:
            return -1
        if givendistance <= self.distances[5]:
            return 5
        elif self.distances[5] < givendistance <= self.distances[4]:
            return 4
        elif self.distances[4] < givendistance <= self.distances[3]:
            return 3
        elif self.distances[3] < givendistance <= self.distances[2]:
            return 2
        elif self.distances[2] < givendistance <= self.distances[1]:
            return 1
        elif givendistance > self.distances[1]:
            return 0
        else:
            return 0

    def get_bandwidth(self, modulation_level):
        switch = {
            0: 12.5,
            1: 25.0,
            2: 37.5,
            3: 50.0,
            4: 62.5,
            5: 75.0,
        }
        return switch.get(modulation_level, 0)