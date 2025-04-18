
class Modulation:
    def __init__(self):
        # Source: Revisiting the modulation format selection problem in crosstalk-aware SDM-EONs
        # BPSK, QPSK, 8-QAM, 16-QAM, 32-QAM, 64-QAM
        self.distances = (6300,3500,1200,600,250,125)

    @staticmethod
    def xt_threshold(modulation):
        # BPSK, QPSK, 8-QAM, 16-QAM, 32-QAM, 64-QAM
        thresholds = [-21.7, -26.2, -28.7, -32.7]
        thresholds = [-14.0, -18.0, -21.0, -25.0, -27.0, -34.0]# According to Super-channel oriented routing, spectrum and core assignment under crosstalk limit in spatial division multiplexing elastic optical networks, 2017
        if 0 <= modulation <= 5:
            return thresholds[modulation]
        else:
            return thresholds[5]

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
        elif self.distances[1] < givendistance <= self.distances[0]:
            return 0
        elif givendistance > self.distances[0]:
            return -1

    def get_bandwidth(self, modulation_level):
        # BPSK, QPSK, 8-QAM, 16-QAM, 32-QAM, 64-QAM
        switch = {
            0: 12.5,
            1: 25.0,
            2: 37.5,
            3: 50.0,
            4: 62.5,
            5: 75.0,
        }
        return switch.get(modulation_level, 0)


    def aggregate_intercore_crosstalk(cores):
        if cores <= 7:
            return -45
        elif 7 < cores <= 12:
            return -31
        elif cores > 12:
            return -20
        else:
            return 1000

    #CHECK VALUES: https://www.researchgate.net/publication/230765698_Penalties_from_In-Band_Crosstalk_for_Advanced_Optical_Modulation_Formats
    def in_band_xt(modulation):
        values = [-7, -9, -13, -15, -17, -19]
        return values[modulation]

    # CHECK VALUES
    def snr_threshold(modulation):
        #SNR_THRESHOLD[] = {4.2,7.2,13.9,19.8};
        thresholds = [4.2,7.2,13.9,19.8]
        if 0 <= modulation <= 3:
            return thresholds[modulation]
        else:
            return thresholds[0]






###### THIS ONE IS BASED ON https://www.sciencedirect.com/science/article/abs/pii/S1389128622005588
# class Modulation:
#     def __init__(self):
#         #                  BPSK QPSK 8-QAM 16-QAM
#         self.distances = (6300, 3500, 1200, 600 )
#
#
#     def get_modulation_by_distance(self, givendistance):
#         if givendistance == -1:
#             return -1
#         if givendistance <= self.distances[3]:
#             return 3
#         elif self.distances[3] < givendistance <= self.distances[2]:
#             return 2
#         elif self.distances[2] < givendistance <= self.distances[1]:
#             return 1
#         elif self.distances[1] < givendistance <= self.distances[0]:
#             return 0
#         else:
#             return -1
#
#     def get_bandwidth(self, modulation_level):
#         switch = {
#             -1: -1,
#             0: 12.5,
#             1: 25.0,
#             2: 33.3,
#             3: 50.0,
#         }
#         return switch.get(modulation_level, 0)
#
#
#     def aggregate_intercore_crosstalk(self, cores):
#         if cores <= 7:
#             return -45
#         elif 7 < cores <= 12:
#             return -31
#         elif cores > 12:
#             return -20
#         else:
#             return 1000
#
#     #CHECK VALUES: https://www.researchgate.net/publication/230765698_Penalties_from_In-Band_Crosstalk_for_Advanced_Optical_Modulation_Formats
#     def in_band_xt(modulation):
#         values = [-7, -9, -13, -15, -17, -19]
#         return values[modulation]
#
#     #CHECK VALUES
    # Source: Revisiting the modulation format selection problem in crosstalk-aware SDM-EONs
