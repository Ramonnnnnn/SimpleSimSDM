import math
import Modulation


class LinkManager:
    def __init__(self):
        self.link_dict = {}
        self.Modulation = Modulation.Modulation()

    def add_link_info(self, id, mod, current_xt, distance):
        if id not in self.link_dict:
            self.link_dict[id] = {
                "mod": mod,
                "current_xt": current_xt,
                "distances": [distance]
            }
        else:
            self.link_dict[id]["distances"].append(distance)

    def get_mod(self, id):
        if id in self.link_dict:
            return self.link_dict[id]["mod"]
        else:
            raise KeyError(f"ID {id} does not exist.")

    def get_current_xt(self, id):
        if id in self.link_dict:
            return self.link_dict[id]["current_xt"]
        else:
            raise KeyError(f"ID {id} does not exist.")

    def get_distances(self, id):
        if id in self.link_dict:
            return self.link_dict[id]["distances"]
        else:
            raise KeyError(f"ID {id} does not exist.")

    def delete_link_info(self, id):
        if id in self.link_dict:
            del self.link_dict[id]
        else:
            raise KeyError(f"ID {id} does not exist.")

    def delete_all(self):
        self.link_dict = {}

    def project_xt(self, lightpath_ID):
        if lightpath_ID in self.link_dict:
            distances = self.get_distances(lightpath_ID)
            aux = 0
            for distance in distances:
                aux += 1 * 10 ** -9 * distance
            projected_xt = self.add_decibel(self.get_current_xt(lightpath_ID), self.linear_to_decibel(aux))
        return projected_xt

    def can_allocate_for_mod(self):
        for lightpath_ID in self.link_dict.keys():
            if self.project_xt(lightpath_ID) > self.Modulation.xt_threshold(self.get_mod(lightpath_ID)):
                return False
        else:
            return True

    @staticmethod
    def linear_to_decibel(value):
        return 10 * math.log10(value)
    @staticmethod
    def add_decibel(v1, v2):
        return 10 * math.log10(math.pow(10, (v1 / 10)) + math.pow(10, (v2 / 10)))


# # Example usage
# link_manager = LinkManager()
#
# # Add link information
# link_manager.add_link_info(1, 16, -50, 100)
# link_manager.add_link_info(1, 16, -50, 150)
# link_manager.add_link_info(2, 64, -40, 200)
#
# # Retrieve individual fields
# print(link_manager.get_mod(1))  # Output: 16
# print(link_manager.get_current_xt(1))  # Output: -50
# print(link_manager.get_distances(1))  # Output: [100, 150]
# print(link_manager.get_mod(2))  # Output: 64
# print(link_manager.get_current_xt(2))  # Output: -40
# print(link_manager.get_distances(2))  # Output: [200]
#
# # Delete link information
# link_manager.delete_link_info(1)
#
# # Try to retrieve the deleted link information
# try:
#     print(link_manager.get_mod(1))
# except KeyError as e:
#     print(e)  # Output: ID 1 does not exist.
