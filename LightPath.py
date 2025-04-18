class Lightpath:
    def __init__(self):
        self.lightpaths = {}

    def add_lightpath(self, lp_id, path, region, modulation, required_fs):
        self.lightpaths[lp_id] = {
            "path": path,
            "source": path[0],
            "destination": path[-1],
            "occupied_slot_list": region[:required_fs],
            "modulation": modulation,
        }

    def get_attribute(self, lp_id, attribute):
        if lp_id in self.lightpaths:
            return self.lightpaths[lp_id].get(attribute)
        else:
            raise KeyError(f"Lightpath with id {lp_id} does not exist.")

    def set_attribute(self, lp_id, attribute, value):
        if lp_id in self.lightpaths:
            if attribute in self.lightpaths[lp_id]:
                self.lightpaths[lp_id][attribute] = value
            else:
                raise KeyError(f"Attribute '{attribute}' does not exist in lightpath with id {lp_id}.")
        else:
            raise KeyError(f"Lightpath with id {lp_id} does not exist.")

    def remove_lightpath(self, lp_id):
        if lp_id in self.lightpaths:
            del self.lightpaths[lp_id]
        else:
            raise KeyError(f"Lightpath with id {lp_id} does not exist.")
