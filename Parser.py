import xml.etree.ElementTree as ET

class XmlParser:
    def __init__(self, xml_file):
        self.nodes = []
        self.traffic_info = {}
        self.calls_info = []
        self.slots_bandwidth = None

        # Parse the XML file
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Extract nodes
        nodes_element = root.find(".//nodes")
        for node in nodes_element.iter("node"):
            node_id = node.attrib["id"]
            self.nodes.append(node_id)

        # Extract traffic information
        traffic_element = root.find(".//traffic")
        self.traffic_info['calls'] = int(traffic_element.attrib['calls'])
        self.traffic_info['load'] = int(traffic_element.attrib['load'])
        self.traffic_info['max_rate'] = int(traffic_element.attrib['max-rate'])

        # Extract calls information
        calls_info = []
        for call in traffic_element.iter("calls"):
            call_info = {
                'holding_time': float(call.attrib['holding-time']),
                'rate': int(call.attrib['rate']),
                'cos': int(call.attrib['cos']),
                'weight': int(call.attrib['weight'])
            }
            calls_info.append(call_info)
        self.calls_info = calls_info

        # Extract slots bandwidth
        physical_topology_element = root.find(".//physical-topology")
        self.slots_bandwidth = float(physical_topology_element.attrib.get('slotsBandwidth', 0))

    def get_nodes(self):
        return self.nodes

    def get_traffic_info(self):
        return self.traffic_info

    def get_calls_info(self):
        return self.calls_info

    def get_slots_bandwidth(self):
        return self.slots_bandwidth

