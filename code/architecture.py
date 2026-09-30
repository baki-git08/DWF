"""
GENERAL MULTI-NODE ENERGY HARVESTING — LOCATION AGNOSTIC
Master (solar) + Node 1 (thermal) + Node 2 (wave) + mobile Slave
"""

class Node:
    """Generic harvesting node — location agnostic."""
    def __init__(self, name, source, depth_m, harvest_fn, capacity_kj):
        self.name = name
        self.source = source
        self.depth_m = depth_m
        self.harvest = harvest_fn         # function(season) -> kJ/day
        self.capacity_kj = capacity_kj
        self.battery_kj = 0.0
        self.message_queue = []

    def daily_charge(self, season):
        self.battery_kj = min(self.capacity_kj,
                              self.battery_kj + self.harvest(season))

    def deliver_energy_to_slave(self, requested_kj):
        given = min(self.battery_kj, requested_kj)
        self.battery_kj -= given
        return given

    def enqueue_message(self, msg):
        self.message_queue.append(msg)

    def handover_messages(self):
        msgs = self.message_queue.copy()
        self.message_queue.clear()
        return msgs


class Slave:
    """Mobile energy mule + messenger."""
    def __init__(self, capacity_kj, travel_cost_kj_per_km):
        self.battery_kj = 0.0
        self.capacity_kj = capacity_kj
        self.travel_cost = travel_cost_kj_per_km

    def travel(self, distance_km):
        cost = distance_km * self.travel_cost
        self.battery_kj = max(0.0, self.battery_kj - cost)
        return cost

    def charge_from(self, node, requested_kj):
        return node.deliver_energy_to_slave(requested_kj)

    def carry_messages(self, nodes):
        all_msgs = []
        for n in nodes:
            all_msgs.extend(n.handover_messages())
        return all_msgs


class System:
    """Full general architecture."""
    def __init__(self, master, node1, node2, slave, distances_km):
        self.master = master
        self.node1 = node1
        self.node2 = node2
        self.slave = slave
        self.distances = distances_km    # dict of pair → km

    def run_day(self, season):
        # 1. Charge all nodes
        self.master.daily_charge(season)
        self.node1.daily_charge(season)
        self.node2.daily_charge(season)

        # 2. Slave visits nodes in priority order
        for node in [self.master, self.node1, self.node2]:
            self.slave.travel(self.distances[('master', node.name)])
            self.slave.charge_from(node, requested_kj=100)

        # 3. Collect messages and return
        msgs = self.slave.carry_messages([self.node1, self.node2])
        self.slave.travel(self.distances[('slave', 'master')])
        for m in msgs:
            self.master.message_queue.append(m)

        return self.slave.battery_kj, len(msgs)