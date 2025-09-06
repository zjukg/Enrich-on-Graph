from collections import defaultdict

class Graph:
    def __init__(self):
        # Use dictionary to represent adjacency list, key is entity, value is a list where each element is a (relation, target entity) tuple
        self.adjacency_list = defaultdict(list)

    def add_triplet(self, triplet):
        """Add triplet"""
        if len(triplet) != 3:
            raise ValueError("Triplet length must be 3")
        subject, relation, obj = triplet
        # Add triplet information to adjacency list
        self.adjacency_list[subject].append((relation, obj))

    def get_one_hop_paths(self, topics):
        """Given one or more topic entities, identify all 1-hop paths"""
        one_hop_paths = []
        # Check if any topic appears in the adjacency list
        has_topic_in_graph = any(topic in self.adjacency_list for topic in topics)

        if has_topic_in_graph:
            # If topics exist in adjacency list, traverse 1-hop paths of specified topics
            for topic in topics:
                if topic in self.adjacency_list:
                    # Traverse all relations and target entities connected to this entity
                    for relation, obj in self.adjacency_list[topic]:
                        one_hop_paths.append([topic, relation, obj])
        else:
            # If no topic appears in adjacency list, traverse 1-hop paths of all entities
            for subject, relations in self.adjacency_list.items():
                for relation, obj in relations:
                    one_hop_paths.append([subject, relation, obj])

        return one_hop_paths

    def get_two_hop_paths(self, topics):
        """Given one or more topic entities, identify all 2-hop paths"""
        two_hop_paths = []
        # Check if any topic appears in the adjacency list
        has_topic_in_graph = any(topic in self.adjacency_list for topic in topics)

        if has_topic_in_graph:
            # If topics exist in adjacency list, traverse 2-hop paths of specified topics
            for topic in topics:
                if topic in self.adjacency_list:
                    # Traverse all relations and target entities connected to this entity
                    for relation1, intermediate in self.adjacency_list[topic]:
                        if intermediate in self.adjacency_list:
                            # Traverse all relations and target entities of intermediate entity
                            for relation2, obj in self.adjacency_list[intermediate]:
                                two_hop_paths.append([[topic, relation1, intermediate], 
                                                      [intermediate, relation2, obj]])
        else:
            # If no topic appears in adjacency list, traverse 2-hop paths of all entities
            for subject, relations in self.adjacency_list.items():
                for relation1, intermediate in relations:
                    if intermediate in self.adjacency_list:
                        for relation2, obj in self.adjacency_list[intermediate]:
                            two_hop_paths.append([[subject, relation1, intermediate], 
                                                  [intermediate, relation2, obj]])

        return two_hop_paths

# Usage example
if __name__ == "__main__":
    # Create graph
    graph = Graph()

    # Add triplets
    graph.add_triplet(["e1", "r1", "e2"])
    graph.add_triplet(["e2", "r2", "e3"])
    graph.add_triplet(["e1", "r3", "e4"])
    graph.add_triplet(["e4", "r4", "e5"])

    # Query 1-hop paths
    one_hop = graph.get_one_hop_paths(["e1", "e4"])
    print("1-hop paths:", one_hop)

    # Query 2-hop paths
    two_hop = graph.get_two_hop_paths(["e1"])
    print("2-hop paths:", two_hop)