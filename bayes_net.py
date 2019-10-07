# -*- coding: utf-8 -*-
# import pydot_ng as pydot
import networkx as nx
import matplotlib.pyplot as plt


def almost_equal(val1, val2, epsilon=0.0001):
    return abs(val1 - val2) < epsilon

"""
Defines a BayesNode object for tracking marginal or conditional probability distributions.
"""

class BayesNode:
    def __init__(self, var_name):
        self.name = var_name
        # Maps from (parent, val) tuples to map from value to probability.
        # Parents are BayesNodes themselves.
        # E.g. set((A, True), (B, 9)) -> {1: 0.2, 2: 0.7, 3: 0.1}
        self.parents = []
        self.conditional_distribution = {}
        
        self.marginal_distribution = {}
        
        self.domain = set()
        
    # Defines the marginal distribution for a node, so there's no conditioning allowed. Explicitly
    # sets the domain as well.
    def set_marginal_distribution(self, distribution):
        assert not self.conditional_distribution, "Can't set distribution if already have conditional dist."
        self.marginal_distribution = distribution
        self.domain = distribution.keys()
    
    # Defines part of the conditional distribution for a node. Only defines the distribution over the domain
    # for a particular setting of the parent values. I assume that the distribution explicitly mentions all
    # variables in the domain - a future feature could infer that values should be zero if left out.
    def add_entry(self, parent_vals, distribution):
        assert not self.marginal_distribution, "Can't set conditional if already have marginal distribution " + self.marginal_distribution
        # The distribution being passed in had better be proper.
        assert almost_equal(sum(distribution.values()), 1), "Improper distribution passed in " + str(distribution)
        # Convert the parent vals into a set because order doesn't matter; use frozenset to make hashable.
        conditioning_event = frozenset(parent_vals)
        assert conditioning_event not in self.conditional_distribution.keys(), "Already have event for " + conditioning_event
        if self.domain:
            assert distribution.keys() == self.domain, "Mismatched domains."
        else:  # Haven't set domain yet, so do so now.
            self.domain = distribution.keys()
        self.parents = [parent_node for parent_node, val in parent_vals]
        # Check the parent domains as well.
        for parent_node, val in parent_vals:
            assert val in parent_node.domain, "Couldn't find " + str(val) + " in parent node " + str(parent_node) + " domain."
        self.conditional_distribution[conditioning_event] = distribution
        
    def get_prob_value(self, val, parent_vals=None):
        if parent_vals is None:
            assert not self.parents, "Cannot ask for probability without giving parent values"
        if not self.parents:
            return self.marginal_distribution.get(val)
        # Turn the parent assignments into a set.
        conditioning_event = frozenset(parent_vals)
        return self.conditional_distribution.get(conditioning_event).get(val)
        

"""
Defines a BayesNet object. Just a holder for a bunch of nodes.
"""

class BayesNet:
    def __init__(self, nodes):
        self.nodes = nodes
        # Helper lookup map that goes from a variable name to the node itself (with the distribution info)
        self.name_to_nodes = {}
        for node in nodes:
            self.name_to_nodes[node.name] = node

        # This would also be a great place to do sanity checks on the distributions themselves fully specifying
        # the distributions and summing to 1.
        # Could also come up with some sort of topographical ordering (and check that there are no loops)
        
    def get_node(self, name):
        assert name in self.name_to_nodes.keys(), "Could not find node named " + name
        return self.name_to_nodes.get(name)
    
    # Given assignments for all the variables, return the probability of that assignment
    def calc_joint(self, var_assignments):
        assert len(var_assignments) == len(self.nodes)
        running_prob = 1.0
        for var, value in var_assignments:
            # Loop up the corresponding probability, which means I also need to check parents and their values.
            if not var.parents:
                marginal_prob = var.get_prob_value(value)
                running_prob = running_prob * marginal_prob
                continue
            # Must look up parents and their values.
            parent_assignments = []
            for possible_var, possible_val in var_assignments:
                if possible_var in var.parents:
                    parent_assignments.append((possible_var, possible_val))
            fetched_prob = var.get_prob_value(value, parent_assignments)
            running_prob = running_prob * fetched_prob
        return running_prob

    def draw_net(self):
        nxg = nx.DiGraph()
        edges = []
        for node in self.nodes:
            if node.parents:
                edges.extend([(parent, node) for parent in node.parents])
        nxg.add_edges_from(edges)
        print("Have", nxg.number_of_nodes(), "nodes and", nxg.number_of_edges(), "edges")
        
        # Define the node labels
        node_labels = {}
        for node in self.nodes:
            node_labels[node] = node.name
        
        # Define some layout. Clearly, I could do something way better that would force the parents up top, but
        # I haven't spent time on that yet.
        pos = nx.spring_layout(nxg)
        nx.draw_networkx_nodes(nxg, pos, node_color='r')
        nx.draw_networkx_edges(nxg, pos, edges)
        nx.draw_networkx_labels(nxg, pos, node_labels, font_size=16)
        plt.axis('off')
        plt.show()