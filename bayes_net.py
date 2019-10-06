# -*- coding: utf-8 -*-
class BayesNode:
    def __init__(self, var_name):
        self.name = var_name
        # Maps from (parent, val) tuples to map from value to probability.
        # Parents are BayesNodes themselves.
        # E.g. set((A, True), (B, 9)) -> {1: 0.2, 2: 0.7, 3: 0.1}
        self.conditional_distribution = {}
        self.marginal_distribution = {}
        
        self.domain = set()
        
    def set_marginal_distribution(self, distribution):
        assert not self.conditional_distribution, "Can't set distribution if already have conditional dist."
        self.marginal_distribution = distribution
        self.domain = distribution.keys()
    
    def add_entry(self, parent_vals, distribution):
        assert not self.marginal_distribution, "Can't set conditional if already have marginal distribution " + self.marginal_distribution
        # Convert the parent vals into a set because order doesn't matter; use frozenset to make hashable.
        conditioning_event = frozenset(parent_vals)
        assert conditioning_event not in self.conditional_distribution.keys(), "Already have event for " + conditioning_event
        if self.domain:
            assert distribution.keys() == self.domain, "Mismatched domains."
        else:  # Haven't set domain yet, so do so now.
            self.domain = distribution.keys()
        # Check the parent domains as well.
        for parent_node, val in parent_vals:
            assert val in parent_node.domain, "Couldn't find " + str(val) + " in parent node " + str(parent_node) + " domain."
        self.conditional_distribution[conditioning_event] = distribution

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
        
    def get_node(self, name):
        assert name in self.name_to_nodes.keys(), "Could not find node named " + name
        return self.name_to_nodes.get(name)