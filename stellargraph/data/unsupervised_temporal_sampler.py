#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2019-2020 Data61, CSIRO
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

__all__ = ["UnsupervisedTemporalSampler"]


import numpy as np
import pandas as pd
import random

from stellargraph.core.utils import is_real_iterable
from stellargraph.core.graph import StellarGraph
from stellargraph.data.explorer import UniformRandomWalk
from stellargraph.data.explorer import TemporalRandomWalk
from stellargraph.random import random_state


def _warn_if_ignored(value, default, name):
    if value != default:
        raise ValueError(
            f"walker, {name}: cannot specify both 'walker' and '{name}'. Please use one or the other."
        )


class UnsupervisedTemporalSampler:
    """
        The UnsupervisedTemporalSampler is responsible for sampling walks in the given graph
        and returning positive and negative samples w.r.t. those walks, on demand.

        The positive samples are all the (target, context) pairs from the walks and the negative
        samples are contexts generated for each target based on a sampling distribution.

        By default, a UniformRandomWalk is used, but a custom `walker` can be specified instead. An
        error will be raised if other parameters are specified along with a custom `walker`.

        .. seealso::

           Examples using this sampler:

           - Attri2Vec: `node classification <https://stellargraph.readthedocs.io/en/stable/demos/node-classification/attri2vec-node-classification.html>`__ `link prediction <https://stellargraph.readthedocs.io/en/stable/demos/link-prediction/attri2vec-link-prediction.html>`__, `unsupervised representation learning <https://stellargraph.readthedocs.io/en/stable/demos/embeddings/attri2vec-embeddings.html>`__
           - GraphSAGE: `unsupervised representation learning <https://stellargraph.readthedocs.io/en/stable/demos/embeddings/graphsage-unsupervised-sampler-embeddings.html>`__
           - Node2Vec: `node classification <https://stellargraph.readthedocs.io/en/stable/demos/node-classification/keras-node2vec-node-classification.html>`__, `unsupervised representation learning <https://stellargraph.readthedocs.io/en/stable/demos/embeddings/keras-node2vec-embeddings.html>`__
           - `comparison of link prediction algorithms <https://stellargraph.readthedocs.io/en/stable/demos/link-prediction/homogeneous-comparison-link-prediction.html>`__

           Built-in classes for ``walker``: :class:`.UniformRandomWalk`, :class:`.BiasedRandomWalk`, :class:`.UniformRandomMetaPathWalk`.

        Args:
            G (StellarGraph): A stellargraph with features.
            nodes (iterable, optional) The root nodes from which individual walks start.
                If not provided, all nodes in the graph are used.
            length (int): Length of the walks for the default UniformRandomWalk walker. Length must
                be at least 2.
            number_of_walks (int): Number of walks from each root node for the default
                UniformRandomWalk walker.
            seed (int, optional): Random seed for the default UniformRandomWalk walker.
            walker (RandomWalk, optional): A RandomWalk object to use instead of the default
                UniformRandomWalk walker.
    """

    def __init__(
        self, G, negative_nodes_dict, nodes=None, length=2, number_of_walks=1, context_window_size=2, neg_sampling_length=1, 
        seed=None, walker=None, neg_sampling_method=None, add_complete_matches=False
    ):
        if not isinstance(G, StellarGraph):
            raise ValueError(
                "({}) Graph must be a StellarGraph or StellarDigraph object.".format(
                    type(self).__name__
                )
            )
        else:
            self.graph = G

        # Instantiate the walker class used to generate random walks in the graph
        if walker is not None:
            _warn_if_ignored(length, 2, "length")
            _warn_if_ignored(number_of_walks, 1, "number_of_walks")
            _warn_if_ignored(seed, None, "seed")
            self.walker = walker
        else:
            self.walker = UniformRandomWalk(
                G, n=number_of_walks, length=length, seed=seed
            )

        # Define the root nodes for the walks
        # if no root nodes are provided for sampling defaulting to using all nodes as root nodes.
        if nodes is None:
            self.nodes = list(G.nodes())
        elif is_real_iterable(nodes):  # check whether the nodes provided are valid.
            self.nodes = list(nodes)
        else:
            raise ValueError("nodes parameter should be an iterable of node IDs.")

        # Require walks of at lease length two because to create a sample pair we need at least two nodes.
        if length < 2:
            raise ValueError(
                "({}) For generating (target,context) samples, walk length has to be at least 2".format(
                    type(self).__name__
                )
            )
        else:
            self.length = length

        if number_of_walks < 1:
            raise ValueError(
                "({}) At least 1 walk from each head node has to be done".format(
                    type(self).__name__
                )
            )
        else:
            self.number_of_walks = number_of_walks


        # Add context window parameter and other
        self.context_window_size = context_window_size
        self.negative_nodes_dict = negative_nodes_dict
        self.neg_sampling_length = neg_sampling_length
        self.neg_sampling_method = neg_sampling_method
        self.add_complete_matches = add_complete_matches
    
        # Setup an interal random state with the given seed
        _, self.np_random = random_state(seed)

    def run(self, batch_size):
        """
        This method returns a batch_size number of positive and negative samples from the graph.
        A random walk is generated from each root node, which are transformed into positive context
        pairs, and the same number of negative pairs are generated from a global node sampling
        distribution. The resulting list of context pairs are shuffled and converted to batches of
        size ``batch_size``.

        Currently the global node sampling distribution for the negative pairs is the degree
        distribution to the 3/4 power. This is the same used in node2vec
        (https://snap.stanford.edu/node2vec/).

        Args:
             batch_size (int): The number of samples to generate for each batch.
                This must be an even number.

        Returns:
            List of batches, where each batch is a tuple of (list context pairs, list of labels)
        """
        self._check_parameter_values(batch_size)

        walks = self.walker.run(nodes=self.nodes)
        
        # first item in each walk is the target/head node
        targets = [walk[0] for walk in walks]

        positive_pairs = np.array(
            [
                (target, positive_context)
                for target, walk in zip(targets, walks)
                for positive_context in walk[1:]
            ]
        )       

        if self.add_complete_matches: # finds connected nodes from head node
            bankstatements = [self.graph.nodes()[k] for k,v in self.negative_nodes_dict.items() if len(v) if v[1] == 'Bank Statement']

            added_positive_pairs = np.array(
                [
                    (sampled_statement, matched_lineitem)
                    for sampled_statement in list(set(positive_pairs.flatten()).intersection(bankstatements))
                    for matched_lineitem in self.graph.in_node_arrays(node=sampled_statement)
                ]
            )

            positive_pairs = np.concatenate((positive_pairs, added_positive_pairs), axis=0)


        positive_pairs = self.graph.node_ids_to_ilocs(positive_pairs.flatten()).reshape(
            positive_pairs.shape
        )


        ################## normal sampling method ##################
        all_nodes = list(self.graph.nodes(use_ilocs=True))
        # Use the sampling distribution as per node2vec
        degrees = self.graph.node_degrees(use_ilocs=True)
        sampling_distribution = np.array([degrees[n] ** 0.75 for n in all_nodes])
        sampling_distribution_norm = sampling_distribution / np.sum(
            sampling_distribution
        )

        negative_samples = self.np_random.choice(
            all_nodes, size=len(positive_pairs)*self.neg_sampling_length, p=sampling_distribution_norm
        )

        negative_pairs = np.column_stack((positive_pairs[:, 0], negative_samples))

        ############# temporally-aware sampling method #############
        temporal_negative_pairs = np.array(
            [
                (x, negative_context)
                for x,_ in positive_pairs
                for negative_context in self._return_node_list(self.negative_nodes_dict[x], hard_negatives=False)
            ]
        )

        ############## hard negative sampling method ##############
        hard_negative_pairs = np.array(
            [
                (x, negative_context)
                for x,_ in positive_pairs
                for negative_context in self._return_node_list(self.negative_nodes_dict[x], hard_negatives=True)
            ]
        )

        if self.neg_sampling_method == 'norm_and_temporal': negative_pairs = np.concatenate((negative_pairs, temporal_negative_pairs), axis=0)
        elif self.neg_sampling_method == 'norm_and_hard': negative_pairs = np.concatenate((negative_pairs, hard_negative_pairs), axis=0)
        elif self.neg_sampling_method == 'temporal_only': negative_pairs = temporal_negative_pairs
        elif self.neg_sampling_method == 'hard_only': negative_pairs = hard_negative_pairs
        elif self.neg_sampling_method == 'temporal_and_hard': negative_pairs = np.concatenate((temporal_negative_pairs, hard_negative_pairs), axis=0)
        else: pass

        # remove empties and duplicates
        if self.neg_sampling_method is not None:
            negative_pairs = negative_pairs[negative_pairs[:,-1] != None]
            negative_pairs = pd.DataFrame(negative_pairs).drop_duplicates().values

        pairs = np.concatenate((positive_pairs, negative_pairs), axis=0)
        labels = np.concatenate((np.repeat([1], len(positive_pairs)),
                                np.repeat([0], len(negative_pairs))), axis=0)

        # shuffle indices - note this doesn't ensure an equal number of positive/negative examples in
        # each batch, just an equal number overall
        indices = self.np_random.permutation(len(pairs))

        batch_indices = [
            indices[i : i + batch_size] for i in range(0, len(indices), batch_size)
        ]

        return [(pairs[i].astype(int), labels[i]) for i in batch_indices]

    def _check_parameter_values(self, batch_size):
        """
        Checks that the parameter values are valid or raises ValueError exceptions with a message indicating the
        parameter (the first one encountered in the checks) with invalid value.

        Args:
            batch_size: <int> number of samples to generate in each call of generator

        """

        if (
            batch_size is None
        ):  # must provide a batch size since this is an indicator of how many samples to return
            raise ValueError(
                "({}) The batch_size must be provided to generate samples for each batch in the epoch".format(
                    type(self).__name__
                )
            )

        if type(batch_size) != int:  # must be an integer
            raise TypeError(
                "({}) The batch_size must be positive integer.".format(
                    type(self).__name__
                )
            )

        if batch_size < 1:  # must be greater than 0
            raise ValueError(
                "({}) The batch_size must be positive integer.".format(
                    type(self).__name__
                )
            )

        if (
            batch_size % 2 != 0
        ):  # should be even since we generate 1 negative sample for each positive one.
            raise ValueError(
                "({}) The batch_size must be an even integer since equal number of positive and negative samples are generated in each batch.".format(
                    type(self).__name__
                )
            )


    def _return_node_list(self, negative_samples, hard_negatives=False): 
        maxlen = self.neg_sampling_length

        if len(negative_samples) > 0: negative_samples = negative_samples[0]

        if hard_negatives:
            if len(negative_samples) >= maxlen: return negative_samples[:maxlen]
            elif len(negative_samples) == 0: return [None]
            else: return negative_samples[:len(negative_samples)]
            
        else:
            if len(negative_samples) >= maxlen: return self.np_random.choice(negative_samples,size=maxlen)
            elif len(negative_samples) == 0: return [None]
            else: return self.np_random.choice(negative_samples,size=len(negative_samples))
