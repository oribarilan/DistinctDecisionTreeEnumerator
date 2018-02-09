from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd

class TreeAndFeatures(object):
    def __init__(self, tree: DecisionTreeClassifier, feature_list: list):
        self.dtree = tree
        self.feature_list = feature_list

class DTdistinct(object):
    def __init__(self, dataset: pd.DataFrame, random_state=None):
        self.dataset = dataset
        self.random_state = random_state

    def get_feature_samplecount_tuple(self, estimator: DecisionTreeClassifier):
        #   - left_child, id of the left child of the node
        #   - right_child, id of the right child of the node
        #   - feature, feature used for splitting the node
        #   - n_node_samples, samples reached each node
        n_nodes = estimator.tree_.node_count
        children_left = estimator.tree_.children_left
        children_right = estimator.tree_.children_right
        feature = estimator.tree_.feature
        node_sample_count = estimator.tree_.n_node_samples
        # The tree structure can be traversed to compute various properties such
        # as the depth of each node and whether or not it is a leaf.
        node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
        is_leaves = np.zeros(shape=n_nodes, dtype=bool)
        stack = [(0, -1)]  # seed is the root node id and its parent depth
        while len(stack) > 0:
            node_id, parent_depth = stack.pop()
            node_depth[node_id] = parent_depth + 1

            # If we have a test node
            if (children_left[node_id] != children_right[node_id]):
                stack.append((children_left[node_id], parent_depth + 1))
                stack.append((children_right[node_id], parent_depth + 1))
            else:
                is_leaves[node_id] = True
        feature_samplecount_tuple = []
        for i in range(n_nodes):
            if not is_leaves[i]:
                current_node_feature = feature[i]
                current_node_sample_count = node_sample_count[i]
                feature_samplecount_tuple.append((current_node_feature, current_node_sample_count))
        return feature_samplecount_tuple

    def get_feature_indexes(self, T: DecisionTreeClassifier):
        '''
        
        '''
        feature_samplecount_tuples = self.get_feature_samplecount_tuple(T)
        feature_idx_set = set([feature_idx for (feature_idx, samplecount) in feature_samplecount_tuples])
        return feature_idx_set

    def frontier(self, TF: TreeAndFeatures, SmU: set):
        #sum unique sample count 
        unique_samplecount = dict.fromkeys(self.get_feature_indexes(TF.dtree), 0)
        feature_samplecount_tuples = self.get_feature_samplecount_tuple(TF.dtree)
        for feature_idx, samplecount in feature_samplecount_tuples:
            unique_samplecount[feature_idx] += samplecount
        unique_feature_samplecount_tuples = unique_samplecount.items()
        #feature indexes to feature values
        unique_feature_samplecount_tuples = [(TF.feature_list[feature_idx], samplecount) for (feature_idx, samplecount) in unique_feature_samplecount_tuples]
        #filter out features that are not in SmU
        feature_samplecount_tuples = list(filter(lambda tuple: tuple[0] in SmU, unique_feature_samplecount_tuples))        
        #order by samplecount, ascending. feature tuples := (feature_idx, samplecount)
        feature_tuples = sorted(feature_samplecount_tuples,
                                                    key = lambda tuple: tuple[1], reverse=False)
        feature_value = [feature_value for (feature_value, samplecount) in feature_tuples]
        return feature_value

    def DT(self, feature_index_set: set):
        ''' 
        returns (TF, U)
        TF - TreeAndFeatures from R U S
        U - Set of unused features from R U S
        '''
        # [11, 12, 13]
        # TODO TF feature list should contain all give features
        # fix feature set
        feature_index_list_val = list(feature_index_set)
        X = self.dataset.iloc[:, feature_index_list_val]
        y = self.dataset.iloc[:, -1:]
        T = DecisionTreeClassifier(random_state=self.random_state)
        T = T.fit(X, y)
        # [0, 1]
        feature_index_list_idx = self.get_feature_indexes(T)
        feature_list = list(map(lambda idx: feature_index_list_val[idx], feature_index_list_idx))
        TF = TreeAndFeatures(tree=T, feature_list=feature_index_list_val)
        U = feature_index_set - set(feature_list)
        return TF, U

    def DTdistinct_enumerator_core(self, R: set, S: set, trees: list, subsets: list, indent: int):
        '''
        R, S - set of feature names
        '''
        if len(R | S) == 0:
            return
        TF, U = self.DT(R | S)
        indentation = ' ' * indent
        print(indentation + "=====================================")
        print(indentation + f"R = {R}")
        print(indentation + f"S = {S}")
        print(indentation + f"chosen features = {TF.feature_list}")
        print(indentation + f"unused features = {U}")
        print(indentation + f"all features = {R | S}")
        if len(R & U) == 0:
            print(indentation + "tree chosen")
            trees.append(TF)
            subsets.append(R | S)
        print(indentation + "=====================================")
        if len(S) == 0:
            return
        Rtag = R | ( S - U )
        Stag = S & U
        front = self.frontier(TF, S-U)
        print(indentation + f"front = {front}")
        for ai in front:
            print(indentation + f"ai = {ai}")
            Rtag = Rtag - set([ai])
            self.DTdistinct_enumerator_core(Rtag, Stag, trees, subsets, indent+1)
            Stag = Stag | set([ai])

    def DTdistinct_enumerator(self, R, S):
        trees = []
        subsets = []
        self.DTdistinct_enumerator_core(R, S, trees, subsets, 0)
        return trees, subsets

    @staticmethod
    def subset_core(R: set, S: set, trees: list):
        '''
        R, S - set of feature names
        '''
        if len(S) == 0:
            return
        trees.append(R | S)
        Rtag = set(R | S)
        Stag = set()
        for ai in S:
            Rtag = Rtag - set([ai])
            DTdistinct.subset_core(Rtag, Stag, trees)
            Stag = Stag | set([ai])

    @staticmethod
    def subset(R, S):
        trees = []
        DTdistinct.subset_core(R, S, trees)
        return trees
