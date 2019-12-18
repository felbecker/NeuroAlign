import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import scipy
from sklearn.cluster import KMeans

import AnchorSet


MAX_PAIRWISE_SCORE = 11 #based on blosum62, probably needs adjustment for other matrices

# based on an anchor set, defines a set of patterns in these anchors
class PatternSet:

    def __init__(self, anchor_set):

        self.anchor_set = anchor_set

        self.num_pattern = 0

        #since region_node_list is sorted pattern-wise, this list fully describes
        #the distribution of the region nodes to the patterns
        self.num_regions_per_pattern = []

        #the number of instances of each pattern
        #(the more regions per sequence a pattern has, the more
        #instances of it are needed for accurate prediction)
        self.num_instances_per_pattern = []

        #a complete list of all regions occuring in the anchor set,
        #but sorted pattern-wise
        #that means region_node_list[0:num_members_per_pattern[0]] are the nodes of the first pattern,
        #region_node_list[num_members_per_pattern[0],num_members_per_pattern[1]] the second and so on
        self.region_node_list = []

        #adjacency list representation of the pattern graph
        self.pattern_graph = []

        #flag indicates whether the values for a training target have been computed
        self.target_initialized = False

        #for each region node, this holds the relative position of the region in the reference alignment
        self.region_rp_targets = None

        #for each pattern node (instance), this holds the relative position of the pattern column
        #in the reference alignment, if #(exact columns of the pattern in the alignment) <= #(instances of the pattern)
        #or a sufficient approximation elsewise
        self.pattern_rp_targets = None

        #datastructure that describes for each pattern, how the corresponding
        #region nodes are partitionized on the instances of the pattern
        #this partition is exact if #(exact columns of the pattern in the alignment) <= #(instances of the pattern)
        #and approximative elsewise (soft k-means)
        self.region_to_pattern_partition = None

        #datastructure that maps anchors to patterns
        self.anchor_to_pattern = []

        self.instance_membership_targets = []

        #the score of a pattern is the mean of the intrinsic anchor scores
        self.pattern_scores = None

        #the score of a region is the mean score if incident anchors
        self.region_scores = None


        #flag indicates whether a predicting has be computed and attached
        self.prediction_attached = False
        self.pred_region_rp = None
        self.pred_pattern_rp = None


    #draws the sequence graph and the pattern graph with respect to their initial positions (raw)
    #labels nodes with the relative positions in the alignment
    #labels transition edges with the degree of membership
    def draw_pattern_graph(self, ax):

        #produce random colors for each pattern
        cmap = plt.get_cmap('hsv')
        pattern_colors = cmap(np.random.uniform(0,1,self.num_pattern))

        seq_g = nx.DiGraph()
        node_colors = []
        for i,n in enumerate(self.num_regions_per_pattern):
            node_colors += [pattern_colors[i]]*n
        for n,col,trp,prp in zip(self.region_node_list, node_colors, self.region_rp_targets, self.pred_region_rp):
            seq_g.add_node(n, color=col, target_rp=trp, pred_rp=prp)
        node_colors = nx.get_node_attributes(seq_g,'color')
        node_colors = [node_colors[n] for n in seq_g.nodes]

        d = self.anchor_set.anchor_data
        edges = [((d[i,AnchorSet.SEQ_ST], d[i,AnchorSet.IND_ST], d[i,AnchorSet.LEN]),
                    (d[i,AnchorSet.SEQ_END], d[i,AnchorSet.IND_END], d[i,AnchorSet.LEN]))
                        for i in range(self.anchor_set.anchor_data.shape[0])]
        edge_colors = [pattern_colors[self.anchor_to_pattern[i]] for i in range(self.anchor_set.anchor_data.shape[0])]
        for e, col in zip(edges, edge_colors):
            seq_g.add_edge(e[0], e[1], color = col)
            seq_g.add_edge(e[1], e[0], color = col)
        edge_colors = nx.get_edge_attributes(seq_g,'color')
        edge_colors = [edge_colors[e] for e in seq_g.edges]


        node_labels = {} #defaults when no target is initialized
        if self.target_initialized:
            node_labels = {node: ("\n\n\nT={:.2g}".format(data["target_rp"])+"\nP={:.2g}".format(data["pred_rp"][0]))
                 for node, data in seq_g.nodes(data=True)}

        for num, (seq,i,l), score in zip(range(len(self.region_node_list)), self.region_node_list, self.region_scores):
            sum_inst = 0
            for pattern, num_inst in enumerate(self.num_regions_per_pattern):
                sum_inst += num_inst
                if num < sum_inst:
                    break
            rect = patches.Rectangle((i,-seq),l,0.1*score,facecolor=pattern_colors[pattern])
            ax.add_patch(rect)

        #draw the sequence graph
        nodepos = {(seq,i,l):(i,-seq) for seq,i,l in seq_g.nodes}
        nx.draw(seq_g, pos=nodepos, edge_color=edge_colors, node_color = node_colors, ax=ax, node_size=10, labels=node_labels, font_size=10)

        nodes_sequence_wise = []
        for s in range(self.anchor_set.num_seq):
            nodes_on_seq = [self.region_node_list[i] for i in range(len(self.region_node_list)) if self.region_node_list[i][0] == s]
            nodes_on_seq.sort(key=lambda x: x[1])
            nodes_sequence_wise.append(nodes_on_seq)

        forward_edges = []
        for s in nodes_sequence_wise:
            for i in range(1,len(s)):
                forward_edges.append((s[i-1], s[i]))
        nx.draw_networkx_edges(seq_g, pos=nodepos, edgelist=forward_edges)

        #draw the pattern graph
        pattern_g = nx.Graph()
        i = 0
        for pattern_index, num in enumerate(self.num_instances_per_pattern):
            for _ in range(num):
                pattern_g.add_node(i, pattern=pattern_index, target_rp=self.pattern_rp_targets[i], score=self.pattern_scores[pattern_index])
                i += 1

        pattern_node_colors = [pattern_colors[data["pattern"]] for node, data in pattern_g.nodes(data=True)]
        maxlen = max([r[1] for r in self.region_node_list])
        minlen = min([r[1] for r in self.region_node_list])
        pattern_node_pos = {i:(minlen*0.5+ i*(maxlen-minlen)*1.5/len(pattern_g.nodes), -self.anchor_set.num_seq-1) for i in pattern_g.nodes}

        pattern_node_labels = {} #defaults when no target is initialized
        if self.target_initialized:
            pattern_node_labels = {node: "\n\n{:.2g}".format(data["target_rp"])
                 for node, data in pattern_g.nodes(data=True)}
        pattern_node_sizes = [data["score"]*80 for node, data in pattern_g.nodes(data=True)]
        nx.draw(pattern_g, pos=pattern_node_pos, node_color = pattern_node_colors, ax=ax, node_size=pattern_node_sizes, labels=pattern_node_labels, font_size=9)




#given a pattern, returns the number of instance nodes for this pattern
#this computation is a tradeoff between efficiency and exactness
#pattern here is a set of regions {(seq, pos)-pairs}
#todo: replace this ad hoc computation with something more elaborated ?
def get_instance_count(pattern, maxlen):
    C1 = 0.5
    C2 = 2
    num_per_sequence = {}
    for node in pattern:
        if not int(node[0]) in num_per_sequence:
            num_per_sequence[int(node[0])] = 0
        num_per_sequence[int(node[0])] += 1
    distances = scipy.spatial.distance.pdist(np.reshape(np.array([int(region[1]/maxlen) for region in pattern]), [-1,1]))
    #ninst = C1*max(num_per_sequence.values()) + C2*np.mean(np.ndarray.flatten(distances))
    ninst = C1*max(num_per_sequence) + C2*np.mean(np.ndarray.flatten(distances))
    return max(1, min(len(pattern) ,int(ninst)))


#finds all patterns in the given anchor set
#outputs a nodelist such that anchors belonging to the same pattern are grouped
#concecutively and groups occur in the same order as patterns do. Outputs the number of nodes for each pattern as well as the number
#of instances of the pattern ( = max number of nodes of the respective pattern per sequence )
#outputs the edges of the pattern graph in adjacency lists representation
def find_patterns(anchor_set):

    pattern_set = PatternSet(anchor_set)

    #find connected components
    g = nx.Graph()
    d = anchor_set.anchor_data
    for i in range(anchor_set.anchor_data.shape[0]):
        assert d[i, AnchorSet.LEN] <= anchor_set.len_seqs[int(d[i, AnchorSet.SEQ_ST])], "Bug: anchor cannot be longer than whole seq"
        assert d[i, AnchorSet.LEN] <= anchor_set.len_seqs[int(d[i, AnchorSet.SEQ_END])], "Bug: anchor be longer than whole seq"
        g.add_edge((d[i, AnchorSet.SEQ_ST], d[i, AnchorSet.IND_ST], d[i, AnchorSet.LEN]),
                    (d[i, AnchorSet.SEQ_END], d[i, AnchorSet.IND_END], d[i, AnchorSet.LEN]), score=d[i, AnchorSet.SCORE])

    ccs = [c for c in nx.connected_components(g)]

    pattern_set.num_pattern = len(ccs)

    pattern_set.num_regions_per_pattern = [len(c) for c in ccs]

    #computes the maximum number of nodes on a single sequences per connected component
    pattern_set.num_instances_per_pattern = []
    for c in ccs:
        pattern_set.num_instances_per_pattern.append(get_instance_count(c, max(anchor_set.len_seqs)))

    pattern_set.region_node_list = [n for c in ccs for n in c]

    pattern_set.anchor_to_pattern = np.zeros(d.shape[0], dtype=int)
    #todo, make this efficient if neccessary...
    for i in range(d.shape[0]):
        for j,c in enumerate(ccs):
            if (d[i, AnchorSet.SEQ_ST], d[i, AnchorSet.IND_ST], d[i, AnchorSet.LEN]) in c:
                pattern_set.anchor_to_pattern[i] = j

    #compute the scores
    pattern_set.region_scores = np.zeros(len(pattern_set.region_node_list))
    for i,region in enumerate(pattern_set.region_node_list):
        pattern_set.region_scores[i] = np.mean([data["score"] for u,v, data in g.edges(region, data=True)])


    pattern_set.pattern_scores = np.array([np.mean([data["score"] for u,v, data in g.edges(c, data=True)]) for c in ccs])

    return pattern_set

#computes training targets given an pattern set
#the corresponding anchor set must have a reference solution already loaded
#outputs:
#a vector of relative positions of each sequence node in the reference solution
#edge labels (0/1) for the transition edges between sequence- and pattern-nodes (membership)
#relative positions of the patterns (columns) in the reference alignment
def compute_targets(pattern_set):
    assert pattern_set.anchor_set.solution_loaded, "no reference solution loaded for the anchor set"

    #a note on this:
    #we assume here, that parameters for seq preprocessing were chosen in a way,
    #such that predicting gapless regions that are not gapless in the reference alignment occur almost never
    #with this simplification, we can just compute the relative position of the first Protein in the reference alignment
    pattern_set.region_rp_targets = np.zeros(len(pattern_set.region_node_list))
    for i, region in enumerate(pattern_set.region_node_list):
        pattern_set.region_rp_targets[i] = (region[1]+pattern_set.anchor_set.gap_counts[int(region[0])][int(region[1])])/pattern_set.anchor_set.len_ref_seq

    pattern_set.instance_membership_targets = np.zeros(len(pattern_set.region_node_list))

    pattern_set.pattern_rp_targets = np.zeros(sum(pattern_set.num_instances_per_pattern))
    sum_inst = 0
    sum_region = 0
    for num_inst, num_region in zip(pattern_set.num_instances_per_pattern, pattern_set.num_regions_per_pattern):
        uniques = list(set([rt for rt in pattern_set.region_rp_targets[sum_region:(sum_region+num_region)]]))
        if len(uniques) <= num_inst:
            sorted_centers = sorted(uniques)
            for j in range(sum_region, sum_region+num_region):
                pattern_set.instance_membership_targets[j] = sorted_centers.index(pattern_set.region_rp_targets[j])
        else:
            kmeans = KMeans(n_clusters=num_inst, max_iter=10).fit(np.reshape(pattern_set.region_rp_targets[sum_region:(sum_region+num_region)], (-1, 1)))
            sorted_indices, sorted_centers = zip(*sorted(enumerate(np.reshape(kmeans.cluster_centers_, (-1)))))
            sorted_indices, sorted_centers = list(sorted_indices), list(sorted_centers)
            for j in range(sum_region, sum_region+num_region):
                pattern_set.instance_membership_targets[j] = sorted_indices.index(kmeans.labels_[j-sum_region])
        for j,c in enumerate(sorted_centers):
            pattern_set.pattern_rp_targets[sum_inst+j] = c
        sum_inst += num_inst
        sum_region += num_region

    pattern_set.target_initialized = True


def attach_prediction(pattern_set, region_rp, pattern_rp):
    pattern_set.prediction_attached = True
    pattern_set.pred_region_rp = region_rp
    pattern_set.pred_pattern_rp = pattern_rp


#computes anchor probabilities given a pattern set and a vector of predicted relative positions of the regions
def get_anchor_probs(pattern_set, region_rp):

    region_rp = np.reshape(region_rp, (-1))

    region_index_mapper = {region:i for i,region in enumerate(pattern_set.region_node_list)}

    d = pattern_set.anchor_set.anchor_data
    num_anchors = d.shape[0]
    anchor_probs = np.zeros(num_anchors)
    for i in range(num_anchors):
        r1 = (d[i, AnchorSet.SEQ_ST], d[i, AnchorSet.IND_ST], d[i, AnchorSet.LEN])
        r2 = (d[i, AnchorSet.SEQ_END], d[i, AnchorSet.IND_END], d[i, AnchorSet.LEN])
        diff =  abs(region_rp[region_index_mapper[r1]] - region_rp[region_index_mapper[r2]])
        anchor_probs[i] = 1 - diff
    return anchor_probs
