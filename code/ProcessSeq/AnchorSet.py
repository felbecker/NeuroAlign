import argparse
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import itertools
from concurrent.futures import ProcessPoolExecutor
import os
from scipy import signal
import bisect
import copy


#enum for data access
SEQ_ST = 0
IND_ST = 1
SEQ_END = 2
IND_END = 3
SCORE = 4
LEN = 5

##############################################################################################################
##############################################################################################################
##############################################################################################################
#wraps a scoring matrix to be shared among instances
class ScoringMatrix:
    #reads a blosum file from harddrive
    def __init__(self, filename):
        protein_2_index = {}
        scores = []

        with open(filename) as file:
            desc_line = False
            for line in file:
                if line[0] == '#':
                    continue
                if not desc_line:
                    alphabet = line.split()
                    desc_line = True
                    for i in range(len(alphabet)):
                        protein_2_index[alphabet[i].lower()] = i
                        protein_2_index[alphabet[i].upper()] = i
                else:
                    scores.append( [int(x) for x in line.split()[1:]] )

        self.scores = np.matrix(scores)
        self.protein_2_index = protein_2_index


##############################################################################################################
##############################################################################################################
##############################################################################################################
#wraps a MSA instance
class MSAInstance:
    #reads a fasta file, returns a list of strings
    def __init__(self, filename, skip_gaps=False):
        _, file_extension = os.path.splitext(filename)
        with open(filename) as f:
            content = f.readlines()
        self.seq = []
        if file_extension == ".tfa" or file_extension == ".fasta":
            seq_open = False
            for line in content:
                line = line.strip()
                if len(line)>0:
                    if line[0]=='>':
                        seq_open = True
                    elif seq_open:
                        self.seq.append(line)
                        seq_open = False
                    else:
                        self.seq[-1] += line
        elif file_extension == ".xml":
            for line in content:
                if "<seq-data>" in line:
                    self.seq.append(line[10:])
        if skip_gaps:
            self.seq = [s.replace('-','') for s in self.seq]


#returns a suggestion for the minrow parameter based on the average row length
def sample_min_row(rows):
    av_len = int(np.floor(sum([len(r) for r in rows])/len(rows)))
    mr = av_len
    return max(2, min(mr, 200))

#returns a suggestion for the threshold parameter based on the total length of the sequences
def sample_threshold(instance, radius):
    sum_len = sum([len(s) for s in instance.seq])
    t = int(np.floor(np.sqrt(sum_len)*radius/16))-7
    return max(0, t)

##############################################################################################################
##############################################################################################################
##############################################################################################################

#helper for multithreading
#computes all anchors between sequences i and j
def compute_sij(ij, index_seq, scores, radius, t):
    i, j = ij[0], ij[1]
    anchors = []
    anchor_scores = []
    num_kmer_i = len(index_seq[i])-2*radius
    num_kmer_j = len(index_seq[j])-2*radius
    anchor_data = []
    for d in range(-num_kmer_i+radius, num_kmer_j-radius+1):
        Sij_d = np.array([scores[index_seq[i][x], index_seq[j][x+d]] for x in range(max(0, -d), min(len(index_seq[i]), len(index_seq[j])-d))],dtype=float)
        Cij_d = signal.correlate(Sij_d, np.ones([2*radius+1]), mode='valid')
        gargs_d = np.argwhere(Cij_d >= t)
        anchor_data_d = np.zeros((gargs_d.shape[0], 6)) #also reserves storage for other attributes introduced later
        for num, g in enumerate(gargs_d):
            anchor_data_d[num, SEQ_ST] = i
            anchor_data_d[num, IND_ST] = g[0]+radius+max(0,-d)
            anchor_data_d[num, SEQ_END] = j
            anchor_data_d[num, IND_END] = g[0]+max(d,0)+radius
            anchor_data_d[num, SCORE] = Cij_d[g[0]]/(2*radius+1)
        anchor_data.append(anchor_data_d)
    return np.concatenate(anchor_data, axis=0)

#helper for multithreading
def unpack_compute_sij(args):
    return compute_sij(*args)

#wraps a set of anchors for a given MSAInstance
#
# an anchor is defined through a pair of positions in the sequences and a segment length
#
# Assumptions that a required to be true for the anchor set:
# 1) an anchor is a pair of pairs: ((s1,i),(s2,j)), it always holds that s1 < s2 (anchors always point top->down)
# 2) the list of anchors is sorted by the first index pair (s1,i) from top-left to down-right
#    (if sequences are printed line by line starting with the first in the first)
#    If two anchors start the the same position, they are sorted by the second index pair (s2,j)
# 3) segements defined through anchorpos+len do not overlap
#
class AnchorSet:

    def __init__(self):
        self.anchor_data = None #data matrix with columns: 0 - seq_start, 1 - index_start, 2 - seq_end, 3 - index_end, 4 - score, 5 - length
        self.solution = None #array where 1 indicates that an anchor is also part of a reference solution
        self.num_seq = 0 #number of input sequences
        self.len_seqs = [] #lengths of each input sequence
        self.loaded = False
        self.solution_loaded = False
        self.minrow = 1
        self.gap_counts = None #if a solution is loaded, this is a datastructure, that holds for each position in the sequences the number of gaps occuring before this sequence
        self.len_ref_seq = 0 #if a solution is loaded, this is the length of the sequences with gaps


    #stores the set on harddrive
    #2 files are generated:
    #a tab separated data file containing anchor information (ending .anchors)
    #a lightweight meta file with sequence information (ending .meta)
    #both files are required for loading the set
    #if a solution is loaded, a third file is generated which contains anchor presence values in the solution
    #if the anchor set contains anchors with length > 1, these values can be rationale
    def to_file(self, filename):
        if not self.loaded:
            return
        #create dict if not existing
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename+".meta", "w") as f:
            f.write("num_seq: " + str(self.num_seq) + " minrow: " + str(self.minrow) + " len_seqs: " + ' '.join([str(l) for l in self.len_seqs]) + "\n")
        np.savetxt(filename+".anchors", self.anchor_data, delimiter='\t', fmt='%1.3f')
        if self.solution_loaded:
            np.savetxt(filename+".sol", self.solution, delimiter='\t', fmt='%1.3f')


    #draws the alignment graph derived from the anchor_set to the given mathplotlib-device
    #if a solution is loaded, anchors that are also in the solution are drawn bold
    #a color map is applied to the anchors that depends on the edge score (higher score anchors are darker)
    def draw_alignment_graph(self, ax):
        assert self.loaded, "nothing to draw"
        num_anchors = self.anchor_data.shape[0]
        g = nx.Graph()
        edges = [((self.anchor_data[i,SEQ_ST], self.anchor_data[i,IND_ST]),
                    (self.anchor_data[i,SEQ_END], self.anchor_data[i,IND_END])) for i in range(num_anchors)]
        if self.solution_loaded:
            solution = self.solution
        else:
            solution = np.zeros((num_anchors))
        for i,e in enumerate(edges):
            g.add_edge(e[0], e[1], score = self.anchor_data[i,SCORE], solution = solution[i])
        max_score = np.max(self.anchor_data[:,SCORE])
        cmap = plt.get_cmap('hot')
        sc_attr = nx.get_edge_attributes(g,'score')
        ecolors = cmap(np.array([1-sc_attr[e]/max_score for e in g.edges]))
        labels = {n:str(n[1]) for n in g.nodes}
        sol_attr = nx.get_edge_attributes(g,'solution')
        ewidth =  [1+4*sol_attr[e] for e in g.edges]
        nx.draw(g, pos={(s,i):(i,-s) for s,i in g.nodes}, edge_color=ecolors, width=ewidth, ax=ax, node_size=80)
        nx.draw_networkx_labels(g, pos={(s,i):(i,-s+0.1) for s,i in g.nodes}, ax=ax, font_size = 8, labels=labels)


#returns an AnchorSet
#For two sequence positions (s1,i) and (s2,j), an anchor is created if and only if the region centered at either
#position extended by radius has a score >= threshold
#the score of 2 regions of equal length is defined as the sum of the individual protein pairs
#multithreading is possible and recommended
#among preprocessing steps, anchor computation is usually the bottleneck
def anchor_set_kmer_threshold(msa_instance, scoring_matrix, radius, threshold, threads_to_use=4):

    anchor_set = AnchorSet()

    anchor_set.num_seq = len(msa_instance.seq)
    anchor_set.len_seqs = [len(s) for s in msa_instance.seq]

    index_seq = []
    for s in msa_instance.seq:
        index_str = np.zeros(len(s), dtype=int)
        for i in range(len(s)):
            index_str[i] = scoring_matrix.protein_2_index[s[i]]
        index_seq.append(index_str)

    ij = [x for x in itertools.combinations(range(len(index_seq)), 2)]
    with ProcessPoolExecutor(max_workers=threads_to_use) as pool:
        all_anchors = pool.map(unpack_compute_sij,
                            zip(ij,
                            [index_seq]*len(ij),
                            [scoring_matrix.scores]*len(ij),
                            [radius]*len(ij),
                            [threshold]*len(ij)))
        anchor_set.anchor_data = np.concatenate([data for data in all_anchors], axis=0)

    anchor_set.anchor_data[:,LEN] = np.ones((anchor_set.anchor_data.shape[0]))

    maxlen = max(anchor_set.len_seqs)

    #sort the data
    ordering = ((anchor_set.anchor_data[:,SEQ_ST]*maxlen +
                    anchor_set.anchor_data[:,IND_ST])*maxlen*len(msa_instance.seq) +
                            (anchor_set.anchor_data[:,SEQ_END]*maxlen + anchor_set.anchor_data[:,IND_END]))
    indices = np.argsort(ordering)
    anchor_set.anchor_data = anchor_set.anchor_data[indices]

    anchor_set.loaded = True

    return anchor_set


#constructs the anchor set from a given list of anchors
def anchor_set_from_data(msa_instance, anchor_data, with_solution = False, solution = None):
    anchor_set = AnchorSet()
    anchor_set.num_seq = len(msa_instance.seq)
    anchor_set.len_seqs = [len(s) for s in msa_instance.seq]
    nodes = [(anchor_data[i,SEQ_ST], anchor_data[i,IND_ST]) for i in range(anchor_data.shape[0])] + [(anchor_data[i,SEQ_END], anchor_data[i,IND_END]) for i in range(anchor_data.shape[0])]
    anchor_set.anchor_data = anchor_data
    anchor_set.loaded = True
    if with_solution:
        anchor_set.solution = solution
        anchor_set.solution_loaded = True
    return anchor_set


#reads a precomputed anchor set from file
def anchor_set_from_file(filename):
    anchor_set = AnchorSet()
    with open(filename+".meta", 'r') as f:
        try:
            content = f.readlines()
            meta = content[0].split()
            anchor_set.num_seq = int(meta[1])
            anchor_set.minrow = int(meta[3])
            anchor_set.len_seqs = [int(l) for l in meta[5:]]
        except:
            print("Failed to load anchor set meta file ", filename)
            return
    anchor_set.anchor_data = np.reshape(np.genfromtxt(filename+".anchors", delimiter='\t'), (-1,7))
    anchor_set.loaded = True
    if os.path.isfile(filename+".sol"):
        anchor_set.solution = np.genfromtxt(filename+".sol", delimiter='\t')
        anchor_set.solution_loaded = True
    return anchor_set

#unwraps an anchor-set containing anchors of length > 1; returns a new anchor set
def unwrap_anchor_set(anchor_set):
    unwrapped_anchor_set = AnchorSet()
    unwrapped_anchor_set.num_seq = anchor_set.num_seq
    unwrapped_anchor_set.len_seqs = anchor_set.len_seqs
    unwrapped_anchor_set.loaded = True
    anchor_data = []
    unwrapped_anchor_set.minrow = anchor_set.minrow
    for i in range(anchor_set.anchor_data.shape[0]):
        for j in range(int(anchor_set.anchor_data[i,LEN])):
            anchor_data.append([anchor_set.anchor_data[i,SEQ_ST],
                                anchor_set.anchor_data[i,IND_ST]+j,
                                anchor_set.anchor_data[i,SEQ_END],
                                anchor_set.anchor_data[i,IND_END]+j,
                                anchor_set.anchor_data[i,SCORE]/anchor_set.anchor_data[i,LEN],
                                1])
    unwrapped_anchor_set.anchor_data = np.array(anchor_data)
    return unwrapped_anchor_set


##############################################################################################################
##############################################################################################################
##############################################################################################################
#      METHODS FOR ANCHOR-SET PREPROCESSING
##############################################################################################################
##############################################################################################################
##############################################################################################################
#extract only anchors between sequences i and j from the set of anchors
#returns 3D list "sub_anchors" where sub_anchors[i][j] (i < j) is the subset of anchors between sequences i and j
#subsets are stored as lists of indices in anchors, not as explicit anchors
def extract_anchor_subsets(anchor_set):

    sub_anchors = [[[] for j in range(i+1, anchor_set.num_seq)] for i in range(anchor_set.num_seq-1)]

    for i in range(anchor_set.anchor_data.shape[0]):
        sub_anchors[ int(anchor_set.anchor_data[i,SEQ_ST]) ][ int(anchor_set.anchor_data[i,SEQ_END]-anchor_set.anchor_data[i,SEQ_ST]-1) ].append(i)

    #note that each sub_anchors[i][j] is automatically sorted if anchors was

    return sub_anchors


#returns all pairwise local inconsistencies between sequences i,j given a 3d list of anchor subsets
#assumes that each subset of edges is sorted
#IMPORTANT: inconsistencies for each pair are only stored ONCE to save memory
#That is: if anchors a and b are inconsistent, either local_inconsistencies[a] contains b or the other way around
def compute_local_inconsistencies(anchor_set):

    anchor_subsets = extract_anchor_subsets(anchor_set)

    local_inconsistencies = [[] for i in range(anchor_set.anchor_data.shape[0])]

    #anchors are sorted by first index
    #at each iteration i, keep a list of all anchors <= i sorted by second index
    for i in range(len(anchor_subsets)):
        for j in range(len(anchor_subsets[i])):

            if len(anchor_subsets[i][j]) == 0:
                continue

            sec_ordered = [ 0 ] #keep a list of all edges occured so far, sorted by second index
            keys = [anchor_set.anchor_data[0,IND_END]] #helper list for bisect module

            for a in range(1,len(anchor_subsets[i][j])):
                anchor = anchor_set.anchor_data[ anchor_subsets[i][j][a] ]
                x = len(sec_ordered)-1
                while x >= 0 and anchor[IND_END] <= anchor_set.anchor_data[ anchor_subsets[i][j][sec_ordered[x]], IND_END ]:
                    local_inconsistencies[anchor_subsets[i][j][sec_ordered[x]]].append(anchor_subsets[i][j][a])
                    #local_inconsistencies[anchor_subsets[i][j][a]].append(anchor_subsets[i][j][sec_ordered[x]])
                    x = x-1
                pos = bisect.bisect(keys, anchor[IND_END]) #determines where to insert anchor such that ordering of "sec_ordered" is maintained
                keys.insert(pos, anchor[IND_END]) #here, O(n) insertion dominates O(log(n)) position search.. however still faster than naive approach
                sec_ordered.insert(pos, a)

    return local_inconsistencies


#returns a list of all alignment rows
#an alignment row is a consecution of anchors (s1,i)-(s2,j), (s1,i+1)-(s2,j+1), (s1,i+2)-(s2,j+2), ... of length at least "minrow"
#O(E) (assumes that anchors is ordered)
#return value contains indices into "anchor_set" instead of tuples;
#therefore, the return value is only valid in combination with the corresponding anchor_set
def build_alignment_rows(anchor_set):

    index_rows = [-1]*anchor_set.anchor_data.shape[0] #if edge e is in a row and not the first entry of such,
                                 #index_rows[e] will contain the edge index of the previous edge in the row

    prev_checked = 0
    cur_start = 0
    for i in range(1,anchor_set.anchor_data.shape[0]):

        if not anchor_set.anchor_data[i,SEQ_ST] == anchor_set.anchor_data[i-1,SEQ_ST] or not anchor_set.anchor_data[i,IND_ST] == anchor_set.anchor_data[i-1,IND_ST]: #new starting node
            prev_checked = cur_start
            cur_start = i

        while prev_checked < cur_start-1 and ( anchor_set.anchor_data[prev_checked, SEQ_END] < anchor_set.anchor_data[i, SEQ_END] or
            (anchor_set.anchor_data[prev_checked, SEQ_END] == anchor_set.anchor_data[i, SEQ_END] and
            anchor_set.anchor_data[prev_checked, IND_END] < (anchor_set.anchor_data[i, IND_END]-1) )):  #amortized linear runtime
            prev_checked += 1

        if (prev_checked < cur_start and
            anchor_set.anchor_data[prev_checked, IND_END] == anchor_set.anchor_data[i, IND_END]-1 and
            anchor_set.anchor_data[prev_checked,SEQ_END] == anchor_set.anchor_data[i,SEQ_END] and
            anchor_set.anchor_data[prev_checked,IND_ST] == anchor_set.anchor_data[i,IND_ST]-1):
            index_rows[i] = prev_checked

    #backtrack in index row to get the actual rows
    rows = []
    for ei in list(range(len(index_rows)))[::-1]: #inverse traversal
        back_iter = ei
        row = []
        while not index_rows[back_iter] == -1:  #amortized linear runtime
            row.insert(0, back_iter)
            prev = index_rows[back_iter]
            index_rows[back_iter] = -1
            back_iter = prev
        row.insert(0, back_iter)
        if len(row) > 1:
            rows.insert(0, row)

    return rows


#reduces the given AnchorSet only to anchors contained in a row
def reduce_anchors_to_rows(msa_instance, anchor_set, rows, minrow):
    indices = [i for r in rows for i in r]
    indices.sort()
    data_reduced = np.copy(anchor_set.anchor_data[indices])
    if anchor_set.solution_loaded:
        anchors = anchor_set_from_data(msa_instance, data_reduced, True, anchor_set.solution[indices])
    else:
        anchors = anchor_set_from_data(msa_instance, data_reduced)
    anchors.minrow = minrow
    return anchors


#reduces the anchor set to the k best anchors based on score
#if num anchor < k, the set is unchanged (no copy is made in that case, otherwise a new object is returned)
def kBestAnchors(msa_instance, anchor_set, k):

    if (anchor_set.anchor_data.shape[0] <= k):
        return anchor_set
    _, indices = zip(*sorted(zip(anchor_set.anchor_data[:,SCORE], range(anchor_set.anchor_data.shape[0]))))
    indices = list(indices)[-k:]
    indices.sort()
    data_reduced = np.copy(anchor_set.anchor_data[indices])
    if anchor_set.solution_loaded:
        anchors = anchor_set_from_data(msa_instance, data_reduced, True, anchor_set.solution[indices])
    else:
        anchors = anchor_set_from_data(msa_instance, data_reduced)
    return anchors



#given a set of rows, returns a reduced set of anchors based on row contraction
def row_contraction(msa_instance, anchor_set, rows, minrow):

    if len(rows) == 0:
        return AnchorSet()

    #######################################
    #helpers
    #######################################
    class Region():
        def __init__(self, sequence, start, end, top, partner_region = None):
            self.sequence = int(sequence)
            self.start = int(start)
            self.end = int(end)
            self.partner_region = partner_region
            self.top = top
            self.splitpoints_in = [] #marks positions i; start <= i < end, such that a split AFTER i is required

    def determine_intersecting_regions(regions):
        groups = [[0]] #groups regions together that overlap
        cur_end = regions[0].end
        for i in range(1, len(regions)):
            if regions[i].sequence == regions[i-1].sequence:
                if regions[i].start <= cur_end:
                    groups[-1].append(i)
                    cur_end = max(cur_end, regions[i].end)
                else:
                    groups.append([i])
                    cur_end = regions[i].end
            else: #new sequence starts
                groups.append([i])
                cur_end = regions[i].end

        for group_ind in groups:
            group = regions[group_ind[0]:group_ind[-1]+1]
            for r1 in group:
                r1.splitpoints_in = [r1.start-1, r1.end]
                for r2 in group:
                    if not (r1.start == r2.start and r1.end == r2.end):
                        if r2.start >= r1.start and r2.end <= r1.end: #full intersect
                            r1.splitpoints_in.append(r2.start-1)
                            r1.splitpoints_in.append(r2.end)
                        elif r2.start >= r1.start and r2.start <= r1.end: #left intersect
                            r1.splitpoints_in.append(r2.start-1)
                        elif r2.end >= r1.start and r2.end <= r1.end: #right intersect
                            r1.splitpoints_in.append(r2.end)


    def split_regions(regions):

        projected = False

        #split rows accordingly
        rows_as_region_pairs_new = []
        for r in regions:
            if not r.top:
                continue
            split_all = sorted(list(set(r.splitpoints_in+[r.start + sp - r.partner_region.start for sp in r.partner_region.splitpoints_in])))
            if len(split_all)>2:
                projected = True
            for sp_start, sp_end in zip(split_all[:len(split_all)-1], split_all[1:]):
                top_region = Region(r.sequence, sp_start+1, sp_end, True)
                bot_region = Region(r.partner_region.sequence, r.partner_region.start + sp_start+1 - r.start, r.partner_region.start + sp_end - r.start, False)
                top_region.partner_region = bot_region
                bot_region.partner_region = top_region
                rows_as_region_pairs_new.append((top_region, bot_region))

        return projected, rows_as_region_pairs_new


    #######################################
    #######################################

    #create a list of Regions
    top_regions = [ Region(anchor_set.anchor_data[r[0],SEQ_ST], anchor_set.anchor_data[r[0],IND_ST], anchor_set.anchor_data[r[-1],IND_ST], True) for r in rows ]
    bot_regions = [ Region(anchor_set.anchor_data[r[0],SEQ_END], anchor_set.anchor_data[r[0],IND_END], anchor_set.anchor_data[r[-1],IND_END], False) for r in rows ]

    #intialize partners
    for i,r in enumerate(top_regions):
        bot_regions[i].partner_region = r
    for i,r in enumerate(bot_regions):
        top_regions[i].partner_region = r

    #algorithm keeps track of rows as a list of region pairs
    #throughout the iterations, regions may be splitted at intersection points with other regions into smaller parts
    rows_as_region_pairs = [(top_regions[i], bot_regions[i]) for i in range(len(rows))]

    maxlen = max(anchor_set.len_seqs)

    #determine splitpoints project along rows; break if nothing is projected anymore
    while True:
        regions = [x[0] for x in rows_as_region_pairs]+[x[1] for x in rows_as_region_pairs]

        #sort by region start
        regions.sort(key= lambda x: x.sequence*maxlen + x.start)

        determine_intersecting_regions(regions) #fills the "splitpoints_in" list for each region

        projected, rows_as_region_pairs = split_regions(regions)

        if not projected:
            break

    #region pairs
    anchor_data = np.zeros((len(rows_as_region_pairs),7))
    for i, (top, bot) in enumerate(rows_as_region_pairs):
        anchor_data[i,SEQ_ST] = top.sequence
        anchor_data[i,IND_ST] = top.start
        anchor_data[i,SEQ_END] = bot.sequence
        anchor_data[i,IND_END] = bot.start
        anchor_data[i,LEN] = top.end - top.start +1

    #helper dict for anchor - index mapping
    anchors_as_tuple = [((int(anchor_set.anchor_data[i,SEQ_ST]), int(anchor_set.anchor_data[i,IND_ST])),
                    (int(anchor_set.anchor_data[i,SEQ_END]), int(anchor_set.anchor_data[i,IND_END]))) for i in range(anchor_set.anchor_data.shape[0])]
    anchorsind = {a:i for i,a in enumerate(anchors_as_tuple)}

    edge_scores = [[anchor_set.anchor_data[anchorsind[((top.sequence, i), (bot.sequence, bot.start+i-top.start))], SCORE]
                        for i in range(top.start, top.end+1)]
                            for top, bot in rows_as_region_pairs]
    anchor_data[:,SCORE] = np.array([sum(es)/len(es) for es in edge_scores])

    if anchor_set.solution_loaded:
        solution = [np.mean(np.array([anchor_set.solution[anchorsind[((top.sequence, i), (bot.sequence, bot.start+i-top.start))]]
            for i in range(top.start, top.end+1)]))
                for top, bot in rows_as_region_pairs]
        anchors = anchor_set_from_data(msa_instance, anchor_data, True, np.array(solution))
    else:
        anchors = anchor_set_from_data(msa_instance, anchor_data)
    anchors.minrow = minrow
    return anchors


def compute_achor_graph(nodes, edges, k):

    #map edges to indices
    #todo: improve this
    #currently mapping both ordered pairs u,v and v,u to the same index
    #however, there should be a better solution..
    #unordered pairs (sets) are not hashable
    eind = {e : i for e,i in zip(edges + [(n2,n1) for n1,n2 in edges], list(range(len(edges))) + list(range(len(edges))))}

    anchorgraph = nx.Graph()
    anchorgraph.add_nodes_from(range(len(edges))) #edges become nodes

    # 1) find all directed cycles in the graph where:
    # - each locus of each sequence is a nodes
    # - in every sequence j, subsequent loci are connected by a forward edge: s^j_i -> s^j_(i+1)
    # - an undirected edge <-> is inserted between s^j_i and s^j'_i', if the regions around i and i' extended by k to the left and right
    #   have a score of a least t
    gdi = nx.DiGraph()
    gdi.add_nodes_from(nodes)

    for n1,n2 in edges:
        gdi.add_edge(n1,n2)
        gdi.add_edge(n2,n1)

    num_seq = max([s for s,i in nodes])+1
    len_seq = [max([i for s,i in nodes if s == x])+1 for x in range(num_seq)]
    for s in range(num_seq):
        ilast = k
        for i in range(1,len_seq[s]):
            if gdi.out_degree((s,i)) > 0:
                gdi.add_edge((s,ilast), (s,i))
                ilast = i

    gdi.remove_nodes_from(list(nx.isolates(gdi)))

    print("directed graph has ", len(gdi.nodes), " nodes and ", len(gdi.edges), " edges")

    nx.draw(gdi, pos={(s,i):(i,-s) for s,i in nodes})
    plt.draw()
    plt.show()

    cycles = nx.simple_cycles(gdi)
    cycles = [c for c in cycles if len(c) > 2]
    print("directed graph has ", len(cycles), " cycles of length > 2")
    for c in cycles:

        nprev = c[0]
        has_forward_edge = False
        for n in c[1:]:
            if nprev[0] == n[0]:
                has_forward_edge = True
                break
            nprev = n

        if has_forward_edge:
            #find c_top and c_bot
            top_left = 0
            top_right = 0
            bot_left = 0
            bot_right = 0
            for i in range(1, len(c)):
                if c[i][0] < c[top_left][0] or (c[i][0] == c[top_left][0] and c[i][1] < c[top_left][1]):
                    top_left = i
                if c[i][0] < c[top_right][0] or (c[i][0] == c[top_right][0] and c[i][1] > c[top_right][1]):
                    top_right = i
                if c[i][0] > c[bot_left][0] or (c[i][0] == c[bot_left][0] and c[i][1] < c[bot_left][1]):
                    bot_left = i
                if c[i][0] > c[bot_right][0] or (c[i][0] == c[bot_right][0] and c[i][1] > c[bot_right][1]):
                    bot_right = i

            if top_right < bot_left:
                c_1 = list(range(top_right, bot_left+1))
            else:
                c_1 = list(range(top_right, len(c)))+list(range(bot_left+1))
            if bot_right < top_left:
                c_2 = list(range(bot_right, top_left+1))
            else:
                c_2 = list(range(bot_right, len(c)))+list(range(top_left+1))

            #print(c)
            #print(c_1, c_2)


            for j1,j2 in itertools.combinations(c_1, 2):
                j1plus = j1+1 if j1 < len(c)-1 else 0
                j2plus = j2+1 if j2 < len(c)-1 else 0
                if not c[j1][0] == c[j1plus][0] and not c[j2][0] == c[j2plus][0]:
                    anchorgraph.add_edge(eind[(c[j1],c[j1plus])], eind[(c[j2],c[j2plus])], type=0)

            for j1,j2 in itertools.combinations(c_2, 2):
                j1plus = j1+1 if j1 < len(c)-1 else 0
                j2plus = j2+1 if j2 < len(c)-1 else 0
                if not c[j1][0] == c[j1plus][0] and not c[j2][0] == c[j2plus][0]:
                    anchorgraph.add_edge(eind[(c[j1],c[j1plus])], eind[(c[j2],c[j2plus])], type=0)

            for j1 in c_1:
                for j2 in c_2:
                    j1plus = j1+1 if j1 < len(c)-1 else 0
                    j2plus = j2+1 if j2 < len(c)-1 else 0
                    if not c[j1][0] == c[j1plus][0] and not c[j2][0] == c[j2plus][0]:
                        anchorgraph.add_edge(eind[(c[j1],c[j1plus])], eind[(c[j2],c[j2plus])], type=1)


    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    g.remove_nodes_from(list(nx.isolates(g)))

    ccg = [g.subgraph(c) for c in nx.connected_components(g)]
    for sg in ccg:
        for e1,e2 in itertools.combinations(sg.edges, 2):
            e1, e2 = eind[e1], eind[e2]
            if not anchorgraph.has_edge(e1,e2): #if a type 1 edge was not previously added, add a type 0 edge
                anchorgraph.add_edge(e1,e2, type=0)

    #anchorgraph.remove_nodes_from(list(nx.isolates(anchorgraph)))

    return anchorgraph


#builds a network graph based on a given set of anchors
#these anchors may be simple k-mer based locus-to-locus connections or
#larger high scoring regions contracted to a single edge
#the network graph holds a node for every anchor
#anchors are directly connected if
# 1) they are locally inconsistent
# 2) they are in an alignment column
#the type of connection is also given as a binary edge label
#nodes labels are a tuple (norm_pos1, norm_pos2, prior, norm_score)
def build_network_graph(anchor_set):
    return build_network_graph_sp(anchor_set, compute_local_inconsistencies(anchor_set), build_alignment_columns(anchor_set), prior_prob(anchor_set))

def build_network_graph_sp(anchor_set, local_inconsistencies, columns, priors):

    row_graph = nx.Graph()

    plain_rows = [e for row in rows for e in row]
    max_e_ind = max(plain_rows)+1

    #add connections between anchors and rows
    #nodes are automatically added to the graph as needed
    for i in range(len(rows)):
        for e in rows[i]:
            row_graph.add_edge(max_e_ind+i, e)

    #connect the "row-nodes" that themself group up individual anchors
    for c in conn:
        row_graph.add_edge(max_e_ind+conn[0], max_e_ind+conn[1])
        sccs = nx.strongly_connected_components(row_graph)
        for scc in sccs:
            if sorted_anchors[i][0] in scc and sorted_anchors[i][1] in scc:
                for e in alignment_graph.subgraph(scc).edges:
                    #forward edge?
                    if e[0][0] == e[1][0]:
                        #getting here means the selection of edges is INCONSISTENT
                        #therefore, the edge is removed again
                        alignment_graph.remove_edge(sorted_anchors[i][0], sorted_anchors[i][1])
                        alignment_graph.remove_edge(sorted_anchors[i][1], sorted_anchors[i][0])
                        consistent = False
                        break

    return row_graph


#reads a xml file containing subnode name "seq-data" from which the aligned sequences are gathered
#returns a binary array with the same length as "anchors" where 1 indicates that an edge is also part of the reference solution
def read_solution(filename, anchor_set):
    instance = MSAInstance(filename)
    assert(len(instance.seq)>= 2, "must have at least 2 sequences")
    anchor_set.len_ref_seq = len(instance.seq[0])
    anchor_set.gap_counts = np.zeros( (len(instance.seq), max([len(s) for s in instance.seq])), dtype=int )
    i = 0
    for seq in instance.seq:
        gap_cnt = 0
        j = 0
        for c in seq.strip():
            if c == '-':
                gap_cnt += 1
            else:
                anchor_set.gap_counts[i][j] = gap_cnt
                j += 1
        i += 1

    anchor_set.solution = np.zeros(anchor_set.anchor_data.shape[0])
    for i in range(anchor_set.anchor_data.shape[0]):
        for j in range(int(anchor_set.anchor_data[i,LEN])):
            if (anchor_set.anchor_data[i,IND_ST] + j + anchor_set.gap_counts[int(anchor_set.anchor_data[i,SEQ_ST])][int(anchor_set.anchor_data[i,IND_ST])+j] ==
                anchor_set.anchor_data[i,IND_END] + j + anchor_set.gap_counts[int(anchor_set.anchor_data[i,SEQ_END])][int(anchor_set.anchor_data[i,IND_END])+j]):
                anchor_set.solution[i] += 1
        if anchor_set.anchor_data[i,LEN] > 0:
            anchor_set.solution[i] /= anchor_set.anchor_data[i,LEN]

    anchor_set.solution_loaded = True


#computes the Jaccard-Index of two edge selections
#in: two binary arrays
def jaccard_index(setA, setB):

    intersect = np.sum(setA*setB)
    union = np.sum((setA+setB)>0)
    if union == 0:
        return 0
    return intersect/union


#given a numpy array of predictions, returns the "greedy-best" consistent choice
#based on the predicted probabilities
#O(E^2)
#TODO: O(E) possible with optimizations?
#returns a set of indices into anchors denoting the greedily selected subset
def greedy_best(anchor_set, pred):

    QUALITY_THRESHOLD = 0.0
    assert anchor_set.anchor_data.shape[0]==len(pred), "anchors and predictions do not have matching lengths"

    #simple predictions based on score for comparison, comment out in case
    # max_score = np.max(anchor_set.anchor_data[:, SCORE])
    # for i in range(anchor_set.anchor_data.shape[0]):
    #     pred[i] = anchor_set.anchor_data[i, SCORE]/max_score

    anchors = [((anchor_set.anchor_data[i, SEQ_ST], anchor_set.anchor_data[i, IND_ST]), (anchor_set.anchor_data[i, SEQ_END], anchor_set.anchor_data[i, IND_END]))
                for i in range(anchor_set.anchor_data.shape[0])]

    alignment_graph = nx.DiGraph()

    #add all foward edges
    extracted_nodes = list(set([n[0] for n in anchors] + [n[1] for n in anchors]))
    maxlen = max([n[1] for n in extracted_nodes])+1
    extracted_nodes.sort(key=lambda n: n[0]*maxlen + n[1])
    for i in range(1,len(extracted_nodes)):
        if extracted_nodes[i-1][0] == extracted_nodes[i][0]:
            alignment_graph.add_edge(extracted_nodes[i-1], extracted_nodes[i])

    sorted_pred, sorted_anchors, sorted_indices = zip(*sorted(zip(pred, anchors, range(len(pred))), reverse=True, key = lambda x : x[0]))

    selected_subset = np.zeros(anchor_set.anchor_data.shape[0])

    for i in range(len(sorted_pred)):

        if sorted_pred[i] < QUALITY_THRESHOLD:
            break

        alignment_graph.add_edge(sorted_anchors[i][0], sorted_anchors[i][1])
        alignment_graph.add_edge(sorted_anchors[i][1], sorted_anchors[i][0])

        consistent = True

        #TODO: the following can be optimized by exploiting the fact, that at each step, the consistency of the current
        #alignment graph is given and there are no cycles containing forward edges
        sccs = nx.strongly_connected_components(alignment_graph) #O(E)
        for scc in sccs:
            #the only strongly connected component that coanchorsuld contain an inconsistent cycle
            #is the one containing both nodes of the newlyanchors inserted edge
            if sorted_anchors[i][0] in scc and sorted_anchors[i][1] in scc:
                for e in alignment_graph.subgraph(scc).edges:
                    #forward edge?
                    if e[0][0] == e[1][0]:
                        #getting here means the selection of edges is INCONSISTENT
                        #therefore, the edge is removed again
                        alignment_graph.remove_edge(sorted_anchors[i][0], sorted_anchors[i][1])
                        alignment_graph.remove_edge(sorted_anchors[i][1], sorted_anchors[i][0])
                        consistent = False
                        break

        if consistent:
            selected_subset[sorted_indices[i]] = 1.0

    return selected_subset


#"unwraps" a binary vector indicating "anchor chosen" or "anchor not chosen" to
#a binary vector where each anchor of length L is represented by L anchors of length 1
#with each having an individual binary indicator
def unwrap_selection(anchor_set, selection):
    assert anchor_set.anchor_data.shape[0]==selection.shape[0], "anchors and selection do not have matching lengths"
    unwrapped_selection = []
    for i in range(anchor_set.anchor_data.shape[0]):
        for j in range(int(anchor_set.anchor_data[i,LEN])):
            unwrapped_selection.append(selection[i])
    return np.array(unwrapped_selection)


#normalizes the anchor data
def normalize_anchor_set(anchor_set, max_score, max_len):

    norm_data = np.copy(anchor_set.anchor_data).astype(dtype=np.float32)
    max_seq_len = max(anchor_set.len_seqs)
    norm_data[:,SEQ_ST] /= anchor_set.num_seq
    norm_data[:,SEQ_END] /= anchor_set.num_seq
    norm_data[:,IND_ST] /= max_seq_len
    norm_data[:,IND_END] /= max_seq_len
    norm_data[:,SCORE] /= max_score
    norm_data[:,LEN] /= max_len

    return norm_data
