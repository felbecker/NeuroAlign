import numpy as np
import networkx as nx



#fastest but ignores consistency
#this output is not suitable for constructing complete alignments
#it can be used to test precision and recall
def max_likely(msa, memberships):
    return np.argmax(memberships, axis=1)



#fast but only ensures sequence-level consistency
# takes raw predicted memberships and outputs sequence-consistent column indices
# that fulfill: i < j -> col[i] < col[j]
# inconsistencies in the prediction are resolved by introducing new columns
def seq_consistent(msa, memberships):
    memberships = np.copy(memberships) #we do not want to modify the original array
    cols = -np.ones(memberships.shape[0])
    lsum = 0
    for l in msa.seq_lens:
        seq_mem = memberships[lsum:(lsum+l),:]
        pos_done = 0
        while pos_done < l:
            i,j = np.unravel_index(np.argmax(seq_mem, axis=None), seq_mem.shape) #find pos i and col j with maximum probability
            if seq_mem[i,j] == -1:
                break
            cols[lsum+i] = j
            #we don't want to pick position i again
            seq_mem[i,:] = -1
            #all positions that come before position i in the sequence will not be able chooce a column greater or equal j
            seq_mem[:i,j:] = -1
            #all positions that come after position i in the sequence will not be able chooce a column less or equal j
            seq_mem[(i+1):,:(j+1)] = -1
            pos_done += 1
        lsum += l
    # -1-columns indicate inconsistencies in the prediction
    # the model that made the predictions has a soft mechanism to prevent this case,
    # but we have to handle the rare cases where this soft mechanism fails
    # if a -1 column occurs, it's always a bad thing, although it can be resolved
    # by introducing a new column containing only the position i with cols[i] = -1
    num_inconsistent = np.sum(cols == -1)
    if num_inconsistent > 0:
        print(num_inconsistent, " positions could not be assigned consistently to columns. Inserting additional columns for them.")
        lsum = 0
        for l in msa.seq_lens:
            last_valid_column = -1
            for i,j in enumerate(cols[lsum:(lsum+l)]):
                if j == -1:
                    #introduce a new column
                    cols[lsum+i] = last_valid_column+1
                    #shift all entries with a column greater equal last_valid_column by one
                    cols += (cols >= last_valid_column+1)
                else:
                    last_valid_column = j #j's are strictly increasing
            lsum += l
    return cols



#returns a consistent selection
#greedily select the position -column pair with highest probability that is
#consistent with previous selections
def fully_consistent(msa, memberships):
    #2d descending argsort the predicted memberships
    greedy_sorted = np.dstack(np.unravel_index(np.argsort(-memberships, axis=None), memberships.shape))[0]
    #greedy_sorted[:,1] = np.max(greedy_sorted[:,1]) - greedy_sorted[:,1]
    #results; -1 indicates a uncertain column, 2 positions in incertain columns compare as "not aligned"
    checker = ConsistencyChecker(msa, memberships.shape)
    for pos,col in greedy_sorted:
        checker.try_add(pos, col)
    return checker.get_result()



class ConsistencyChecker():
    def __init__(self, msa, n):
        self.picks = -np.ones(n[0], dtype=np.int32)
        self.alignment_graph = nx.DiGraph()
        edges = []
        self.lsum = 0
        for l in msa.seq_lens:
            if l == 1:
                self.alignment_graph.add_node(self.lsum)
            else:
                edges.extend([(self.lsum+i, self.lsum+i+1) for i in range(l-1)])
            self.lsum += l
        edges.extend([(self.lsum+i, self.lsum+i+1) for i in range(n[1]-1)])
        self.alignment_graph.add_edges_from(edges)

    def test_cycle(self, graph, u, v):
        for _,t in nx.bfs_edges(graph, v):
            if t == u: #cycle containing edge (u,v) found
                return True
        for _,t in nx.bfs_edges(graph, u):
            if t == v: #cycle containing edge (v,u) found
                return True
        return False

    def try_add(self, pos, col):
        cycle = self.test_cycle(self.alignment_graph, self.lsum+col, pos)
        #no cycle introduced by adding edge (pos, lsum+col)
        #edge is consistent
        if not cycle:
            self.alignment_graph.add_edge(self.lsum+col, pos)
            self.alignment_graph.add_edge(pos, self.lsum+col)
            self.picks[pos] = col


    def get_result(self):
        return self.picks
