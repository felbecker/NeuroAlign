import numpy as np
import networkx as nx


#fast but ignores consistency
def greedy_col_max_likely(msa, memberships):
    return np.argmax(memberships, axis=1)


#returns a consistent selection
#greedily select the position -column pair with highest probability that is
#consistent with previous selections
def greedy_consistent(msa, memberships):
    #2d descending argsort the predicted memberships
    greedy_sorted = np.dstack(np.unravel_index(np.argsort(-memberships, axis=None), memberships.shape))[0]
    #results; -1 indicates a uncertain column, 2 positions in incertain columns compare as "not aligned"
    checker = ConsistencyChecker(msa, memberships.shape)
    print(np.argmax(memberships, axis=1))
    for pos,col in greedy_sorted:
        checker.try_add(pos, col)
    return checker.get_result()


class ConsistencyChecker():
    def __init__(self, msa, n):
        self.picks = -np.ones(n[0])
        self.alignment_graph = nx.DiGraph()
        edges = []
        self.lsum = 0
        for l in msa.seq_lens:
            edges.extend([(self.lsum+i, self.lsum+i+1) for i in range(l-1)])
            self.lsum += l
        edges.extend([(self.lsum+i, self.lsum+i+1) for i in range(n[1]-1)])
        self.alignment_graph.add_edges_from(edges)

    def test_cycle(self, graph, u, v):
        graph.add_edge(u, v)
        cycle = False
        for _,t in nx.bfs_edges(graph, v):
            if t == u: #cycle containing edge (u,v) found
                cycle = True
                break
        graph.remove_edge(u, v)
        return cycle

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
