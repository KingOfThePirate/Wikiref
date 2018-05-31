# Program to print BFS traversal from a given source
# vertex. BFS(int s) traverses vertices reachable
# from s.
from collections import defaultdict
import pickle,sys
# This class represents a directed graph using adjacency
# list representation
def save_pickle(object_to_save,filename):
    fileObject = open(filename,'wb')
    pickle.dump(a,fileObject) 
    fileObject.close()

def load_pickle(filename):
    fileObject = open(filename,'rb') 
    b = pickle.load(fileObject) 
    fileObject.close()
    return b

class Graph:
 
    # Constructor
    def __init__(self):
 
        # default dictionary to store graph
        self.graph = defaultdict(list)
 
    # function to add an edge to graph
    def addEdge(self,u,v):
        self.graph[u].append(v)
 
    # Function to print a BFS of graph
    def BFS(self, s):
 
        # Mark all the vertices as not visited
        # visited = [False]*(len(self.graph))
        visited = [False]*V
        # Create a queue for BFS
        queue = []
 
        # Mark the source node as visited and enqueue it
        queue.append(s)
        visited[s] = True
 
        while queue:
 
            # Dequeue a vertex from queue and print it
            s = queue.pop(0)
            # print s,
 
            # Get all adjacent vertices of the dequeued
            # vertex s. If a adjacent has not been visited,
            # then mark it visited and enqueue it
            for i in self.graph[s]:
                if visited[i] == False:
                    queue.append(i)
                    visited[i] = True

id_to_cat = load_pickle("id_to_cat")
cat_to_id = load_pickle("cat_to_id")
cat_graph = load_pickle("cat_net_adj")
V = load_pickle("number_cat")


g = Graph()
for a in cat_graph:
    for b in cat_graph[a]:
        g.addEdge(a,b)

root_id = cat_to_id['Contents']

cat_a = cat_to_id['Corpus_linguistics']
cat_b = cat_to_id['Finite_automata']

g.BFS(root_id)