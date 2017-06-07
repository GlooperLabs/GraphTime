import matplotlib.pyplot as plt
import networkx as nx
import tempfile

""" A library for visualisation function
The aim is to use networkX for graph visualisation
and analysis. We may need to extend this somewhat
to deal with dynamics.
"""

""" This is deprecated, decided to associate networkx graph
directly with the dynamic graph object, rather than go via gexf
def plot_graph(DGM, t):
    # Given DynamicGraphModel object, plots
    # the graph at a particular time

    temp = tempfile.NamedTemporaryFile(mode='w+')
    try:
            # Temporarily write to file
        temp.write(DGM.graphs[t].gexf)
        temp.seek(0)
        G = nx.read_gexf(temp.name, node_type=None, relabel=False)

    finally:
        temp.close()

    nx.draw(G)
    plt.show()  # For viewing inline
"""