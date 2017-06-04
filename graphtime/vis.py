import matplotlib.pyplot as plt
import networkx as nx

""" A library for visualisation function
The aim is to use networkX for graph visualisation
and analysis. We may need to extend this somewhat
to deal with dynamics."""

def plot_graph(DGM, t):
	# Given DynamicGraphModel object, plots
	# the graph at a particular time

	G = nx.read_gexf(DGM.graphs[t].gexf, node_type=None, relabel=False)
	nx.draw(G)
	plt.show()	# For viewing inline
