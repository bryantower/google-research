import networkx as nx
import csv
import argparse
import scipy
import numpy as np
import matplotlib.pyplot as plt
import community
from graph_embedding.slaq.example import netlsd_naive, vnge_naive
from graph_embedding.slaq.slaq import netlsd
from graph_embedding.slaq.slaq import vnge
from graph_embedding.slaq.util import laplacian

"""
This file takes in a standard edge list file of type (source, target, weight)
if the weight is missing then it assumes unweighted edges.
"""


def edge_list_to_adjacency(edge_list, num_nodes):
	adjacency = scipy.sparse.dok_matrix((num_nodes, num_nodes),
													dtype=np.float32)
	for s, t, w in edge_list:
		adjacency[s, t] = w
		adjacency[t, s] = w
	adjacency = adjacency.tocsr()
	adjacency.data = np.ones(
		adjacency.data.shape,
		dtype=np.float32)  # Set all elements to one in case of duplicate rows.
	return adjacency

def nx_graph_to_adjacency(nxg):
	num_nodes = nxg.number_of_nodes()
	adjacency = scipy.sparse.dok_matrix((num_nodes, num_nodes),
													dtype=np.float32)
	for s, t, w in nxg.edges(None, data='weight', default=1):
		adjacency[s, t] = w
		adjacency[t, s] = w
	adjacency = adjacency.tocsr()
	adjacency.data = np.ones(
		adjacency.data.shape,
		dtype=np.float32)  # Set all elements to one in case of duplicate rows.
	return adjacency


def read_graph(infile, delim, has_header=False, ingnore_weights=True, skip_mapping=False):
	nid_to_remapped_id = {}
	edge_list = []
	next_nid = 0
	max_id = -1
	with open(infile, 'r', encoding='utf-8') as ifile:
		reader = csv.reader(ifile, delimiter=delim)
		if has_header:
			header = next(reader)
		for row in reader:
			if len(row) == 2 or ingnore_weights:
				source,target = row[0], row[1]
				weight = 1.0
			elif len(row) >= 3:
				source,target,weight = row[0], row[1], float(row[2])
			else:
				print (f"invalid row format only {len(row)} columns")
				raise Exception("invalid row format")
			if skip_mapping:
				source_id = int(source)
				target_id = int(target)
				max_id = max(source_id, max_id)
				max_id = max(target_id, max_id)
			else:
				if source not in nid_to_remapped_id:
					nid_to_remapped_id[source] = next_nid
					next_nid += 1 
				if target not in nid_to_remapped_id:
					nid_to_remapped_id[target] = next_nid
					next_nid += 1 
				source_id = nid_to_remapped_id[source]
				target_id = nid_to_remapped_id[target]

			edge_list.append([source_id, target_id, weight])
			if not skip_mapping:
				max_id = len(nid_to_remapped_id)

	return edge_list, nid_to_remapped_id, max_id

def compare_graphs(graph1, graph2):
	lsd_slaq1 = netlsd(graph1)
	lsd_slaq2 = netlsd(graph2)
	netlsd_error = np.linalg.norm(lsd_slaq1 - lsd_slaq2) / np.linalg.norm(lsd_slaq1)
	#print('NetLSD approximation error:', netlsd_error)

	vnge_slaq1 = vnge(graph1)
	vnge_slaq2 = vnge(graph2)
	vnge_error = np.linalg.norm(vnge_slaq1 - vnge_slaq2) / np.linalg.norm(vnge_slaq1)
	#print('VNGE approximation error:', vnge_error)
	return [netlsd_error, vnge_error]

def make_erdos_reinyi(num_nodes, edge_probability, seed=None, directed=False):
	return nx.fast_gnp_random_graph(num_nodes, edge_probability, seed, directed)


def make_sbm_graph(num_blocks, nodes_per_block, in_prob, out_prob):
	size_list = [nodes_per_block]* num_blocks
	probabilites = []
	for b in range(num_blocks):
		row = []
		for b2 in range(num_blocks):
			if b == b2:
				row.append(in_prob)
			else:
				row.append(out_prob)
		probabilites.append(row)
	return nx.generators.community.stochastic_block_model(size_list, probabilites)


def plot_data_second_axis(x, y1, y2=None, y3=None, title=None, xlabel=None, ylabel=None, y2label=None, outfile=None):
	fig, ax_errors = plt.subplots()
	ax_modularity = ax_errors.twinx()
	if ylabel is not None:
		ax_errors.set_ylabel(ylabel)
	if xlabel is not None:
		ax_errors.set_xlabel(xlabel)
	if title is not None:
		plt.title(title)
	line1, = ax_errors.plot(x, y1)
	line_labels = ['NetLSD']
	lines = [line1]
	if y2 is not None:
		line12, = ax_errors.plot(x, y2)
		line_labels += ['VNGE']
		lines += [line12]
	if y3 is not None:
		ax_modularity.set_ylabel(y2label)
		line2, = ax_modularity.plot(x, y3, color="red")
		lines += [line2]
		line_labels += ['Modularity']
	plt.legend(lines, line_labels)
	if outfile is None:
		plt.show()
	else:
		plt.savefig(outfile)


def plot_data(x, y1, y2=None, title=None, xlabel=None, ylabel=None, outfile=None):
	if ylabel is not None:
		plt.ylabel(ylabel)
	if xlabel is not None:
		plt.xlabel(xlabel)
	if title is not None:
		plt.title(title)
	line1, = plt.plot(x, y1)
	line_labels = ['NetLSD']
	lines = [line1]
	if y2 is not None:
		line12, = plt.plot(x, y2)
		line_labels += ['VNGE']
		lines += [line12]
	plt.legend(lines, line_labels)
	#plt.title(f'Gamma = {gamma}')
	if outfile is None:
		plt.show()
	else:
		plt.savefig(outfile)

def plot_data_different(data, title=None, xlabel=None, ylabel=None, outfile=None):
	"""Expects three lists in the data map
	"""
	if ylabel is not None:
		plt.ylabel(ylabel)
	if xlabel is not None:
		plt.xlabel(xlabel)
	if title is not None:
		plt.title(title)
	line_labels = []
	lines = []
	for prob_int in sorted(data.keys()):
		x = data[prob_int][0]
		y = data[prob_int][2]
		line, = plt.plot(x, y)
		line_labels += ['NetLSD - delta probability: {:.2f}'.format(prob_int*0.01 - 0.1)]
		lines += [line]
		plt.legend(lines, line_labels)

	if outfile is None:
		plt.show()
	else:
		plt.savefig(outfile)


def graph_delta_p(num_nodes=100):
	original_p = 0.05
	g1 = make_erdos_reinyi(num_nodes, original_p, seed=None)
	graph1 = nx_graph_to_adjacency(g1) 

	delta_p = []
	vnge_res = []
	netlsd_res = []
	for p_int in range(5, 100, 5):
		p = 0.01*p_int
		g2 = make_erdos_reinyi(num_nodes, p, seed=None)
		graph2 = nx_graph_to_adjacency(g2) 
		netlsd_error, vnge_error = compare_graphs(graph1, graph2)
		delta_p.append(p - original_p)
		netlsd_res.append(netlsd_error)
		vnge_res.append(vnge_error)

	title = f'{num_nodes} Nodes'
	ylabel='Error'
	xlabel='Delta P'
	plot_data(delta_p, netlsd_res, vnge_res, title, xlabel, ylabel)

def graph_node_size(prob=0.05):
	original_num_nodes = 20
	g1 = make_erdos_reinyi(original_num_nodes, edge_probability=prob, seed=None)
	graph1 = nx_graph_to_adjacency(g1) 

	num_nodes_diff = []
	vnge_res = []
	netlsd_res = []
	for num_nodes in range(20, 501, 20):
		g2 = make_erdos_reinyi(num_nodes, prob, seed=None)
		graph2 = nx_graph_to_adjacency(g2) 
		netlsd_error, vnge_error= compare_graphs(graph1, graph2)
		num_nodes_diff.append(num_nodes - original_num_nodes)
		vnge_res.append(netlsd_error)
		netlsd_res.append(vnge_error)

	title = f'{prob} Edge Probability'
	ylabel='Error'
	xlabel='Delta Num Nodes'
	plot_data(num_nodes_diff, netlsd_res, None, title, xlabel, ylabel)

def add_nodes(grph, num_to_add):
	next_id = grph.number_of_nodes()
	for i in range(num_to_add):
		grph.add_node(next_id)
		next_id += 1


def graph_node_size_v2(prob=0.05):
	original_num_nodes = 50
	g1 = make_erdos_reinyi(original_num_nodes, edge_probability=prob, seed=None)
	graph1 = nx_graph_to_adjacency(g1) 

	num_nodes_diff = []
	vnge_res = []
	netlsd_res = []
	g2 = g1.copy(False)
	g1_copy = g1.copy(False)
	for i in range(0, 21):
		g1_copy = g1.copy(False)
		add_nodes(g1_copy, 50*i)
		print (f"Current number of nodes {g1_copy.number_of_nodes()}")
		graph1 = nx_graph_to_adjacency(g1_copy) 
		g2 = make_erdos_reinyi(original_num_nodes + 50*i, edge_probability=prob, seed=None)
		graph2 = nx_graph_to_adjacency(g2) 
		netlsd_error, vnge_error= compare_graphs(graph1, graph2)
		num_nodes_diff.append(g2.number_of_nodes() - original_num_nodes)
		vnge_res.append(netlsd_error)
		netlsd_res.append(vnge_error)

	title = f'{prob} Edge Probability'
	ylabel='Error'
	xlabel='Delta Num Nodes'
	plot_data(num_nodes_diff, netlsd_res, None, title, xlabel, ylabel)

def graph_node_size_v3():
	prob = 0.10
	original_num_nodes = 50
	g1 = make_erdos_reinyi(original_num_nodes, edge_probability=prob, seed=None)
	graph1 = nx_graph_to_adjacency(g1) 

	probabilities_to_data = {}
	for prob_int in range(10, 101, 10):
		prob = prob_int *.01
		num_nodes_diff = []
		vnge_res = []
		netlsd_res = []
		g2 = g1.copy(False)
		for i in range(21):
			g1_copy = g1.copy(False)
			add_nodes(g1_copy, 50*i)
			print (f"probability: {prob}, Current number of nodes {g1_copy.number_of_nodes()}")
			graph1 = nx_graph_to_adjacency(g1) 
			g2 = make_erdos_reinyi(original_num_nodes + 50*i, edge_probability=prob, seed=None)
			graph2 = nx_graph_to_adjacency(g2) 
			netlsd_error, vnge_error= compare_graphs(graph1, graph2)
			num_nodes_diff.append(g2.number_of_nodes() - original_num_nodes)
			vnge_res.append(netlsd_error)
			netlsd_res.append(vnge_error)
		probabilities_to_data[prob_int] = [num_nodes_diff, vnge_res, netlsd_res]

	title = f'Edge Probability Sweep'
	ylabel='Error'
	xlabel='Delta Num Nodes'
	plot_data_different(probabilities_to_data, title, xlabel, ylabel)




def sbm_vs_er():
	g1 = make_erdos_reinyi(1000, 0.1, seed=None)
	graph1 = nx_graph_to_adjacency(g1) 
	block_list = []
	netlsd_errors = []
	vnge_errors = []
	for blocks in [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
		nodes = 1000 // blocks
		g2 = make_sbm_graph(blocks, nodes, 0.5, 0.01)
		graph2 = nx_graph_to_adjacency(g2) 
		netlsd_error, vnge_error = compare_graphs(graph1, graph2)
		print (f"blocks: {blocks}, nodes: {nodes}, netlsd: {netlsd_error}, vnge: {vnge_error}")
		block_list.append(blocks)
		netlsd_errors.append(netlsd_error)
		vnge_errors.append(vnge_error)
	title = "SBM vs ER"
	ylabel = "Error"
	xlabel = "Blocks"
	plot_data(block_list, netlsd_errors, vnge_errors, title, xlabel, ylabel)
		
def sbm_vs_sbm():
	#g1 = make_erdos_reinyi(1000, 0.1, seed=None)
	g1 = make_sbm_graph(20, 50, 0.5, 0.01)
	graph1 = nx_graph_to_adjacency(g1) 
	louvain_partition = community.best_partition(g1)
	modularity1 = community.modularity(louvain_partition, g1)
	print (f'lovain Modularity is {modularity1}')
	block_list = []
	netlsd_errors = []
	vnge_errors = []
	modularities = []
	for blocks in [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
		nodes = 1000 // blocks
		g2 = make_sbm_graph(blocks, nodes, 0.5, 0.01)
		graph2 = nx_graph_to_adjacency(g2) 
		louvain_partition = community.best_partition(g2)
		modularity2 = community.modularity(louvain_partition, g2)
		modularities.append(modularity2)
		netlsd_error, vnge_error = compare_graphs(graph1, graph2)
		print (f"blocks: {blocks}, nodes: {nodes}, netlsd: {netlsd_error}, vnge: {vnge_error}, modularirty: {modularity2}")
		block_list.append(blocks)
		netlsd_errors.append(netlsd_error)
		vnge_errors.append(vnge_error)
	title = "SBM vs SBM, static graph modularity {:3f}".format(modularity1)
	ylabel = "Error"
	xlabel = "Blocks"
	y2label = "Modularity"
	plot_data_second_axis(block_list, netlsd_errors, vnge_errors, modularities, title, xlabel, ylabel, y2label)
		

def main2():
	g1 = make_erdos_reinyi(2000, 0.5, seed=None)
	graph1 = nx_graph_to_adjacency(g1) 
	g2 = make_erdos_reinyi(2000, 0.5, seed=None)
	graph2 = nx_graph_to_adjacency(g2) 
	print("close graphs")
	print (f'edges g1 {nx.number_of_edges(g1)}, g2: {nx.number_of_edges(g2)}')
	result1 = compare_graphs(graph1, graph2)
	stats1 = [g1.number_of_nodes(), g2.number_of_nodes(), nx.number_of_edges(g1), nx.number_of_edges(g2)]
	print (stats1 + result1)

	g1 = make_erdos_reinyi(2000, 0.05, seed=None)
	graph1 = nx_graph_to_adjacency(g1) 
	g2 = make_erdos_reinyi(2000, 0.75, seed=None)
	graph2 = nx_graph_to_adjacency(g2) 
	print("far graphs")
	print (f'edges g1 {nx.number_of_edges(g1)}, g2: {nx.number_of_edges(g2)}')
	result2 = compare_graphs(graph1, graph2)
	stats2 = [g1.number_of_nodes(), g2.number_of_nodes(), nx.number_of_edges(g1), nx.number_of_edges(g2)]
	print (stats2 + result2)

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--infile", help='CSV input file, (source, target, weight)', required=True)
	parser.add_argument("--delim", help='file delimiter', required=False, default='\t')
	args = parser.parse_args()

	infile = args.infile
	delim = args.delim
	edge_list, remapping = read_graph(infile, delim=delim)
	print (f"Read graph with {len(remapping)} nodes and {len(edge_list)} edges.")
	graph = edge_list_to_adjacency(edge_list, len(remapping))
	print (f"computed adjacency matrix for graph")

	lsd_full = netlsd_naive(graph)
	lsd_slaq = netlsd(graph)

	print('NetLSD approximation error:',
		np.linalg.norm(lsd_full - lsd_slaq) / np.linalg.norm(lsd_full))

	vnge_full = vnge_naive(graph)
	vnge_slaq = vnge(graph)

	print('VNGE approximation error:',
		np.linalg.norm(vnge_full - vnge_slaq) / np.linalg.norm(vnge_full))
	return

if __name__ == '__main__':
	sbm_vs_sbm()
	#graph_delta_p(500)
	#graph_node_size_v2(0.25)
	#graph_node_size_v3()