import networkx as nx
import argparse
import time
from graph_embedding.slaq.exploring import edge_list_to_adjacency, read_graph, compare_graphs
from graph_embedding.slaq.slaq import netlsd, vnge



def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--graphone", help='CSV input file, (source, target, weight)', required=True)
	parser.add_argument("--graphtwo", help='CSV input file, (source, target, weight)', required=True)
	parser.add_argument("--delim", help='file delimiter', required=False, default='\t')
	args = parser.parse_args()

	graph_one_file = args.graphone
	graph_two_file = args.graphtwo
	delim = args.delim

	start = time.time()
	graph_one_edge_list, graph_one_remapping, graph_one_max_id = read_graph(graph_one_file, delim, skip_mapping=True)
	end = time.time()
	print(f"graph one has {len(graph_one_edge_list)}, max_id: {graph_one_max_id}, read in {end-start} secs")
	
	start = time.time()
	graph_one = edge_list_to_adjacency(graph_one_edge_list, graph_one_max_id+1)
	end = time.time()
	graph_one_edge_list = None
	print(f"graph one adjacency list in {end-start} secs")
	start = time.time()
	graph_two_edge_list, graph_two_remapping, graph_two_max_id = read_graph(graph_two_file, delim, skip_mapping=True)
	end = time.time()
	print(f"graph two has {len(graph_two_edge_list)}, max_id: {graph_two_max_id}, read in {end-start} secs")
	start = time.time()
	graph_two = edge_list_to_adjacency(graph_two_edge_list, graph_two_max_id+1)
	end = time.time()
	graph_two_edge_list = None
	print(f"graph two adjacency list in {end-start} secs")

	start = time.time()
	result = compare_graphs(graph_one, graph_two)
	end = time.time()
	print (f"found {result} in {end-start} secs")

if __name__ == '__main__':
	main()