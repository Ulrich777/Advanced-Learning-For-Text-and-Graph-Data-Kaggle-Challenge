import os
import networkx as nx
import re
import numpy as np
import sys
import pickle
import time
import random


def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [atoi(c) for c in re.split('(\d+)', text)]

def build_graphs(edgelists, path_to_data=''):
    graphs = []

    for idx, edgelist in enumerate(edgelists):
        g = nx.read_edgelist(path_to_data + 'edge_lists/' + edgelist)
        mapping = {node:int(node) for node in g.nodes()}
        g = nx.relabel_nodes(g,mapping)
        graphs.append(g)
    
        # track process
        sys.stderr.write('\rGraph: %d/%d' % (idx+1, len(edgelists)))
        sys.stderr.flush()
    
    return graphs

def get_graph_features(G, all_embds, nmax=15):

    #n = len(G.nodes())
    
    node2id = {node:i for i, node in enumerate(G.nodes())}
    #id2node = {i:node for node,i in node2id.items()}

    adj = np.zeros((nmax,nmax))
    embds = np.zeros((nmax, all_embds.shape[1]))

    for i in G.nodes():
        embds[node2id[i]] = all_embds[i]
        for j in G.neighbors(i):
            adj[node2id[j],node2id[i]] = 1
    
    return adj, embds

def tensorize_graphs(graphs, embds, nmax=29):
    Adjs, Ids = [], []
    for graph in graphs:
        adj, embds_g = get_graph_features(graph, embds, nmax=nmax)
        Adjs.append(adj)
        Ids.append(embds_g)
        
    ADJ = np.array(Adjs)
    EMB = np.array(Ids)
    return ADJ, EMB

def save_obj(path, name, obj ):
    with open(path + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(path, name ):
    with open(path + name + '.pkl', 'rb') as f:
        return pickle.load(f)


#RESET = False
#nmax = 29

def load_data(nmax=29, RESET=False, path_to_data='n'):
    print('Loading all the embeddings')
    embds = np.load(path_to_data + 'embeddings.npy') 

    if RESET:
    
    
        print('retrieving list of edges........\n')
        edgelists = os.listdir(path_to_data + 'edge_lists/')
        edgelists.sort(key=natural_keys)

        print('building graphs..............')
        graphs  = build_graphs(edgelists, path_to_data = path_to_data)
    
        print('saving graphs..........')
        save_obj(path_to_data, 'graphs', graphs)

        print('retrieving training and test ids........\n')
        with open(path_to_data + 'train_idxs.txt', 'r') as file:
            train_idxs = file.read().splitlines()
    
        with open(path_to_data + 'test_idxs.txt', 'r') as file:
            test_idxs = file.read().splitlines()
    
        train_idxs = [int(elt) for elt in train_idxs]
        test_idxs = [int(elt) for elt in test_idxs]

        print('generating graph tensors...........\n')
        nmax = max([len(graph.nodes()) for graph in graphs])

        graphs_opt = [graphs[ind] for ind in train_idxs]
        graphs_test = [graphs[ind] for ind in test_idxs]
    
    

        ADJ_OPT, EMB_OPT = tensorize_graphs(graphs_opt, embds, nmax=nmax)
        ADJ_TEST, EMB_TEST = tensorize_graphs(graphs_test, embds, nmax=nmax)

        print('saving the graph tensors.......\n')
        np.save(path_to_data + 'ADJ_OPT.npy', ADJ_OPT)
        np.save(path_to_data + 'ADJ_TEST.npy', ADJ_TEST)
        np.save(path_to_data + 'EMB_OPT.npy', EMB_OPT)
        np.save(path_to_data + 'EMB_TEST.npy', EMB_TEST)
    
    else:
        print('load graphs......\n')
        graphs = load_obj(path_to_data, 'graphs')
        print('loading the graph tensors.......\n')
        ADJ_OPT = np.load(path_to_data + 'ADJ_OPT.npy')
        ADJ_TEST = np.load(path_to_data + 'ADJ_TEST.npy')
        EMB_OPT = np.load(path_to_data + 'EMB_OPT.npy')
        EMB_TEST = np.load(path_to_data + 'EMB_TEST.npy')

    print('Loading all the targets......\n')
    tgts = [0,1,2,3,]
    targets = []

    for tgt in tgts:
        with open(path_to_data + 'targets/train/target_' + str(tgt) + '.txt', 'r') as file:
            target = file.read().splitlines()
            targets.append(target)
        
    targets = np.array(targets).astype('float')
        
    return graphs, embds, ADJ_OPT, EMB_OPT, ADJ_TEST, EMB_TEST, targets

###===========================================#####################
def random_walk(graph,node,walk_length):
    walk = [node]
    for i in range(walk_length):
        neighbors = graph.neighbors(walk[i])
        walk.append(random.choice(list(neighbors)))
    return walk

def generate_walks(graph,num_walks,walk_length):
    '''
    samples num_walks walks of length walk_length+1 from each node of graph
    '''
    graph_nodes = graph.nodes()
    n_nodes = len(graph_nodes)
    walks = []
    for i in range(num_walks):
        nodes = np.random.permutation(graph_nodes)
        for j in range(n_nodes):
            walk = random_walk(graph, nodes[j], walk_length)
            walks.append(walk)
    return walks


def load_docs(BUILD_DOCS = False, pad_vec_idx = 1685894, num_walks = 5, 
              walk_length = 10, max_doc_size = 70, path_to_data=''):
    """
    Loading the documents
    
    """
    if BUILD_DOCS:

        start_time = time.time()
    
        graphs = load_obj(path_to_data, 'graphs')

        docs = []
        for idx,graph in enumerate(graphs):
            doc = generate_walks(graph,num_walks,walk_length) # create the pseudo-document representation of the graph
            docs.append(doc)
        
            if idx % round(len(graphs)/10) == 0:
                print(idx)

        print('documents generated')
    
        # truncation-padding at the document level, i.e., adding or removing entire 'sentences'
        docs = [d+[[pad_vec_idx]*(walk_length+1)]*(max_doc_size-len(d)) if len(d)<max_doc_size else d[:max_doc_size] for d in docs] 

        docs = np.array(docs).astype('int')
        print('document array shape:',docs.shape)

        np.save(path_to_data + 'documents.npy', docs, allow_pickle=False)

        print('documents saved')
        print('everything done in', round(time.time() - start_time,2))
        
    else:
        docs = np.load(path_to_data + 'documents.npy')
        
    return docs




