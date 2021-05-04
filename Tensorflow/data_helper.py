import networkx as nx
import numpy as np
from sklearn.model_selection import train_test_split
import random
import pandas as pd
from sklearn.utils import shuffle


def get_label(filename):
    labels=[]
    with open(filename) as f:
        for line in f:
            line=line.strip()
            node,label=line.split()
            labels.append([int(node),label])
    return np.array(sorted(labels,key=lambda x:x[0]))

def link_cut(edge_path,percent,use_negative_test=True,use_random_test=True,ensure_reach=True):
    '''
    :param edge_path: the path of graph
    :param percent: test rate
    :param use_negative_test:
    :param use_random_test:
    :param ensure_reach: ensure two node exist path after link cut
    :return:
    '''
    origin_graph_edges=set(list(nx.read_edgelist(edge_path,create_using=nx.DiGraph(),nodetype=int).edges()))
    G=nx.read_edgelist(edge_path,create_using=nx.DiGraph(),nodetype=int)
    print("node cnt:",len(G.nodes)," edge cnt:",len(G.edges))
    origin_nodes=list(G.nodes)
    G2=nx.read_edgelist(edge_path,create_using=nx.Graph(),nodetype=int)
    n_remove_edge = round(G.number_of_edges()*percent)
    temp=0
    edges = list(G.edges())

    np.random.shuffle(edges)
    test_edges=[]
    for sampled_edge in edges:
        if sampled_edge[0]==sampled_edge[1]:
            continue
        if not G.has_edge(sampled_edge[0],sampled_edge[1]):
            continue
        G.remove_edge(sampled_edge[0],sampled_edge[1])
        if ensure_reach:
            if not G.has_edge(sampled_edge[1],sampled_edge[0]):
                G2.remove_edge(sampled_edge[0],sampled_edge[1])
            if nx.has_path(G2,sampled_edge[0],sampled_edge[1]):
                test_edges.append([sampled_edge[0],sampled_edge[1]])
                temp+=1
            else:
                G.add_edge(sampled_edge[0],sampled_edge[1])
                G2.add_edge(sampled_edge[0],sampled_edge[1])
        else:
            test_edges.append([sampled_edge[0], sampled_edge[1]])
            temp += 1


        if (sampled_edge[1],sampled_edge[0]) in origin_graph_edges:
            G.remove_edge(sampled_edge[1],sampled_edge[0])
            test_edges.append([sampled_edge[1], sampled_edge[0]])
            #temp+=1
        if temp>n_remove_edge:
            break
    if not ensure_reach:
        for node in origin_nodes:
            if G.in_degree(node)+G.out_degree(node)==0:
                G.add_edge(node, node)
    test_graph=nx.DiGraph(nodetype=int)
    pos_cnt,rev_cnt,neg_cnt=0,0,0
    for test_edge in test_edges:
        test_graph.add_edge(test_edge[0],test_edge[1],weight=1)
        pos_cnt+=1
        if use_negative_test:
            if (test_edge[1], test_edge[0]) not in origin_graph_edges:
                test_graph.add_edge(test_edge[1], test_edge[0], weight=0)
                rev_cnt+=1
    if use_random_test:
        test_samples = np.random.randint(G.number_of_nodes(), size=(len(test_edges), 2))
        for test_sample in test_samples:
            if (test_sample[0], test_sample[1]) not in origin_graph_edges:
                test_graph.add_edge(test_sample[0], test_sample[1],weight=0)
                neg_cnt+=1
    print(edge_path,f'pos num:{pos_cnt}, reverse num :{rev_cnt},neg num:{neg_cnt}')
    return G,test_graph



