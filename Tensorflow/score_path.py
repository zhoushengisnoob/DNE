import numpy as np
import networkx as nx
from collections import defaultdict
from graph_walk import Graph
import pandas as pd
from datetime import datetime
import random
import math
from multiprocessing import Pool
def sigmoid(x):
    return 1 / (1 + math.exp(-x))
def score2score(score,k=1.0):
    if score != 0:
        score = k / score
    else:
        score = k
    return score


def make_graph_base(train_graph):
    G=train_graph

    for edge in G.edges():
        G[edge[0]][edge[1]]['weight'] = 1
    edges = G.edges()
    edge_dict={}
    for edge in edges:
        edge_dict[edge]=1
        if (edge[1],edge[0]) in edges:
            edge_dict[edge]=0
        else:
            edge_dict[(edge[1],edge[0])]=-1
    G = G.to_undirected()
    return G,edge_dict

def make_graph(dir):
    G = nx.read_edgelist(dir, nodetype=int,create_using=nx.DiGraph())
    for edge in G.edges():
        G[edge[0]][edge[1]]['weight'] = G.out_degree(edge[0])+G.in_degree(edge[1])
    edges = G.edges()
    edge_dict={}
    for edge in edges:
        edge_dict[edge]=G[edge[0]][edge[1]]['weight']
        if (edge[1],edge[0]) in edges:
            edge_dict[edge]=G[edge[0]][edge[1]]['weight']-G[edge[1]][edge[0]]['weight']
        else:
            edge_dict[(edge[1],edge[0])]=-G[edge[0]][edge[1]]['weight']
    G = G.to_undirected()
    return G,edge_dict


def sample_edges_base(walks,walk_scores,max_distance,walk_length,edge_dict):
    sampled_edge = []
    for i in range(len(walks)):
        walk = walks[i]
        score = walk_scores[i]
        score_list = [0 for _ in walk]
        for d in list(range(max_distance)):
            distance = d + 1
            for j in range(walk_length)[:walk_length-distance]:
                if j>=len(walk) or j+distance>=len(walk):
                    continue
                start_node = walk[j]
                end_node = walk[j+distance]
                temp_score=score_list[j]+score[j+distance-1]
                score_list[j]=temp_score
                flag = edge_dict[(start_node, walk[j + 1])]
                # if temp_score==0 and start_node==end_node:
                #     flag=(-1)**random.randint(0,1)
                sampled_edge.append([start_node,end_node,temp_score,flag,distance])
    return sampled_edge

def sample_edges(walks,walk_scores,max_distance,walk_length):
    sampled_edge = []
    for d in list(range(max_distance)):
        distance = d+1
        for i in range(len(walks)):
            walk = walks[i]
            score = walk_scores[i]
            for j in range(walk_length)[:walk_length-distance]:
                start_node = walk[j]
                end_node = walk[j+distance]
                temp_score = score2score(np.sum(score[j:j+distance]))
                sampled_edge.append([start_node,end_node,temp_score])
    return sampled_edge
def get_walk_scores(walks,edge_dict):
    walk_score = []
    for walk in walks:
        sub_score=[]
        for i in range(len(walk))[:-1]:
            sub_score.append(edge_dict[(walk[i],walk[i+1])])
        walk_score.append(sub_score)
    return walk_score

def negative_sampling(node_count,pos_shape,weight):
    neg_edges = np.random.randint(node_count, size=pos_shape)
    neg_edges[:, 2] = 0
    neg_edges[:, 3] = 0
    neg_edges[:, 4] = 0
    neg_edges[:, 5] = 0
    neg_edges[:, 6] = 1
    neg_edges=pd.DataFrame(neg_edges,columns=['sender','receiver','score','flag','weight','is_pos','distance'])
    neg_edges['weight']=weight
    return neg_edges



def get_train_data_base(train_graph,directed=False,num_walks=10,walk_length=10,target='link',use_weight=True,bias=0.5,max_distance=4):
    start_time=datetime.now()
    G, edge_dict = make_graph_base(train_graph)
    node_num = G.number_of_nodes()
    DG = Graph(G, is_directed=directed, p=1, q=1)

    base = datetime.now()
    print('pre handle graph time:',base-start_time)
    walks = DG.simulate_walks(num_walks, walk_length)

    print('sample time:',datetime.now()-base)
    walk_scores = get_walk_scores(walks, edge_dict)
    sampled_edge = sample_edges_base(walks, walk_scores, max_distance, walk_length,edge_dict)
    print('edge time:', datetime.now() - base)
    pd_sampled_edge = pd.DataFrame(sampled_edge, columns=['sender', 'receiver', 'score','flag','distance'])
    print('pandas time:', datetime.now() - base)
    print(pd_sampled_edge.score.value_counts())

    pd_sampled_edge['is_pos']=1

    # set weight
    if use_weight:
        import math
        pd_sampled_edge.loc[:, 'weight'] = np.log10((pd_sampled_edge['score'].apply(lambda x: (abs(x)+bias) )/ pd_sampled_edge[
            'distance']) + 1)
    else:
        pd_sampled_edge.loc[:, 'weight'] =1

    pd_sampled_edge.loc[pd_sampled_edge['score'] > 0, 'score'] = 1
    pd_sampled_edge.loc[pd_sampled_edge['score'] < 0, 'score'] = -1

    pd_sampled_edge.loc[pd_sampled_edge['score'] != 0, 'flag'] = 0

    negative_rate = pd_sampled_edge['weight'].mean()
    negative_sampled_edge = negative_sampling(len(G.nodes), pd_sampled_edge.shape, negative_rate)

    # processing negative edge, exchange sender and receiver
    negative_sender = pd_sampled_edge.loc[pd_sampled_edge['score'] < 0, 'sender'].copy()
    negative_score=pd_sampled_edge.loc[pd_sampled_edge['score'] < 0, 'score'].copy()
    pd_sampled_edge.loc[pd_sampled_edge['score'] < 0, 'sender'] = pd_sampled_edge.loc[
        pd_sampled_edge['score'] < 0, 'receiver']
    pd_sampled_edge.loc[pd_sampled_edge['score'] < 0, 'receiver'] = negative_sender
    pd_sampled_edge.loc[pd_sampled_edge['score'] < 0, 'score'] = -negative_score

    train_edge = pd.concat([pd_sampled_edge, negative_sampled_edge], sort=False).sample(frac=1.0)
    train_edge['score']=train_edge['is_pos']
    print(train_edge.score.value_counts())
    return node_num, train_edge









