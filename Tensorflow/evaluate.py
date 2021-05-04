
import networkx as nx
import numpy as np
from sklearn.metrics import f1_score,roc_auc_score,accuracy_score,precision_score,recall_score,average_precision_score
from data_helper import *
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score,f1_score
import numpy as np
import pandas as pd
import math
from multiprocessing import Pool
from datetime import datetime
import heapq
def eval_embedding(X,Y,train_percent=0.3):
    #print(X.shape,Y.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=train_percent, test_size=1 - train_percent,random_state=6
                                                        )
    clf=LinearSVC(max_iter=10000)
    #print('start fit')
    clf.fit(X_train, y_train)
    #print('end fit')
    res = clf.predict(X_test)
    micro = f1_score(y_test, res, average='micro')
    macro = f1_score(y_test, res, average='macro')
    print(macro,micro)

def find_topk(pairs,max_k):
    result=[]
    for pair in pairs:
        if len(result)<max_k:
            heapq.heappush(result,pair)
            continue
        if result[0][0]<pair[0]:
            heapq.heappop(result)
            heapq.heappush(result,pair)
    result.sort(reverse=True)
    return [pair[1] for pair in result]

def multi_sort(data):
    node,max_k,line=data
    pairs= [(line[i], i) for i in range(len(line))]
    return find_topk(pairs,max_k)

def get_sort_edge(matrix,train_graph:nx.DiGraph,test_graph:nx.DiGraph,max_k):
    min_value=np.min(matrix)-1
    for train_edge in list(train_graph.edges()):
        #matrix[train_edge[0],train_edge[1]]=min_value #保证训练集中出现过的边为最小值
        matrix[train_edge[0]][train_edge[1]] = min_value
    for node in train_graph.nodes:
        matrix[node][node]=min_value
    test_map = dict(zip(train_graph.nodes, [[] for _ in range(train_graph.number_of_nodes())]))
    for edge, weight in test_graph.edges().items():
        if int(weight['weight']) == 0:
            continue
        sender, receiver = edge
        test_map[sender].append(receiver)
    result = []

    print('start multi',datetime.now())
    data=[(i,max_k,matrix[i]) for i in range(len(matrix))]
    print('end finish prepare data', datetime.now())
    pool=Pool(50)
    print('start map', datetime.now())
    result = pool.map(multi_sort,data )
    pool.close()
    pool.join()

    return np.array(list(result))

def evaluate_top_k(train_graph:nx.DiGraph,test_graph:nx.DiGraph,matrix,top_k,sort_edge):
    '''
    :param train_graph:
    :param test_graph:
    :param matrix: similarity matrix
    :param top_k:
    :param sort_edge:
    :return:
    '''
    def IDCG(n):
        idcg = 0
        for i in range(n):
            idcg += 1 / math.log(i + 2, 2)
        return idcg
    def get_topk_edge():
        result=[]
        for node in range(len(matrix)):
            result.append([ pair for pair in sort_edge[node]][:top_k])
        return np.array(result)

    top_k_edge=get_topk_edge()
    test_map=dict(zip(train_graph.nodes,[[] for _ in range(train_graph.number_of_nodes())]))
    for edge, weight in test_graph.edges().items():
        if int(weight['weight'])==0:
            continue
        sender,receiver=edge
        test_map[sender].append(receiver)

    precision_list = []
    recall_list = []
    ap_list = []
    ndcg_list = []
    rr_list = []

    for sender in train_graph.nodes:
        if len(test_map[sender])==0:
            continue
        rank_list=list(top_k_edge[sender])#[::-1]
        ground_list=set(test_map[sender])
        hits, sum_precs = 0, 0
        rr = None
        dcg = 0
        idcg = IDCG(len(ground_list))

        for idx,receiver in enumerate(rank_list):
            if receiver in ground_list:
                #hit
                hits+=1
                #ap
                sum_precs+=hits/(idx+1)
                #rr
                if rr is None:
                    rr=1/(idx+1)
                #ndcg
                rank=idx+1
                dcg+=1/math.log(rank+1,2)
        pre=hits/len(rank_list)
        rec=hits/len(ground_list)
        ap=sum_precs/len(ground_list)
        rr=rr if rr is not None else 0
        nDCG=dcg/idcg
        precision_list.append(pre)
        recall_list.append(rec)
        ap_list.append(ap)
        rr_list.append(rr)
        ndcg_list.append(nDCG)

    precision=np.mean(precision_list)
    recall=np.mean(recall_list)
    m_ap=np.mean(ap_list)
    mrr=np.mean(rr_list)
    mndcg=np.mean(ndcg_list)
    f1=2*precision*recall/(precision+recall) if precision+recall>0 else 0
    print(f'top_k:{top_k},precision:{round(precision,4)},recall:{round(recall,4)},f1:{round(f1,4)},map:{round(m_ap,4)},mrr:{round(mrr,4)},mndcg:{round(mndcg,4)}')
    return precision,recall,f1,m_ap,mrr,mndcg




def evaluate_double_embedding_link_prediction(sender_embedding,receiver_embedding,test_graph):
    y_true=[]
    y_pred=[]
    for edge, weight in test_graph.edges().items():
        y_true.append(int(weight['weight']))
        pred_prob = np.dot(sender_embedding[edge[0]], receiver_embedding[edge[1]])
        y_pred.append(pred_prob)
    auc=roc_auc_score(y_true, y_pred)
    ap=average_precision_score(y_true,y_pred)
    print(f'link prediction auc:{auc} ,ap:{ap}',)
    return auc,ap
