import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import sys
sys.path.append('..')
import logging
from argparse import ArgumentDefaultsHelpFormatter,ArgumentParser
from keras.layers import Activation, Dense, Input, Subtract, Concatenate, Dropout,noise,Dot,Lambda,Average,Embedding,LSTM,Flatten,BatchNormalization
from keras.layers.advanced_activations import LeakyReLU,PReLU,ThresholdedReLU,ELU,ReLU
from keras.models import Model
from keras import backend as K
import keras.layers
from keras.utils.vis_utils import plot_model
import networkx as nx
import numpy as np
from evaluate import *
from data_helper import *

from keras import backend as K
import keras.backend.tensorflow_backend as KTF
from keras.optimizers import Adam,Nadam,TFOptimizer,Adamax,RMSprop,SGD,Adagrad,Adadelta


import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
session = tf.Session(config=config)
KTF.set_session(session )
from datetime import datetime
from score_path import get_train_data_base
logging.basicConfig(filename=f"dne.log", filemode="a",
                    format="%(asctime)s %(name)s:%(levelname)s:%(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

def build_model(node_num,sender_receiver_dim=128,lr=0.0001):

    sender_input=Input(shape=(1,))
    receiver_input=Input(shape=(1,))

    node_input=Input(shape=(1,))
    sender_layer=Embedding(node_num,sender_receiver_dim,input_length=1)(node_input)
    sender_layer=Flatten()(sender_layer)
    sender_model=Model(node_input,sender_layer)

    receiver_model_input=Input(shape=(1,))
    receiver_embedding=Embedding(node_num,sender_receiver_dim,input_length=1)(receiver_model_input)
    receiver_embedding=Flatten()(receiver_embedding)
    receiver_model=Model(receiver_model_input,receiver_embedding)

    sender_sender=sender_model(sender_input)
    receiver_receiver=receiver_model(receiver_input)



    sender_and_receiver_similarity=Dot(1,name='link',normalize=True)([sender_sender,receiver_receiver])

    result=sender_and_receiver_similarity
    result=Activation(activation='sigmoid')(result)

    final_model=Model([sender_input,receiver_input],[result])
    final_model.summary()

    final_model.compile(optimizer=Adam(lr=lr),loss='binary_crossentropy',metrics=['acc'])

    return final_model,sender_model,receiver_model

if __name__=='__main__':
    parser=ArgumentParser(description='Run DNE')
    parser.add_argument('--dataset',type=str,default='cora')
    parser.add_argument('--task',type=str,default='link',choices=['link','rec','cls'])
    parser.add_argument('--percent',type=int,default=30,help='percent of test edges for link prediction')
    parser.add_argument('--data_dir',type=str,default='./data/')
    parser.add_argument('--use_negative_test',action='store_true',default=False,help='add negative reverse edge to test dataset')
    parser.add_argument('--ensure_reach',action='store_true',default=False,help='make sure two points are accessible if they are accessible before split dataset ')
    parser.add_argument('--use_weight',action='store_true',default=False)
    parser.add_argument('--bias',type=float,default=0.5,help='')
    parser.add_argument('--lr', type=float, default=0.0001, help='')
    parser.add_argument('--epoch', type=int, default=10, help='')
    parser.add_argument('--batch_size', type=int, default=2048*5, help='')
    parser.add_argument('--seed',type=int,default=0,help='')
    parser.add_argument('--dim', type=int, default=128, help='emb dim')

    parser.set_defaults(dataset='pubmed',
                        task='link',
                        percent=30,
                        use_negative_test=False,
                        use_weight=True,
                        ensure_reach=False,
                        seed=5118,
                        bias=1,
                        max_distance=4
                        )
    args=parser.parse_args()


    # set parameter
    parameter=f'{args.dataset}|{args.percent}|{args.use_negative_test}|{args.ensure_reach}|{args.use_weight}|{args.seed}|{args.bias}|'
    print(parameter)
    start_time=datetime.now()
    print('run time %s'% start_time)
    data_name=args.dataset
    print(data_name,args.task)
    top_k_list=[10,20,30]  # for recommendation estimate
    percent=args.percent
    base_dir=f'{args.data_dir}{data_name}/'
    edge_path=base_dir+'edge.txt'
    label_dir=base_dir+'label.txt'
    use_negative_test=args.use_negative_test
    ensure_reach=args.ensure_reach

    use_weight=args.use_weight

    if args.seed!=0:
        print(f'set seed:{args.seed}')
        np.random.seed(args.seed)

    target=args.task

    # get train data and test data
    start_sample_time=datetime.now()
    if target=='link' or target=='rec':
        print('sample train graph for link prediction')
        train_graph,test_graph = link_cut(edge_path, percent/100, use_negative_test,True,ensure_reach)
        node_num, train_edge = get_train_data_base(train_graph,target=target,use_weight=use_weight,bias=args.bias,max_distance=args.max_distance)
    else:
        print('sample full graph for classifier')

        train_graph=nx.read_edgelist(edge_path, nodetype=int,create_using=nx.DiGraph())

        node_num, train_edge = get_train_data_base(train_graph,target=target,use_weight=use_weight,bias=args.bias,max_distance=args.max_distance)

    print('sample time',datetime.now()-start_sample_time)

    #build model and train
    score_model, sender_model, receiver_model = build_model(node_num,args.dim,args.lr)
    score_model.fit([train_edge['sender'].values, train_edge['receiver'].values],
                    [train_edge['score'].values],
                    batch_size=args.batch_size,
                    epochs=args.epoch,
                    shuffle=True,
                    sample_weight=[train_edge['weight'].values],
                    verbose=2
                    )

    #get node embedding
    sender_embedding = sender_model.predict(np.array(range(node_num)))
    receiver_embedding = receiver_model.predict(np.array(range(node_num)))


    #eval
    if target=='link':
        log_str=parameter
        test_result = evaluate_double_embedding_link_prediction(
            sender_embedding,
            receiver_embedding,
            test_graph
            )
        log_str+=f'{test_result[0]}|{test_result[1]}'#auc,ap
        logging.info(log_str)
    if target=='rec':
        test_result = evaluate_double_embedding_link_prediction(
            sender_embedding,
            receiver_embedding,
            test_graph
        )
        print( f'auc:{test_result[0]} ap:{test_result[1]}' ) # auc,ap
        del train_edge
        log_str = parameter
        matrix = sender_embedding.dot(receiver_embedding.T)#.astype(np.float32)
        print('start sort node', datetime.now())
        sort_edge=get_sort_edge(matrix,train_graph,test_graph,max(top_k_list))
        print('end sort node', datetime.now())
        for top_k in top_k_list:
            item_rec = evaluate_top_k(train_graph, test_graph, matrix, top_k,sort_edge)
            precision, recall, f1, m_ap, mrr, mndcg = item_rec
            log_str += f'{top_k}|{round(precision, 4)}|{round(recall, 4)}|' \
                       f'{round(f1, 4)}|{round(m_ap, 4)}|{round(mrr, 4)}|{round(mndcg, 4)}|'
        logging.info(log_str)
        print(log_str)
    elif target=='cls':
        eval_embedding(sender_embedding, get_label(label_dir)[:, 1])
        eval_embedding(receiver_embedding, get_label(label_dir)[:, 1])
        eval_embedding(np.concatenate([sender_embedding,receiver_embedding],axis=1), get_label(label_dir)[:, 1])
        eval_embedding(sender_embedding+receiver_embedding, get_label(label_dir)[:, 1])

    print('run finish time %s' % datetime.now())
    print('total time %s' % (datetime.now()-start_time))

