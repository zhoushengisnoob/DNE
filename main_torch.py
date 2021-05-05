import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import sys
sys.path.append('..')
import logging
from argparse import ArgumentDefaultsHelpFormatter,ArgumentParser

import networkx as nx
import numpy as np
from evaluate import *
from data_helper import *
from sklearn.utils import shuffle

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
torch.backends.cudnn.benchmark = True
from datetime import datetime
from score_path import get_train_data_base
logging.basicConfig(filename=f"dne.log", filemode="a",
                    format="%(asctime)s %(name)s:%(levelname)s:%(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
def set_seed(seed, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed(seed)


class DNE(nn.Module):
    def __init__(self, node_num,dim=128,device='cuda'):
        super(DNE, self).__init__()
        self.device=device
        self.sender=torch.nn.Embedding(node_num,dim).to(self.device)
        self.receiver = torch.nn.Embedding(node_num, dim).to(self.device)

        self.add_module(f'sender_emb', self.sender)
        self.add_module(f'receiver_emb', self.receiver)


    def forward(self,sender_idx,receiver_idx):
        sender_emb=self.sender(sender_idx)
        sender_emb=sender_emb/torch.norm(sender_emb,dim=1).unsqueeze(1)
        receiver_emb=self.receiver(receiver_idx)
        receiver_emb=receiver_emb/torch.norm(receiver_emb,dim=1).unsqueeze(1)

        output=torch.sum(sender_emb*receiver_emb,dim=1)
        return output

def score(criterion, output, label):
    return criterion(output, label)


if __name__=='__main__':
    parser=ArgumentParser(description='Run DNE')
    parser.add_argument('--dataset',type=str,default='cora')
    parser.add_argument('--task',type=str,default='link',choices=['link','rec','cls'])
    parser.add_argument('--percent',type=int,default=30,help='percent of test edges for link prediction')
    parser.add_argument('--data_dir',type=str,default='./data/')
    parser.add_argument('--use_negative_test',action='store_true',default=False,help='add negative reverse edge to test dataset')
    parser.add_argument('--ensure_reach',action='store_true',default=True,help='make sure two points are accessible if they are accessible before split dataset ')
    parser.add_argument('--use_weight',action='store_true',default=False)
    parser.add_argument('--bias',type=float,default=0.5,help='')
    parser.add_argument('--lr', type=float, default=0.01, help='')
    parser.add_argument('--epoch', type=int, default=10, help='')
    parser.add_argument('--batch_size', type=int, default=2048*5, help='')
    parser.add_argument('--seed',type=int,default=0,help='')
    parser.add_argument('--dim', type=int, default=128, help='emb dim')
    parser.add_argument('--device', type=int, default=0, help='device')
    parser.add_argument('--max_distance', type=int, default=4, help='max_distance')
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
        set_seed(args.seed,'cuda')

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

    train_sender_idx=torch.LongTensor(train_edge['sender'].values).cuda()
    train_receiver_idx=torch.LongTensor(train_edge['receiver'].values).cuda()
    label=torch.FloatTensor(train_edge['score'].values).cuda()
    weight=torch.FloatTensor(train_edge['weight'].values).cuda()

    #build model and train
    model = DNE(node_num, args.dim,args.device)
    model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)


    for epoch in range(args.epoch):
        losses = []
        steps = (train_edge.shape[0] // args.batch_size) + (0 if train_edge.shape[0] % args.batch_size == 0 else 1)
        epoch_idx=np.array(range(train_edge.shape[0]))
        epoch_idx=torch.LongTensor(shuffle(epoch_idx)).cuda()
        for step in tqdm(range(steps)):
            start = step * args.batch_size
            end = min((step + 1) * args.batch_size, train_edge.shape[0])
            if end-start<=1:
                continue
            step_train_sender_idx=train_sender_idx[epoch_idx[start:end]]
            step_train_receiver_idx=train_receiver_idx[epoch_idx[start:end]]
            step_train_label=label[epoch_idx[start:end]]
            step_train_weight=weight[epoch_idx[start:end]]

            output=model(step_train_sender_idx,step_train_receiver_idx)
            criterion = torch.nn.BCEWithLogitsLoss(step_train_weight)
            step_loss=score(criterion,output,step_train_label)
            step_loss.backward()
            optimizer.step()
            optimizer.zero_grad()



    #get node embedding
    sender_embedding = model.sender.weight.cpu().detach().numpy()
    receiver_embedding = model.receiver.weight.cpu().detach().numpy()

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

