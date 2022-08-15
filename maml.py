import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
import numpy as np
from Models.meta_stgcn import *
from Models.meta_gwn import *
from utils import *
from copy import deepcopy
from tqdm import tqdm
import scipy.sparse as sp

def asym_adj(adj):
    adj = adj.cpu().numpy()
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat= sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()

class STMAML(nn.Module):
    """
    MAML-based Few-shot learning architecture for STGNN
    """
    def __init__(self, data_args, task_args, model_args, model='GRU'):
        super(STMAML, self).__init__()
        self.data_args = data_args
        self.task_args = task_args
        self.model_args = model_args
        
        self.update_lr = model_args['update_lr']
        self.meta_lr = model_args['meta_lr']
        self.update_step = model_args['update_step']
        self.update_step_test = model_args['update_step_test']
        self.task_num = task_args['task_num']
        self.model_name = model

        self.loss_lambda = model_args['loss_lambda']
        print("loss_lambda = ", self.loss_lambda)

        if model == 'GRU':
            # Meta-GRU
            self.model = MetaSTNN(model_args, task_args)
            print("MAML Model: GRU")
        elif model == 'v_GRU':
            self.model = GRUModel(model_args, task_args)
            print("MAML Model: Vanilla GRU")
        elif model == 'GAT':
            # Meta-GAT
            self.model = MetaSTGAT(model_args, task_args)
            print("MAML Model: GAT")
        elif model == 'GRUGAT':
            # Meta-GAT+GRU
            self.model = MetaGATGRU(model_args, task_args)
            print("MAML Model: GRU + GAT")
        elif model == 'STGCN':
            # Meta-STGCN
            self.model = MetaSTGNN(model_args, task_args)
            print("MAML Model: STGCN")
        elif model == 'v_STGCN':
            self.model = STGCN(model_args, task_args)
            print("MAML Model: vanilla STGCN")
        elif model == 'TCN':
            # Meta-TCN
            self.model = MetaTCN(model_args, task_args)
            print("MAML Model: MetaTCN")    
        elif model == 'r_GRU':
            self.model = RandomGRU(model_args, task_args)
            print("MAML Model: Random GRU") 
        elif model == 'TCNGAT':
            self.model = MetaTCNGAT(model_args, task_args)
            print("MAML Model: MetaTCNGAT")
        elif model == 'GWN':
            self.model = MetaGWN(model_args, task_args)
            print("MAML Model: GraphWave Net")
        else:
            self.model = MetaSTNN(model_args, task_args)
            print("MAML Model: GRU (default)")
        # Meta-Graph WaveNet      
                
        # print(self.model)
        print("model params: ", count_parameters(self.model))
        self.meta_optim = optim.Adam(self.model.parameters(), lr=self.meta_lr, weight_decay=1e-2)
        # self.meta_optim = torch.optim.SGD(self.model.parameters(), lr=self.update_lr, momentum=0.9)
        self.loss_criterion = nn.MSELoss()
    


    def graph_reconstruction_loss(self, meta_graph, adj_graph):
        adj_graph = adj_graph.unsqueeze(0).float()
        for i in range(meta_graph.shape[0]):
            if i == 0:
                matrix = adj_graph
            else:
                matrix = torch.cat((matrix, adj_graph), 0)
        criteria = nn.MSELoss()
        loss = criteria(meta_graph, matrix.float())
        return loss
    
    def calculate_loss(self, out, y, meta_graph, matrix, stage='target', graph_loss=True, loss_lambda=1):
        if loss_lambda == 0:
            loss = self.loss_criterion(out, y)
        if graph_loss:
            if stage == 'source' or stage == 'target_maml':
                loss_predict = self.loss_criterion(out, y)
                loss_reconsturct = self.graph_reconstruction_loss(meta_graph, matrix)
            else:
                loss_predict = self.loss_criterion(out, y)
                loss_reconsturct = self.loss_criterion(meta_graph, matrix.float())
            loss = loss_predict + loss_lambda * loss_reconsturct
        else:
            loss = self.loss_criterion(out, y)

        return loss
    
    def meta_train_revise(self, data_spt, matrix_spt, data_qry, matrix_qry):
        
        model_loss = 0
        model_y_loss, model_g_loss = 0, 0
        init_model = deepcopy(self.model)

        for i in range(self.task_num):
            maml_model = deepcopy(init_model)
            optimizer = optim.Adam(maml_model.parameters(), lr=self.update_lr, weight_decay=1e-2)

            for k in range(self.update_step):
                batch_size, node_num, seq_len, _ = data_spt[i].x.shape
                hidden = torch.zeros(batch_size, node_num, self.model_args['hidden_dim']).cuda()

                if self.model_name == 'GWN':
                    adj_mx = [matrix_spt[i], (matrix_spt[i]).t()]
                    out, meta_graph = maml_model(data_spt[i], adj_mx)
                else:
                    out, meta_graph = maml_model(data_spt[i], matrix_spt[i])

                if self.model_name in ['v_GRU', 'r_GRU', 'v_STGCN']:
                    loss = self.loss_criterion(out, data_spt[i].y)
                else:
                    # loss = self.calculate_loss(out, data_spt[i].y, meta_graph, matrix_spt[i], 'source', graph_loss=False)
                    loss = self.calculate_loss(out, data_spt[i].y, meta_graph, matrix_spt[i], 'source', loss_lambda=self.loss_lambda)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            batch_size, node_num, seq_len, _ = data_qry[i].x.shape
            hidden = torch.zeros(batch_size, node_num, self.model_args['hidden_dim']).cuda()

            self.model = deepcopy(maml_model)

            if self.model_name == 'GWN':
                adj_mx = [matrix_qry[i], (matrix_qry[i]).t()]
                out, meta_graph = self.model(data_qry[i], adj_mx)
            else:
                out, meta_graph = self.model(data_qry[i], matrix_qry[i])

            if self.model_name in ['v_GRU', 'r_GRU', 'v_STGCN']:
                loss_q = self.loss_criterion(out, data_qry[i].y)
            else:
                # loss_q = self.calculate_loss(out, data_qry[i].y, meta_graph, matrix_qry[i], 'target_maml', graph_loss=False)
                loss_q = self.calculate_loss(out, data_qry[i].y, meta_graph, matrix_qry[i], 'target_maml', loss_lambda=self.loss_lambda)
            model_loss += loss_q

        model_loss = model_loss / self.task_num
        self.meta_optim.zero_grad()
        model_loss.backward()
        self.meta_optim.step()

        return model_loss.detach().cpu().numpy()

    def forward(self, data, matrix):
        out, meta_graph = self.model(data, matrix)
        return out, meta_graph

    def finetuning(self, target_dataloader, test_dataloader, target_epochs):
        """
        finetunning stage in MAML
        """
        maml_model = deepcopy(self.model)

        optimizer = optim.Adam(maml_model.parameters(), lr=self.meta_lr, weight_decay=1e-2)
        min_MAE = 10000000
        best_result = ''
        best_meta_graph = -1

        for epoch in tqdm(range(target_epochs)):
            train_losses = []
            start_time = time.time()
            maml_model.train()
            for step, (data, A_wave) in enumerate(target_dataloader):
                data, A_wave = data.cuda(), A_wave.cuda()
                data.node_num = data.node_num[0]

                batch_size, node_num, seq_len, _ = data.x.shape
                hidden = torch.zeros(batch_size, node_num, self.model_args['hidden_dim']).cuda()

                if self.model_name == 'GWN':
                    adj_mx = [A_wave[0].float(), (A_wave[0].float()).t()]
                    out, meta_graph = maml_model(data, adj_mx)
                else:
                    out, meta_graph = maml_model(data, A_wave[0].float())

                if self.model_name in ['v_GRU', 'r_GRU', 'v_STGCN']:
                    loss = self.loss_criterion(out, data.y)
                else:
                    # loss = self.calculate_loss(out, data.y, meta_graph, A_wave, 'test', graph_loss=False)
                    loss = self.calculate_loss(out, data.y, meta_graph, A_wave, 'test', loss_lambda=self.loss_lambda)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_losses.append(loss.detach().cpu().numpy())
            avg_train_loss = sum(train_losses)/len(train_losses)
            end_time = time.time()
            if epoch % 10 == 0:
                print("[Target Fine-tune] epoch #{}/{}: loss is {}, fine-tuning time is {}".format(epoch+1, target_epochs, avg_train_loss, end_time-start_time))

        with torch.no_grad():
            test_start = time.time()
            maml_model.eval()
            for step, (data, A_wave) in enumerate(test_dataloader):
                data, A_wave = data.cuda(), A_wave.cuda()
                data.node_num = data.node_num[0]
                batch_size, node_num, seq_len, _ = data.x.shape
                hidden = torch.zeros(batch_size, node_num, self.model_args['hidden_dim']).cuda()

                if self.model_name == 'GWN':
                    adj_mx = [A_wave[0].float(), (A_wave[0].float()).t()]
                    out, meta_graph = maml_model(data, adj_mx)
                else:
                    out, meta_graph = maml_model(data, A_wave[0].float())

                if step == 0:
                    outputs = out
                    y_label = data.y
                else:
                    outputs = torch.cat((outputs, out))
                    y_label = torch.cat((y_label, data.y))
            outputs = outputs.permute(0, 2, 1).detach().cpu().numpy()
            y_label = y_label.permute(0, 2, 1).detach().cpu().numpy()
            result = metric_func(pred=outputs, y=y_label, times=self.task_args['pred_num'])
            test_end = time.time()

            result_print(result, info_name='Evaluate')
            print("[Target Test] testing time is {}".format(test_end-test_start))
            # if np.sum(result['MAE']) < min_MAE:
            #     best_result = result
            #     best_epoch = epoch
            #     min_MAE = np.sum(result['MAE'])
            #     best_meta_graph = meta_graph
        
        # print("Best epoch is @{}".format(best_epoch))
        # result_print(best_result, info_name='Best')
        # np.save("result/best_meta_graph_shenzhen_0124.npy", best_meta_graph.detach().cpu().numpy())

        
