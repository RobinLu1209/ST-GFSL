import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn import GATConv
from torch_geometric.nn.conv import MessagePassing
from torch import Tensor
from typing import Union, Tuple, Optional
from torch_geometric.typing import PairTensor, Adj, OptTensor, Size
from torch_geometric.typing import (OptPairTensor, Adj, Size, NoneType,
                                    OptTensor)
import time
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from torch.nn.utils import weight_norm

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class TCN(nn.Module):
    def __init__(self, model_args, task_args):
        super(TCN, self).__init__()
        self.model_args = model_args
        self.task_args = task_args
        self.message_dim = model_args['message_dim']
        self.hidden_dim = model_args['hidden_dim']
        self.output_dim = model_args['output_dim']
        self.build()

    def build(self):
        self.tcn_1 = TemporalConvNet(num_inputs=self.message_dim, num_channels=[self.hidden_dim, self.output_dim], kernel_size=3)
        self.tcn_2 = TemporalConvNet(num_inputs=self.message_dim, num_channels=[self.hidden_dim, self.output_dim], kernel_size=2)
        self.tcn_3 = TemporalConvNet(num_inputs=self.message_dim, num_channels=[self.hidden_dim, self.output_dim], kernel_size=4)
        self.predictor = nn.Linear(self.task_args['his_num'], self.task_args['pred_num'])

    def forward(self, data, adj):
        batch_size, node_num, seq_len, c_in = data.x.shape
        input = torch.reshape(data.x, (batch_size*node_num, seq_len, c_in)).permute(0,2,1)
        
        output_1 = self.tcn_1(input).squeeze(1)
        output_2 = self.tcn_2(input).squeeze(1)
        output_3 = self.tcn_3(input).squeeze(1)
        output = output_1 + output_2 + output_3

        output = torch.reshape(output, (batch_size, node_num, seq_len))
        output = self.predictor(output)
        return output, adj

class HyperNetwork(nn.Module):
    def __init__(self, f_size = 3, z_dim = 64, out_size=16, in_size=16):
        super(HyperNetwork, self).__init__()
        self.z_dim = z_dim
        self.f_size = f_size
        self.out_size = out_size
        self.in_size = in_size

        self.w1 = Parameter(torch.fmod(torch.randn((self.z_dim, self.out_size*self.f_size*self.f_size)),2))
        self.b1 = Parameter(torch.fmod(torch.randn((self.out_size*self.f_size*self.f_size)),2))

        self.w2 = Parameter(torch.fmod(torch.randn((self.z_dim, self.in_size*self.z_dim)),2))
        self.b2 = Parameter(torch.fmod(torch.randn((self.in_size*self.z_dim)),2))
        
    def forward(self, z):

        h_in = torch.matmul(z, self.w2) + self.b2
        h_in = h_in.view(self.in_size, self.z_dim)

        h_final = torch.matmul(h_in, self.w1) + self.b1
        kernel = h_final.view(self.out_size, self.in_size, self.f_size, self.f_size)

        return kernel

class HyperNetwork_old(nn.Module):
    def __init__(self, kernel_size = 3, meta_dim = 64, out_channels=16, in_channels=16):
        super(HyperNetwork_old, self).__init__()
        self.meta_dim = meta_dim
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.w1 = Parameter(torch.fmod(torch.randn((self.meta_dim, self.out_channels*1*self.kernel_size)).cuda(),2))
        self.b1 = Parameter(torch.fmod(torch.randn((self.out_channels*1*self.kernel_size)).cuda(),2))
        self.w2 = Parameter(torch.fmod(torch.randn((self.meta_dim, self.in_channels*self.meta_dim)).cuda(),2))
        self.b2 = Parameter(torch.fmod(torch.randn((self.in_channels*self.meta_dim)).cuda(),2))

    def forward(self, z):
        """
        : param z shape is [meta_dim]
        : return kernel shape is [out_channels, in_channels, 1, kernel_size]
        """
        h_in = torch.matmul(z, self.w2) + self.b2
        h_in = h_in.view(self.in_channels, self.meta_dim)
        h_final = torch.matmul(h_in, self.w1) + self.b1
        kernel = h_final.view(self.out_channels, self.in_channels, 1, self.kernel_size)
        return kernel

class GNNConv(MessagePassing):
    def __init__(self, channels: Union[int, Tuple[int, int]], dim: int = 0,
                 aggr: str = 'add', batch_norm: bool = False,
                 bias: bool = True, **kwargs):
        super(GNNConv, self).__init__(aggr=aggr, **kwargs)
        self.channels = channels
        self.dim = dim
        self.batch_norm = batch_norm

        self.linear_neighbor = nn.Linear(channels[0] * 2 + dim, channels[1], bias=bias)
        self.linear_self = nn.Linear(channels[0], channels[1], bias=bias)
        self.bn = nn.BatchNorm1d(channels[1])

        self.reset_parameters()

    def reset_parameters(self):
        self.linear_neighbor.reset_parameters()
        self.linear_self.reset_parameters()
        self.bn.reset_parameters()

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, size: Size = None) -> Tensor:
        if isinstance(x, Tensor):
            x: PairTensor = (x, x)

        # propagate_type: (x: PairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)
        out = self.bn(out) if self.batch_norm else out
        out += self.linear_self(x[1])
        return out

    def message(self, x_i, x_j, edge_attr: OptTensor) -> Tensor:
        if edge_attr is None:
            print("x_i:{}, x_j:{}".format(x_i.shape, x_j.shape))
            z = torch.cat([x_i, x_j], dim=-1)
        else:
            z = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.linear_neighbor(z)

    def __repr__(self):
        return '{}({}, dim={})'.format(self.__class__.__name__, self.channels,
                                       self.dim)

class STMetaLearner_add(nn.Module):
    def __init__(self, model_args, task_args):
        super(STMetaLearner_add, self).__init__()
        self.model_args = model_args
        self.task_args = task_args
        self.tp = model_args['tp']
        self.sp = model_args['sp']
        self.node_feature_dim = model_args['node_feature_dim']
        self.edge_feature_dim = model_args['edge_feature_dim']
        self.message_dim = model_args['message_dim']
        self.hidden_dim = model_args['hidden_dim']
        self.his_num = task_args['his_num']
        self.pred_num = task_args['pred_num']
        self.meta_in = self.node_feature_dim + self.message_dim * self.his_num
        self.meta_out = model_args['meta_dim']
        self.build()
    
    def build(self):
        if self.tp:
            print("tp is True.")
            self.tp_learner = nn.GRU(self.message_dim, 1, batch_first=True)     
        if self.sp:
            print("sp is True.")
            self.sp_learner = GATConv(self.message_dim * self.his_num, self.his_num, 3, False, dropout=0.1)

        if self.tp and self.sp:
            self.alpha = nn.Parameter(torch.FloatTensor(self.his_num))
            stdv = 1. / math.sqrt(self.alpha.shape[0])
            self.alpha.data.uniform_(-stdv, stdv)
        
        if self.tp == False and self.sp == False:
            print("sp and tp are all False.")
            self.meta_knowledge = nn.Parameter(torch.FloatTensor(self.his_num))
        
        self.mk_learner = nn.Linear(self.his_num, self.meta_out)

    def forward(self, data):
        batch_size, node_num, his_len, message_dim = data.x.shape
        # print("node_feature: {}, message_feature: {}, edge_attr: {}".format(node_feature.shape, message_feature.shape, edge_attr.shape))

        if self.tp:
            # tp_learner -> [batch_size * node_num, his_len]
            self.tp_learner.flatten_parameters() 
            tp_input = torch.reshape(data.x, (batch_size * node_num, his_len, message_dim))
            tp_output, _ = self.tp_learner(tp_input)
            tp_output = tp_output.squeeze(-1)

        if self.sp:
            # sp_learner -> [batch_size * node_num, his_len]
            sp_input = torch.reshape(data.x, (batch_size * node_num, his_len, message_dim))
            sp_input = torch.reshape(sp_input, (batch_size * node_num, his_len * message_dim))
            sp_output = self.sp_learner(sp_input, data.edge_index)

        if self.tp and self.sp:
            mk_input = torch.sigmoid(self.alpha) * sp_output + (1-torch.sigmoid(self.alpha)) * tp_output
        elif self.tp:
            mk_input = tp_output
        elif self.sp:
            mk_input = sp_output
        else:
            mk_input = self.meta_knowledge
            # print("sp and tp are all False.")
        
        meta_knowledge = self.mk_learner(mk_input)
        meta_knowledge = torch.reshape(meta_knowledge, (batch_size, node_num, self.meta_out))
        return meta_knowledge

class STMetaLearner(nn.Module):
    def __init__(self, model_args, task_args):
        super(STMetaLearner, self).__init__()
        self.model_args = model_args
        self.task_args = task_args
        self.tp = model_args['tp']
        self.sp = model_args['sp']
        self.node_feature_dim = model_args['node_feature_dim']
        self.edge_feature_dim = model_args['edge_feature_dim']
        self.message_dim = model_args['message_dim']
        self.hidden_dim = model_args['hidden_dim']
        self.his_num = task_args['his_num']
        self.pred_num = task_args['pred_num']
        self.meta_in = self.node_feature_dim + self.message_dim * self.his_num
        self.meta_out = model_args['meta_dim']
        self.build()
    
    def build(self):
        if self.tp:
            print("tp is True.")
            self.tp_learner = nn.GRU(self.message_dim, 1, batch_first=True)     
        if self.sp:
            print("sp is True.")
            self.sp_learner = GATConv(self.message_dim * self.his_num, self.his_num, 3, False, dropout=0.1)

        if self.tp and self.sp:
            self.mk_learner = nn.Linear(2*self.his_num, self.meta_out)
        else:
            self.mk_learner = nn.Linear(self.his_num, self.meta_out)

        if self.tp == False and self.sp == False:
            print("sp and tp are all False.")
            self.meta_knowledge = nn.Parameter(torch.FloatTensor(self.his_num))
    
    def forward(self, data):
        """
        : param node_feature of shape [batch_size, node_num, node_dim]
        : param edge_attr of shape [batch_size * edge_num, edge_dim], torch_geometric based
        : param message_feature of shape [batch_size, node_num, his_len, message_dim] -> [batch_size * node_num, his_len * message_dim]
        : param edge_index: torch_geometric based
        : output meta_knowledge shape [batch_size, node_num, meta_dim]
        """
        
        batch_size, node_num, his_len, message_dim = data.x.shape
        # print("node_feature: {}, message_feature: {}, edge_attr: {}".format(node_feature.shape, message_feature.shape, edge_attr.shape))

        if self.tp:
            # tp_learner -> [batch_size * node_num, his_len]
            self.tp_learner.flatten_parameters() 
            tp_input = torch.reshape(data.x, (batch_size * node_num, his_len, message_dim))
            tp_output, _ = self.tp_learner(tp_input)
            tp_output = tp_output.squeeze(-1)

        if self.sp:
            # sp_learner -> [batch_size * node_num, his_len]
            sp_input = torch.reshape(data.x, (batch_size * node_num, his_len, message_dim))
            sp_input = torch.reshape(sp_input, (batch_size * node_num, his_len * message_dim))
            sp_output = self.sp_learner(sp_input, data.edge_index)

        if self.tp and self.sp:
            mk_input = torch.cat((tp_output, sp_output), -1)
        elif self.tp:
            mk_input = tp_output
        elif self.sp:
            mk_input = sp_output
        else:
            mk_input = self.meta_knowledge
            # print("sp and tp are all False.")
        
        meta_knowledge = self.mk_learner(mk_input)
        meta_knowledge = torch.reshape(meta_knowledge, (batch_size, node_num, self.meta_out))
        return meta_knowledge

        # batch_size, node_num, his_len, message_dim = message_feature.shape
        # message_feature = torch.reshape(message_feature, (-1, node_num, his_len*message_dim))
        # message_feature = torch.reshape(message_feature, (-1, his_len*message_dim))
        # message_feature = self.sp_learner(message_feature, edge_index)
        # # print("[Graph learner] message feature shape is", message_feature.shape)
        # meta_knowledge = torch.reshape(message_feature, (-1, node_num, self.model_args['meta_dim']))
        # return meta_knowledge

class STMetaEdgeLearner(nn.Module):
    def __init__(self, model_args, task_args):
        super(STMetaEdgeLearner, self).__init__()
        self.model_args = model_args
        self.task_args = task_args
    
    def build(self):
        pass

    def forward(self, edge_index, edge_feature):
        pass

class STMetaLearner_old(nn.Module):
    def __init__(self, model_args, task_args):
        super(STMetaLearner_old, self).__init__()
        self.model_args = model_args
        self.task_args = task_args
        self.node_feature_dim = model_args['node_feature_dim']
        self.edge_feature_dim = model_args['edge_feature_dim']
        self.message_dim = model_args['message_dim']
        self.his_num = task_args['his_num']
        self.pred_num = task_args['pred_num']
        self.meta_in = self.node_feature_dim + self.message_dim * self.his_num
        self.meta_out = model_args['meta_dim']
        self.build()
    
    def build(self):
        self.node_learner = nn.Linear(self.node_feature_dim, self.node_feature_dim)
        self.edge_learner = nn.Linear(self.edge_feature_dim, self.edge_feature_dim)
        self.graph_learner = GATConv(self.message_dim * self.his_num, self.message_dim * self.his_num, 1, False, dropout=0.1)
        self.meta_learner_1 = GNNConv([self.meta_in, self.meta_out], dim=self.model_args['edge_feature_dim'])
        # self.meta_learner_2 = GNNConv([self.meta_out, self.meta_out])
    
    def forward(self, data):
        """
        : param node_feature of shape [batch_size, node_num, node_dim]
        : param edge_attr of shape [batch_size * edge_num, edge_dim], torch_geometric based
        : param message_feature of shape [batch_size, node_num, his_len, message_dim] -> [batch_size * node_num, his_len * message_dim]
        : param edge_index: torch_geometric based
        : output meta_knowledge shape [batch_size, node_num, meta_dim]
        """
        node_feature, edge_attr = data.node_feature, data.edge_attr
        message_feature, edge_index = data.x, data.edge_index
        # print("node_feature: {}, message_feature: {}, edge_attr: {}".format(node_feature.shape, message_feature.shape, edge_attr.shape))
        # Node learner
        node_feature = self.node_learner(node_feature)
        node_feature = torch.reshape(node_feature, (-1, self.node_feature_dim))
        # print("[Node learner] node feature shape is", node_feature.shape)
        # Edge learner
        edge_attr = self.edge_learner(edge_attr)
        # Graph learner
        batch_size, node_num, his_len, message_dim = message_feature.shape
        message_feature = torch.reshape(message_feature, (-1, node_num, his_len*message_dim))
        message_feature = torch.reshape(message_feature, (-1, his_len*message_dim))
        message_feature = self.graph_learner(message_feature, edge_index)
        # print("[Graph learner] message feature shape is", message_feature.shape)
        # Meta Learner
        meta_input_feature = torch.cat((message_feature, node_feature), dim=-1)
        # print("meta_input shape is {}".format(meta_input_feature.shape))
        meta_output_feature = self.meta_learner_1(meta_input_feature, edge_index, edge_attr)
        # print("meta_output_1 shape is", meta_output_feature.shape)
        # meta_knowledge = self.meta_learner_2(torch.sigmoid(meta_output_feature), edge_index)
        meta_knowledge = torch.reshape(meta_output_feature, (-1, node_num, self.model_args['meta_dim']))
        # print("meta_knowledge shape is", meta_knowledge.shape)
        return meta_knowledge 

class MetaLinear(nn.Module):
    def __init__(self, meta_dim, in_feature_dim, out_feature_dim, bias=True):
        super(MetaLinear, self).__init__()
        self.meta_dim = meta_dim
        self.in_feature_dim = in_feature_dim
        self.out_feature_dim = out_feature_dim
        self.bias = bias
        self.build()
    
    def build(self):
        self.w_linear = nn.Linear(self.meta_dim, self.in_feature_dim * self.out_feature_dim)
        if self.bias:
            self.b_linear = nn.Linear(self.meta_dim, self.out_feature_dim)

    def forward(self, meta_knowledge, input, dim=3, nonlinear='None'):
        # meta_knowledge shape is [batch_size, node_num, meta_dim]
        # input shape is [batch_size, node_num, in_feature_dim]
        # output shape is [batch_size, node_num, out_feature_dim]

        if dim == 3:
            batch_size, node_num, _ = input.shape
            meta_knowledge = torch.reshape(meta_knowledge, (-1, self.meta_dim))
            x = torch.reshape(input, (-1, self.in_feature_dim)).unsqueeze(1)
        elif dim == 2:
            x = input.unsqueeze(1)
            # print("[Meta-Linear] meta knowledge shape is {}, x shape is {}".format(meta_knowledge.shape, x.shape))
        else:
            raise BBDefinedError("dim error.")

        w = self.w_linear(meta_knowledge)
        w = torch.reshape(w, (-1, self.in_feature_dim, self.out_feature_dim))

        if self.bias:
            b = self.b_linear(meta_knowledge)
            b = torch.reshape(b, (-1, 1, self.out_feature_dim))
            # print("[Meta Linear] w shape is {}, b shape is {}".format(w.shape, b.shape))
            y = torch.bmm(x, w) + b
        else:
            y = torch.bmm(x, w)
        
        if dim == 3:
            y = torch.reshape(y.squeeze(1), (batch_size, node_num, self.out_feature_dim))

        if nonlinear == 'None':
            output = y
        elif nonlinear == 'relu':
            output = nn.ReLU(y)
        elif nonlinear == 'leaky':
            output = F.LeakyReLU(y)
        elif nonlinear == 'tanh':
            output = nn.Tanh(y)
        else:
            print("[Warning] Unsupported nonlinear function")
            output = y
        return output

class MetaConv1d(nn.Module):
    def __init__(self, meta_dim, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
        super(MetaConv1d, self).__init__()
        self.meta_dim = meta_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.build()

    def build(self):
        self.d = self.meta_dim
        self.w1_linear = nn.Linear(self.meta_dim, self.in_channels*self.d)
        self.w2_linear = nn.Linear(self.d, self.out_channels*self.kernel_size)
        self.b_linear = nn.Linear(self.meta_dim, self.out_channels)

    def forward(self, meta_knowledge, input):
        # [B, N, d_mk] -> [B*N, d_mk] -> [B*N, C_in*d] -> [B*N, C_in, d] -> [B*N, C_in, C_out*kernel_size] -> [B*N, C_out, C_in, kernel_size] -> [B*N*C_out, C_in, kernel_size]
        # print("meta_knowledge device: ", meta_knowledge.device, "input device:", input.device)
        # print("self.w1_linear device:", next(self.w1_linear.parameters()).device)
        w_meta = self.w1_linear(meta_knowledge)
        w_meta = torch.reshape(w_meta, (-1, self.in_channels, self.d))
        w_meta = self.w2_linear(w_meta)
        w_meta = torch.reshape(w_meta, (-1, self.in_channels, self.out_channels, self.kernel_size)).permute(0, 2, 1, 3)
        w_meta = torch.reshape(w_meta, (-1, self.in_channels, self.kernel_size))
        b_meta = self.b_linear(meta_knowledge).view(-1)

        # print("w_meta: {}, b_meta: {}".format(w_meta.shape, b_meta.shape))

        batch_size, node_num, seq_len_in, _ = input.shape
        input = torch.reshape(input, (batch_size*node_num, seq_len_in, self.in_channels)).permute(0, 2, 1)
        group_input = torch.reshape(input, (-1, seq_len_in)).unsqueeze(0)

        # print("group_input shape is", group_input.shape, " device:", group_input.device)

        group_output = F.conv1d(group_input, weight=w_meta, bias=b_meta, groups=batch_size*node_num, stride=self.stride, padding=self.padding, dilation=self.dilation)
        group_output = torch.reshape(group_output, (1, batch_size, node_num, self.out_channels, -1)).squeeze(0)
        group_output = group_output.permute(0,1,3,2)
        return group_output

class MetaTemporalBlock(nn.Module):
    def __init__(self, meta_dim, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(MetaTemporalBlock, self).__init__()
        self.conv1 = MetaConv1d(meta_dim, n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = MetaConv1d(meta_dim, n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1, self.conv2, self.chomp2, self.relu2, self.dropout2)

        self.downsample = MetaConv1d(meta_dim, n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, meta_knowledge, x):
        # print("[MetaTemporalBlock] in block")
        # Split sequential network
        out1 = self.conv1(meta_knowledge, x)
        # print("out1 shape is",out1.shape)
        out1 = self.chomp1(out1)
        # print("after chomp, out1 shape is", out1.shape)
        out1 = self.dropout1(self.relu1(out1))
        out2 = self.chomp2(self.conv2(meta_knowledge, out1))
        out = self.dropout2(self.relu2(out2))
        # print("[MetaTemporalBlock] out shape is", out.shape)
        res = x if self.downsample is None else self.downsample(meta_knowledge, x)
        return self.relu(out + res)

class MetaTemporalConvNet(nn.Module):
    def __init__(self, meta_dim, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(MetaTemporalConvNet, self).__init__()
        layers = []
        self.num_levels = len(num_channels)
        for i in range(self.num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [MetaTemporalBlock(meta_dim, in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size, padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        # self.network = nn.Sequential(*layers)
        self.layers = nn.ModuleList(layers)

    def forward(self, meta_knowledge, x):
        for i in range(self.num_levels):
            out = self.layers[i](meta_knowledge, x)
            x = out
        return out

class MetaTCN(nn.Module):
    def __init__(self, model_args, task_args):
        super(MetaTCN, self).__init__()
        self.model_args = model_args
        self.task_args = task_args
        self.meta_dim = model_args['meta_dim']
        self.message_dim = model_args['message_dim']
        self.hidden_dim = model_args['hidden_dim']
        self.output_dim = model_args['output_dim']
        self.build()

    def build(self):
        self.mk_learner = STMetaLearner(self.model_args, self.task_args)
        self.tcn_1 = MetaTemporalConvNet(self.meta_dim, num_inputs=self.message_dim, num_channels=[self.hidden_dim, self.output_dim], kernel_size=3)
        self.tcn_2 = MetaTemporalConvNet(self.meta_dim, num_inputs=self.message_dim, num_channels=[self.hidden_dim, self.output_dim], kernel_size=2)
        # self.tcn_3 = MetaTemporalConvNet(self.meta_dim, num_inputs=self.message_dim, num_channels=[self.hidden_dim, self.output_dim], kernel_size=4)
        self.predictor = nn.Linear(self.task_args['his_num'], self.task_args['pred_num'])
    
    def forward(self, data, A_wave):
        batch_size, node_num, seq_len, c_in = data.x.shape
        meta_knowledge = self.mk_learner(data)
        meta_graph_structure = torch.bmm(meta_knowledge, meta_knowledge.permute(0, 2, 1))
        meta_graph_structure = F.relu(meta_graph_structure)
        # softmax_func = nn.Softmax(dim=-1)
#         meta_graph_structure = softmax_func(torch.relu(meta_graph_structure))

        output_1 = self.tcn_1(meta_knowledge, data.x)
        output_2 = self.tcn_2(meta_knowledge, data.x)
        # output_3 = self.tcn_3(meta_knowledge, data.x)
        output = output_1 + output_2

        output = torch.reshape(output, (batch_size, node_num, seq_len))
        output = self.predictor(output)
        return output, meta_graph_structure

class MetaConv2d_update(nn.Module):
    def __init__(self, meta_dim, in_channels, out_channels, kernel_size=3):
        super(MetaConv2d_update, self).__init__()
        self.meta_dim = meta_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.build()
    
    def build(self):
        self.d = self.meta_dim
        self.w1_linear = nn.Linear(self.meta_dim, self.in_channels * self.d)
        self.w2_linear = nn.Linear(self.d, self.out_channels * self.kernel_size * 1)
        self.b_linear = nn.Linear(self.meta_dim, self.out_channels)
    
    def forward(self, meta_knowledge, input):
        """
        : param meta_knowledge shape is [batch_size, node_num, meta_dim]
        : param input shape is [batch_size, node_num, seq_len_in, input_dim]
        : return output shape is [batch_size, node_num, seq_len_out, output_dim]
        """
        # [B, N, d_mk] -> [B*N, d_mk] -> [B*N, C_in*d] -> [B*N, C_in, d] -> [B*N, C_in, C_out*kernel_size*1] -> [B*N, C_out, C_in, kernel_size] -> [B*N*C_out, C_in, 1, kernel_size]
        meta_knowledge = torch.reshape(meta_knowledge, (-1, self.meta_dim))
        w_meta = self.w1_linear(meta_knowledge)
        w_meta = torch.reshape(w_meta, (-1, self.in_channels, self.d))
        w_meta = self.w2_linear(w_meta)
        w_meta = torch.reshape(w_meta, (-1, self.in_channels, self.out_channels, self.kernel_size)).permute(0, 2, 1, 3)
        w_meta = torch.reshape(w_meta, (-1, self.in_channels, self.kernel_size)).unsqueeze(2)
        b_meta = self.b_linear(meta_knowledge).view(-1)

        # print("meta_knowledge:{}, w:{}, b:{}".format(meta_knowledge.shape, w_meta.shape, b_meta.shape))

        # view batch conv2d as group conv2d
        batch_size, node_num, seq_len_in, _ = input.shape
        # print("input shape is", input.shape)
        input = torch.reshape(input, (batch_size*node_num, seq_len_in, self.in_channels)).permute(0, 2, 1)
        input = input.unsqueeze(2)
        group_input = torch.reshape(input, (-1, 1, seq_len_in)).unsqueeze(0)
        # print("group_input shape is", group_input.shape)
        group_output = F.conv2d(group_input, weight=w_meta, bias=b_meta, groups=batch_size*node_num)
        # print("group_output shape is", group_output.shape)
        group_output = torch.reshape(group_output, (1, batch_size, node_num, self.out_channels, -1)).squeeze(0)
        group_output = group_output.permute(0, 1, 3, 2)
        return group_output

class MetaConv2d(nn.Module):
    def __init__(self, meta_dim, in_channels, out_channels, kernel_size=3):
        super(MetaConv2d, self).__init__()
        self.meta_dim = meta_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.build()

    def build(self):
        self.w_linear = nn.Linear(self.meta_dim, self.in_channels * self.out_channels * self.kernel_size)
        self.b_linear = nn.Linear(self.meta_dim, self.out_channels)
    
    def forward(self, meta_knowledge, input):
        """
        : param meta_knowledge shape is [batch_size, node_num, meta_dim]
        : param input shape is [batch_size, node_num, seq_len_in, input_dim]
        : tmp w_meta shape is [batch_size*node_num*out_channels, in_channels, 1, kernel_size]
        : return output shape is [batch_size, node_num, seq_len_out, output_dim]
        """
        # generate weights and bias
        meta_knowledge = torch.reshape(meta_knowledge, (-1, self.meta_dim))
        w_meta = self.w_linear(meta_knowledge)
        w_meta = torch.reshape(w_meta, (-1, self.out_channels, self.in_channels, self.kernel_size))
        w_meta = torch.reshape(w_meta, (-1, self.in_channels, self.kernel_size)).unsqueeze(2)
        b_meta = self.b_linear(meta_knowledge).view(-1)

        print("meta_knowledge:{}, w:{}, b:{}".format(meta_knowledge.shape, w_meta.shape, b_meta.shape))

        # view batch conv2d as group conv2d
        batch_size, node_num, seq_len_in, _ = input.shape
        input = torch.reshape(input, (batch_size*node_num, seq_len_in, self.in_channels)).permute(0, 2, 1)
        input = input.unsqueeze(2)
        group_input = torch.reshape(input, (-1, 1, seq_len_in)).unsqueeze(0)
        # print("group_input shape is", group_input.shape)
        group_output = F.conv2d(group_input, weight=w_meta, bias=b_meta, groups=batch_size*node_num)
        # print("group_output shape is", group_output.shape)
        group_output = torch.reshape(group_output, (1, batch_size, node_num, self.out_channels, -1)).squeeze(0)
        group_output = group_output.permute(0, 1, 3, 2)
        return group_output

class TimeBlock(nn.Module):
    """
    Neural network block that applies a temporal convolution to each node of
    a graph in isolation.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3):
        """
        :param in_channels: Number of input features at each node in each time
        step.
        :param out_channels: Desired number of output channels at each node in
        each time step.
        :param kernel_size: Size of the 1D temporal kernel.
        """
        super(TimeBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv2 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv3 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))

    def forward(self, X):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels)
        :return: Output data of shape (batch_size, num_nodes,
        num_timesteps_out, num_features_out=out_channels)
        """
        # Convert into NCHW format for pytorch to perform convolutions.
        X = X.permute(0, 3, 1, 2)
        temp = self.conv1(X) + self.conv2(X)
        gate = torch.sigmoid(self.conv3(X))
        out = F.relu(gate * temp)
        # temp = self.conv1(X) + torch.sigmoid(self.conv2(X))
        # out = F.relu(temp + self.conv3(X))
        # Convert back from NCHW to NHWC
        out = out.permute(0, 2, 3, 1)
        return out

class STMetaLearner_update(nn.Module):
    def __init__(self, model_args, task_args):
        super(STMetaLearner_update, self).__init__()
        self.model_args = model_args
        self.task_args = task_args
        self.node_feature_dim = model_args['node_feature_dim']
        self.edge_feature_dim = model_args['edge_feature_dim']
        self.message_dim = model_args['message_dim']
        self.hidden_dim = model_args['hidden_dim']
        self.his_num = task_args['his_num']
        self.pred_num = task_args['pred_num']
        self.meta_dim = model_args['meta_dim']
        self.gat_input = self.his_num-2
        self.build()
    
    def build(self):
        self.temporal_learner_1 = TimeBlock(in_channels=self.message_dim, out_channels=self.hidden_dim)
        self.spatial_learner = GATConv(self.hidden_dim, self.meta_dim, 3, False)
        self.temporal_learner_2 = TimeBlock(in_channels=self.meta_dim, out_channels=1)
        self.mk_linear = nn.Linear(self.gat_input-2, self.meta_dim)

    def forward(self, data):
        """
        : data.x shape is [batch_size, node_num, his_len, message_dim] 
        : after temporal_learner_1 shape is [batch_size, node_num, new_len, 1]
        """
        tp_input = data.x
        batch_size, node_num, his_len, message_dim = tp_input.shape
        tp_output = self.temporal_learner_1(tp_input)
        sp_inputs = torch.reshape(tp_output, (-1, self.gat_input, self.hidden_dim))
        for i in range(self.gat_input):
            sp_output = self.spatial_learner(sp_inputs[:, i, :], data.edge_index).unsqueeze(1)
            if i == 0:
                sp_outputs = sp_output
            else:
                sp_outputs = torch.cat((sp_outputs, sp_output), 1)
        sp_outputs = F.relu(sp_outputs)
        tp_input = torch.reshape(sp_outputs, (batch_size, node_num, self.gat_input, self.meta_dim))        
        mk_feature = self.temporal_learner_2(tp_input).squeeze(-1)
        mk_feature = F.relu(mk_feature)
        output = self.mk_linear(mk_feature)
        meta_knowledge = torch.reshape(output, (-1, node_num, self.model_args['meta_dim']))

        # sp_input = F.relu(torch.reshape(tp_output, (-1, self.gat_input)))
        # sp_output = F.relu(self.spatial_learner(sp_input, data.edge_index))
        # meta_knowledge = self.mk_linear(sp_output)
        # meta_knowledge = torch.reshape(meta_knowledge, (-1, node_num, self.model_args['meta_dim']))
        return meta_knowledge

class MetaTimeBlock(nn.Module):
    def __init__(self, meta_dim, in_channels, out_channels, kernel_size=3):
        super(MetaTimeBlock, self).__init__()
        self.meta_dim = meta_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.build()

    def build(self):
        self.conv1 = MetaConv2d_update(self.meta_dim, self.in_channels, self.out_channels, self.kernel_size)
        self.conv2 = MetaConv2d_update(self.meta_dim, self.in_channels, self.out_channels, self.kernel_size)
        self.conv3 = MetaConv2d_update(self.meta_dim, self.in_channels, self.out_channels, self.kernel_size)

        # self.conv1 = MetaConv2d(self.meta_dim, self.in_channels, self.out_channels, self.kernel_size)
        # self.conv2 = MetaConv2d(self.meta_dim, self.in_channels, self.out_channels, self.kernel_size)
        # self.conv3 = MetaConv2d(self.meta_dim, self.in_channels, self.out_channels, self.kernel_size)
        
    def forward(self, meta_knowledge, input):
        tmp = self.conv1(meta_knowledge, input) + self.conv2(meta_knowledge, input)
        gate = torch.sigmoid(self.conv3(meta_knowledge, input))
        out = F.relu(gate * tmp)
        return out

class MetaSTGCNBlock(nn.Module):
    def __init__(self, meta_dim, in_channels, spatial_channels, out_channels, kernel_size=3):
        super(MetaSTGCNBlock, self).__init__()
        self.meta_dim = meta_dim
        self.in_channels = in_channels
        self.spatial_channels = spatial_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.build()

    def build(self):
        self.temporal_1 = MetaTimeBlock(self.meta_dim, self.in_channels, self.out_channels, self.kernel_size)
        self.theta = nn.Parameter(torch.FloatTensor(self.out_channels, self.spatial_channels))
        self.temporal_2 = MetaTimeBlock(self.meta_dim, self.spatial_channels, self.out_channels, self.kernel_size)
        self.reset_param()
    
    def reset_param(self):
        stdv = 1. / math.sqrt(self.theta.shape[1])
        self.theta.data.uniform_(-stdv, stdv)
    
    def forward(self, meta_knowledge, input, A_hat):
        """
        : param: meta_knowledge shape is [batch_size, node_num, meta_dim]
        : param: input shape is [batch_size, node_num, seq_len, input_dim]
        : param: A_hat shape is [node_num, node_num]
        """
        t_1 = self.temporal_1(meta_knowledge, input)
        lfs = torch.einsum("ij,jklm->kilm", [A_hat, t_1.permute(1, 0, 2, 3)])
        t_2 = F.relu(torch.matmul(lfs, self.theta))
        t_3 = self.temporal_2(meta_knowledge, t_2)
        return t_3

class STGCNBlock(nn.Module):
    """
    Neural network block that applies a temporal convolution on each node in
    isolation, followed by a graph convolution, followed by another temporal
    convolution on each node.
    """

    # def __init__(self, in_channels, spatial_channels, out_channels,
                #  num_nodes):
    def __init__(self, in_channels, spatial_channels, out_channels):
        """
        :param in_channels: Number of input features at each node in each time
        step.
        :param spatial_channels: Number of output channels of the graph
        convolutional, spatial sub-block.
        :param out_channels: Desired number of output features at each node in
        each time step.
        :param num_nodes: Number of nodes in the graph.
        """
        super(STGCNBlock, self).__init__()
        self.temporal1 = TimeBlock(in_channels=in_channels,
                                   out_channels=out_channels)
        self.Theta1 = nn.Parameter(torch.FloatTensor(out_channels,
                                                     spatial_channels))
        self.temporal2 = TimeBlock(in_channels=spatial_channels,
                                   out_channels=out_channels)
        # self.batch_norm = nn.BatchNorm2d(num_nodes)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.Theta1.shape[1])
        self.Theta1.data.uniform_(-stdv, stdv)

    def forward(self, X, A_hat):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels).
        :param A_hat: Normalized adjacency matrix.
        :return: Output data of shape (batch_size, num_nodes,
        num_timesteps_out, num_features=out_channels).
        """
        # print("[Block] X shape is", X.shape)
        t = self.temporal1(X)
        # print("[Block] t1 shape is", t.shape)
        # print("t shape is {}, A_hat shape is {}".format(t.shape, A_hat.shape))
        lfs = torch.einsum("ij,jklm->kilm", [A_hat, t.permute(1, 0, 2, 3)])
        # print("lfs shape is {}".format(lfs.shape))
        t2 = F.relu(torch.einsum("ijkl,lp->ijkp", [lfs, self.Theta1]))
        # t2 = F.relu(torch.matmul(lfs, self.Theta1))
        # print("[Block] t2 shape is", t2.shape)
        t3 = self.temporal2(t2)
        # print("[Block] t3 shape is", t3.shape)
        # return self.batch_norm(t3)
        return t3

class STGCN(nn.Module):
    def __init__(self, model_args, task_args):
        super(STGCN, self).__init__()
        self.model_args = model_args
        self.task_args = task_args
        self.meta_dim = model_args['meta_dim']
        self.message_dim = model_args['message_dim']
        self.his_num = task_args['his_num']
        self.pred_num = task_args['pred_num']
        self.build()
    
    def build(self):
        self.block1 = STGCNBlock(in_channels=self.message_dim, out_channels=64,
                                 spatial_channels=16)
        self.block2 = STGCNBlock(in_channels=64, out_channels=64,
                                 spatial_channels=16)
        self.last_temporal = TimeBlock(in_channels=64, out_channels=64)
        self.fully = nn.Linear((self.his_num - 2 * 5) * 64,
                               self.pred_num)
    
    def forward(self, data, A_hat):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels).
        :param A_hat: Normalized adjacency matrix.
        """
        X = data.x
        # print("x shape is", X.shape)
        out1 = self.block1(X, A_hat)
        # print("out1 shape is", out1.shape)
        out2 = self.block2(out1, A_hat)
        # print("out2 shape is", out2.shape)
        out3 = self.last_temporal(out2)
        # print("out3 shape is", out3.shape)
        out4 = self.fully(out3.reshape((out3.shape[0], out3.shape[1], -1)))
        # print("out4 shape is", out4.shape)
        return out4, A_hat

class MetaSTGCN(nn.Module):
    def __init__(self, model_args, task_args):
        super(MetaSTGCN, self).__init__()
        self.model_args = model_args
        self.task_args = task_args
        self.meta_dim = model_args['meta_dim']
        self.message_dim = model_args['message_dim']
        self.his_num = task_args['his_num']
        self.pred_num = task_args['pred_num']
        self.build()
    
    def build(self):
        # self.block_1 = MetaSTGCNBlock(self.meta_dim, in_channels=self.message_dim, out_channels=64, spatial_channels=16)
        # self.block_2 = MetaSTGCNBlock(in_channels=64, out_channels=64, spatial_channels=16)
        # self.last_temporal = TimeBlock(in_channels=64, out_channels=64)
        # self.predictor = nn.Linear((self.his_num - 2*5)*64, self.pred_num)
        self.block_1 = MetaSTGCNBlock(self.meta_dim, in_channels=self.message_dim, out_channels=20, spatial_channels=12)
        self.block_2 = MetaSTGCNBlock(self.meta_dim, in_channels=20, out_channels=20, spatial_channels=12)
        self.last_temporal = TimeBlock(in_channels=20, out_channels=20)
        self.predictor = nn.Linear((self.his_num - 2*5)*20, self.pred_num)

    def forward(self, meta_knowledge, X, A_hat):
        t1 = time.time()
        # print("[Net-1] x shape is", X.shape)
        out1 = self.block_1(meta_knowledge, X, A_hat)
        t2 = time.time()
        # print("[module-2] MetaSTGCNBlock time is", t2-t1)
        # print("[Net-2] out1 shape is", out1.shape)
        out2 = self.block_2(meta_knowledge, out1, A_hat)
        t3 = time.time()
        # print("[module-3] STGCNBlock time is", t3-t2)
        # print("[Net-3] out2 shape is", out2.shape)
        out3 = self.last_temporal(out2)
        t4 = time.time()
        # print("[module-4] last_temporal time is", t4-t3)
        out4 = self.predictor(out3.reshape(out3.shape[0], out3.shape[1], -1))
        t5 = time.time()
        # print("[module-5] predictor time is", t5-t4)
        # print('-------------------------------------------')
        return out4

class MetaSTGNN(nn.Module):
    def __init__(self, model_args, task_args):
        super(MetaSTGNN, self).__init__()
        self.model_args = model_args
        self.task_args = task_args
        self.build()
    
    def build(self):
        self.meta_learner = STMetaLearner(self.model_args, self.task_args)
        self.meta_network = MetaSTGCN(self.model_args, self.task_args)
    
    def forward(self, data, A_hat):
        time_mk_start = time.time()
        meta_knowledge = self.meta_learner(data)
        meta_graph_structure = torch.bmm(meta_knowledge, meta_knowledge.permute(0,2,1))
        meta_graph_structure = F.relu(meta_graph_structure)
        # softmax_func = nn.Softmax(dim=-1)
#         meta_graph_structure = softmax_func(torch.relu(meta_graph_structure))
        time_mk_end = time.time()
        # print("[module-1] meta_learner time is", time_mk_end - time_mk_start)
        output = self.meta_network(meta_knowledge, data.x, A_hat)
        return output, meta_graph_structure

class MetaGRUCell(nn.Module):
    def __init__(self, meta_dim, input_dim, hidden_dim):
        super(MetaGRUCell, self).__init__()
        self.meta_dim = meta_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.build()
    
    def build(self):
        self.r_x_linear = MetaLinear(self.meta_dim, self.input_dim, self.hidden_dim)
        self.r_h_linear = MetaLinear(self.meta_dim, self.hidden_dim, self.hidden_dim)
        self.z_x_linear = MetaLinear(self.meta_dim, self.input_dim, self.hidden_dim)
        self.z_h_linear = MetaLinear(self.meta_dim, self.hidden_dim, self.hidden_dim)
        self.c_x_linear = MetaLinear(self.meta_dim, self.input_dim, self.hidden_dim)
        self.c_h_linear = MetaLinear(self.meta_dim, self.hidden_dim, self.hidden_dim)

    def forward(self, meta_knowledge, x, hidden):
        """
        : params meta_knowledge shape is [batch_size, node_num, meta_dim]
        : params x shape is [batch_size, node_num, input_dim]
        : params hidden shape is [batch_size, node_num, hidden_dim]
        : return next_hidden shape is [batch_size, node_num, hidden_dim]
        """
        r = torch.sigmoid(
            self.r_x_linear(meta_knowledge, x) + self.r_h_linear(meta_knowledge, hidden)
        )
        z = torch.sigmoid(
            self.z_x_linear(meta_knowledge, x) + self.z_h_linear(meta_knowledge, hidden)
        )
        c = torch.tanh(
            self.c_x_linear(meta_knowledge, x) + r * self.c_h_linear(meta_knowledge, hidden)
        )
        next_hidden = (1 - z) * c + z * hidden
        return next_hidden

class MetaGRU(nn.Module):
    def __init__(self, model_args, task_args, input_dim=None, hidden_dim=None, output_dim=None):
        super(MetaGRU, self).__init__()
        self.meta_dim = model_args['meta_dim']
        self.model_args = model_args
        self.task_args = task_args
        self.input_dim = model_args['message_dim'] if input_dim == None else input_dim
        self.hidden_dim = model_args['hidden_dim'] if hidden_dim == None else hidden_dim
        self.output_dim = model_args['output_dim'] if output_dim == None else output_dim

        self.build()
    
    def build(self):
        self.metagru_layer = MetaGRUCell(self.meta_dim, self.input_dim, self.hidden_dim)
        self.output_layer = nn.Linear(self.hidden_dim, self.output_dim)
    
    def forward(self, meta_knowledge, data, hidden=None, input=None):
        """
        : params meta_knowledge shape is [batch_size, node_num, meta_dim]
        : params input shape is [batch_size, node_num, seq_len, input_dim]
        : return output shape is [batch_size, node_num, seq_len, output_dim]
        """
        input = data.x if input == None else input
        batch_size, node_num, seq_len, _ = input.shape
        if hidden is None:
            hidden = torch.zeros(batch_size, node_num, self.hidden_dim).cuda()
        h_outputs = []
        for i in range(seq_len):
            input_i = input[:, :, i, :]
            hidden = self.metagru_layer(meta_knowledge, input_i, hidden)
            output = self.output_layer(hidden)
            h_outputs.append(output.unsqueeze(0))
        h_outputs = torch.cat(h_outputs, 0)
        return output, h_outputs

class RandomGRU(nn.Module):
    def __init__(self, model_args, task_args) -> None:
        super(RandomGRU, self).__init__()
        self.model_args = model_args
        self.task_args = task_args
        self.build()

    def build(self):
        pass

    def forward(self, data, A_wave):
        pass

class MetaSTNN(nn.Module):
    def __init__(self, model_args, task_args):
        super(MetaSTNN, self).__init__()
        self.model_args = model_args
        self.task_args = task_args
        self.build()
    
    def build(self):
        self.mk_learner = STMetaLearner_add(self.model_args, self.task_args)
        self.meta_gru = MetaGRU(self.model_args, self.task_args)
        self.predictor = nn.Linear(self.task_args['his_num'], self.task_args['pred_num'])
    
    def forward(self, data, A_wave):
        """
        : return output shape is [batch_size, node_num, output_seq_len]
        """
        meta_knowledge = self.mk_learner(data)
        meta_graph_structure = torch.bmm(meta_knowledge, meta_knowledge.permute(0,2,1))
        meta_graph_structure = F.relu(meta_graph_structure)
        # softmax_func = nn.Softmax(dim=-1)
#         meta_graph_structure = softmax_func(torch.relu(meta_graph_structure))
        # print("meta_knowledge shape {}, meta_graph_structure shape {}".format(meta_knowledge.shape, meta_graph_structure.shape))
        # print("meta_graph_structure shape is", meta_graph_structure.shape)
        # print("[MetaSTNN] meta knowledge shape is", meta_knowledge.shape)
        _, gru_h_outputs = self.meta_gru(meta_knowledge, data)
        input = gru_h_outputs.squeeze(-1).permute(1, 2, 0)
        input = F.relu(input)
        output = self.predictor(input)
        return output, meta_graph_structure

class GRUModel(nn.Module):
    def __init__(self, model_args, task_args):
        super(GRUModel, self).__init__()
        self.model_args = model_args
        self.task_args = task_args
        self.input_dim = model_args['message_dim']
        self.hidden_dim = model_args['hidden_dim']
        self.output_dim = model_args['output_dim']
        self.build()
    
    def build(self):
        self.gru_layer = nn.GRU(self.input_dim, self.hidden_dim, batch_first=True)
        self.predictor_1 = nn.Linear(self.hidden_dim, self.output_dim)
        self.predictor_2 = nn.Linear(self.task_args['his_num'], self.task_args['pred_num'])
    
    def forward(self, data, A_wave):
        """
        : param input shape is [batch_size, node_num, seq_len, input_dim] -> [batch_size * node_num, seq_len, hidden_dim](after gru)
        """
        self.gru_layer.flatten_parameters() 
        batch_size, node_num, seq_len, input_dim = data.x.shape
        gru_input = torch.reshape(data.x, (batch_size * node_num, seq_len, input_dim))
        gru_outputs, _ = self.gru_layer(gru_input)
        output = self.predictor_1(gru_outputs).squeeze(-1)
        output = torch.reshape(self.predictor_2(output), (batch_size, node_num, -1))
        return output, A_wave

class MetaGATConv(MessagePassing):

    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, heads: int = 1, concat: bool = True,
                 negative_slope: float = 0.2, dropout: float = 0.0,
                 add_self_loops: bool = True, bias: bool = True, meta_dim: int = 16, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(MetaGATConv, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.meta_dim = meta_dim

        # self.lin = nn.Linear(in_channels, heads * out_channels, False)
        self.lin = MetaLinear(meta_dim, in_channels, heads * out_channels, False)
        
        # The learnable parameters to compute attention coefficients:
        self.att_src = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_dst = nn.Parameter(torch.Tensor(1, heads, out_channels))

        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self._alpha = None


    def forward(self, meta_knowledge, x: Union[Tensor, OptPairTensor], edge_index: Adj, size: Size = None, return_attention_weights=None):

        H, C = self.heads, self.out_channels

        # We first transform the input node features. If a tuple is passed, we
        # transform source and target node features via separate weights:
        if isinstance(x, Tensor):
            assert x.dim() == 2, "Static graphs not supported in 'GATConv'"
            x_src = x_dst = self.lin(meta_knowledge, x, dim=2).view(-1, H, C)
        else:  # Tuple of source and target node features:
            x_src, x_dst = x
            assert x_src.dim() == 2, "Static graphs not supported in 'GATConv'"
            x_src = self.lin_src(x_src).view(-1, H, C)
            if x_dst is not None:
                x_dst = self.lin_dst(x_dst).view(-1, H, C)

        x = (x_src, x_dst)

        # Next, we compute node-level attention coefficients, both for source
        # and target nodes (if present):
        alpha_src = (x_src * self.att_src).sum(dim=-1)
        alpha_dst = None if x_dst is None else (x_dst * self.att_dst).sum(-1)
        alpha = (alpha_src, alpha_dst)

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                # We only want to add self-loops for nodes that appear both as
                # source and target nodes:
                num_nodes = x_src.size(0)
                if x_dst is not None:
                    num_nodes = min(num_nodes, x_dst.size(0))
                num_nodes = min(size) if size is not None else num_nodes
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                edge_index = set_diag(edge_index)

        # propagate_type: (x: OptPairTensor, alpha: OptPairTensor)
        out = self.propagate(edge_index, x=x, alpha=alpha, size=size)

        alpha = self._alpha
        assert alpha is not None
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out += self.bias

        if isinstance(return_attention_weights, bool):
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out


    def message(self, x_j: Tensor, alpha_j: Tensor, alpha_i: OptTensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        # Given egel-level attention coefficients for source and target nodes,
        # we simply need to sum them up to "emulate" concatenation:
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha  # Save for later use.
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)

class GATGRU(nn.Module):
    def __init__(self, model_args, task_args):
        super().__init__()
        self.model_args = model_args
        self.task_args = task_args
        self.gat_in_channel = model_args['message_dim']
        self.gat_out_channel = model_args['hidden_dim']
        self.hidden_dim = model_args['hidden_dim']
        self.meta_dim = model_args['meta_dim']
        self.output_dim = model_args['output_dim']
        self.build()
    
    def build(self):
        self.gat_conv = GATConv(self.gat_in_channel * self.task_args['his_num'], self.gat_out_channel * self.task_args['his_num'], heads=3, concat=False)
        self.gru_layer = nn.GRU(self.gat_out_channel, self.hidden_dim, batch_first=True)
        self.predictor_1 = nn.Linear(self.hidden_dim, self.output_dim)
        self.predictor_2 = nn.Linear(self.task_args['his_num'], self.task_args['pred_num'])

    def forward(self, data, A_wave):
        batch_size, node_num, seq_len, input_dim = data.x.shape
        gat_inputs = torch.reshape(data.x, (-1, self.task_args['his_num'], self.gat_in_channel))
        gat_inputs = torch.reshape(gat_inputs, (-1, self.gat_in_channel * self.task_args['his_num']))
        gat_outputs = self.gat_conv(gat_inputs, data.edge_index)
        gru_input = torch.reshape(gat_outputs, (-1, self.task_args['his_num'], self.gat_out_channel))
        gru_outputs, _ = self.gru_layer(gru_input)
        output = self.predictor_1(gru_outputs).squeeze(-1)
        output = torch.reshape(self.predictor_2(output), (batch_size, node_num, -1))
        return output, A_wave

class MetaGATGRU(nn.Module):
    def __init__(self, model_args, task_args):
        super(MetaGATGRU, self).__init__()
        self.model_args = model_args
        self.task_args = task_args
        self.gat_in_channel = model_args['message_dim']
        self.gat_out_channel = model_args['hidden_dim']
        self.meta_dim = model_args['meta_dim']
        self.hidden_dim = model_args['hidden_dim']
        self.output_dim = model_args['output_dim']
        self.his_num = task_args['his_num']
        self.build()
    
    def build(self):
        self.mk_learner = STMetaLearner(self.model_args, self.task_args)
        self.meta_gat = MetaGATConv(self.gat_in_channel * self.his_num, self.gat_out_channel * self.his_num, heads=3, concat=False, meta_dim=self.meta_dim)
        self.meta_gru = MetaGRU(self.model_args, self.task_args, input_dim=self.gat_out_channel)
        self.predictor = nn.Linear(self.task_args['his_num'], self.task_args['pred_num'])

        # self.mk_learner = STMetaLearner_update(self.model_args, self.task_args)
        # self.meta_gru = MetaGRU(self.model_args, self.task_args, output_dim=self.hidden_dim)
        # self.meta_gat = MetaGATConv(self.hidden_dim * self.his_num, self.task_args['pred_num'], heads=3, concat=False, meta_dim=self.meta_dim) 
        # self.predictor = nn.Linear(self.task_args['pred_num'], self.task_args['pred_num'])

    def forward_grugat(self, data, A_wave):
        batch_size, node_num, seq_len, _ = data.x.shape
        meta_knowledge = self.mk_learner(data)
        meta_graph_structure = torch.bmm(meta_knowledge, meta_knowledge.permute(0,2,1))
        meta_graph_structure = F.relu(meta_graph_structure)
        # softmax_func = nn.Softmax(dim=-1)
#         meta_graph_structure = softmax_func(torch.relu(meta_graph_structure))

        _, gru_h_outputs = self.meta_gru(meta_knowledge, data)
        gru_h_outputs = gru_h_outputs.permute(1, 2, 0, 3)
        
        sp_input = torch.reshape(gru_h_outputs, (-1, self.his_num, self.hidden_dim))
        sp_input = torch.reshape(sp_input, (-1, self.his_num * self.hidden_dim))
        sp_output = self.meta_gat(meta_knowledge, F.relu(sp_input), data.edge_index)
        output = self.predictor(sp_output)
        output = torch.reshape(output, (batch_size, node_num, self.task_args['pred_num']))
        return output, meta_graph_structure
        
    def forward(self, data, A_wave):
        meta_knowledge = self.mk_learner(data)
        meta_graph_structure = torch.bmm(meta_knowledge, meta_knowledge.permute(0,2,1))
        meta_graph_structure = F.relu(meta_graph_structure)
        # softmax_func = nn.Softmax(dim=-1)
#         meta_graph_structure = softmax_func(torch.relu(meta_graph_structure))
        # print("meta_knowledge shape {}, meta_graph_structure shape {}".format(meta_knowledge.shape, meta_graph_structure.shape))
        
        meta_knowledge_gat = torch.reshape(meta_knowledge, (-1, self.meta_dim))
        gat_inputs = torch.reshape(data.x, (-1, self.task_args['his_num'], self.gat_in_channel))

        gat_inputs = torch.reshape(gat_inputs, (-1, self.task_args['his_num'] * self.gat_in_channel))
        gat_outs = self.meta_gat(meta_knowledge, gat_inputs, data.edge_index)
        
        gat_outs = F.relu(torch.reshape(gat_outs, (-1, data.node_num, self.task_args['his_num'], self.model_args['hidden_dim'])))
        # print("gat_outs shape is", gat_outs.shape)

        _, gru_h_outputs = self.meta_gru(meta_knowledge, data, input=gat_outs)
        input = gru_h_outputs.squeeze(-1).permute(1, 2, 0)
        # print("gru_h_outputs shape is", input.shape)
        input = F.relu(input)
        output = self.predictor(input)
        # print("output shape is", output.shape)
        # print("--------------------------------------")
        return output, meta_graph_structure

class MetaSTGAT(nn.Module):
    def __init__(self, model_args, task_args):
        super(MetaSTGAT, self).__init__()
        self.model_args = model_args
        self.task_args = task_args
        self.message_dim = model_args['message_dim']
        self.meta_dim = model_args['meta_dim']
        self.hidden_dim = model_args['hidden_dim']
        self.output_dim = model_args['output_dim']
        self.his_num = task_args['his_num']
        self.pred_num = task_args['pred_num']
        self.build()
    
    def build(self):
        self.mk_learner = STMetaLearner(self.model_args, self.task_args)
        self.gat_layer_one = MetaGATConv(in_channels=self.his_num * self.message_dim, out_channels=self.hidden_dim, heads=3, concat=True, meta_dim=self.meta_dim)
        self.gat_layer_two = MetaGATConv(in_channels=3*self.hidden_dim, out_channels=self.pred_num, heads=3, concat=False, meta_dim=self.meta_dim)
        self.predictor = nn.Linear(self.pred_num, self.pred_num)

    def forward(self, data, A_wave):
        batch_size, node_num, seq_len, _ = data.x.shape
        meta_knowledge = self.mk_learner(data)
        meta_graph_structure = torch.bmm(meta_knowledge, meta_knowledge.permute(0,2,1))
        meta_graph_structure = F.relu(meta_graph_structure)
        # softmax_func = nn.Softmax(dim=-1)
#         meta_graph_structure = softmax_func(torch.relu(meta_graph_structure))

        meta_knowledge = torch.reshape(meta_knowledge, (-1, self.meta_dim))
        sp_input = torch.reshape(data.x, (-1, seq_len, self.message_dim))
        sp_input = torch.reshape(sp_input, (-1, seq_len * self.message_dim))
        # print("meta_knowledge: {}, input: {}".format(meta_knowledge.shape, sp_input.shape))
        sp_output_1 = F.relu(self.gat_layer_one(meta_knowledge, sp_input, data.edge_index))
        # print("sp_output_1 shape is", sp_output_1.shape)
        sp_output_2 = F.relu(self.gat_layer_two(meta_knowledge, sp_output_1, data.edge_index))
        
        output = self.predictor(sp_output_2)
        output = torch.reshape(output, (batch_size, node_num, self.pred_num))
        return output, meta_graph_structure

class RandomLinear(nn.Module):
    def __init__(self, meta_dim, in_feature_dim, out_feature_dim):
        super(RandomLinear, self).__init__()
        self.meta_dim = meta_dim
        self.in_feature_dim = in_feature_dim
        self.out_feature_dim = out_feature_dim
        self.build()
    
    def build(self):
        # self.w_z = Parameter(torch.fmod(torch.randn((self.meta_dim)),2))
        # self.b_z = Parameter(torch.fmod(torch.randn((self.meta_dim)),2))
        # self.w_hope = HyperNetwork(f_size=1, z_dim=self.meta_dim, out_size=self.out_feature_dim, in_size=self.in_feature_dim)
        # self.z_hope = HyperNetwork(f_size=1, z_dim=self.meta_dim, out_size=self.out_feature_dim, in_size=1)
        self.w_linear = nn.Linear(self.meta_dim, self.out_feature_dim*self.in_feature_dim)
        self.b_linear = nn.Linear(self.meta_dim, self.out_feature_dim)
        
    def forward(self, x, w_z, b_z):
        """
        x shape should be [N, *, in_feature]
        """
        w_kernel = torch.reshape(self.w_linear(w_z), (self.out_feature_dim, self.in_feature_dim))
        b_kernel = self.b_linear(b_z)
        y = F.linear(x, weight=w_kernel, bias=b_kernel)
        return y

class RandomGRUCell(nn.Module):
    def __init__(self, meta_dim, input_dim, hidden_dim):
        super(RandomGRUCell, self).__init__()
        self.meta_dim = meta_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.build()
    
    def build(self):
        self.r_x_linear = RandomLinear(self.meta_dim, self.input_dim, self.hidden_dim)
        self.r_h_linear = RandomLinear(self.meta_dim, self.hidden_dim, self.hidden_dim)
        self.z_x_linear = RandomLinear(self.meta_dim, self.input_dim, self.hidden_dim)
        self.z_h_linear = RandomLinear(self.meta_dim, self.hidden_dim, self.hidden_dim)
        self.c_x_linear = RandomLinear(self.meta_dim, self.input_dim, self.hidden_dim)
        self.c_h_linear = RandomLinear(self.meta_dim, self.hidden_dim, self.hidden_dim)

    def forward(self, x, hidden):
        """
        : params x shape is [batch_size, node_num, input_dim]
        : params hidden shape is [batch_size, node_num, hidden_dim]
        : return next_hidden shape is [batch_size, node_num, hidden_dim]
        """
        w_z = Parameter(torch.fmod(torch.randn((self.meta_dim)),2)) * 0.01
        b_z = Parameter(torch.fmod(torch.randn((self.meta_dim)),2)) * 0.01
        w_z, b_z = w_z.cuda(), b_z.cuda()
        r = torch.sigmoid(self.r_x_linear(x, w_z, b_z) + self.r_h_linear(hidden, w_z, b_z))
        z = torch.sigmoid(self.z_x_linear(x, w_z, b_z) + self.z_h_linear(hidden, w_z, b_z))
        c = torch.tanh(self.c_x_linear(x, w_z, b_z) + r * self.c_h_linear(hidden, w_z, b_z))
        next_hidden = (1 - z) * c + z * hidden
        return next_hidden

class RandomGRU(nn.Module):
    def __init__(self, model_args, task_args):
        super(RandomGRU, self).__init__()
        self.model_args = model_args
        self.task_args = task_args
        self.meta_dim = model_args['meta_dim']
        self.input_dim = model_args['message_dim']
        self.hidden_dim = model_args['hidden_dim']
        self.output_dim = model_args['output_dim']
        self.build()
    
    def build(self):
        self.metagru_layer = RandomGRUCell(self.meta_dim, self.input_dim, self.hidden_dim)
        self.output_layer = nn.Linear(self.hidden_dim, self.output_dim)
        self.predictor = nn.Linear(self.task_args['his_num'], self.task_args['pred_num'])
    
    def forward(self, data, matrix, hidden=None, input=None):
        """
        : params input shape is [batch_size, node_num, seq_len, input_dim]
        : return output shape is [batch_size, node_num, seq_len, output_dim]
        """
        input = data.x if input == None else input
        batch_size, node_num, seq_len, _ = input.shape
        if hidden is None:
            # print("hidden is None")
            hidden = torch.zeros(batch_size, node_num, self.hidden_dim).cuda()
        h_outputs = []
        for i in range(seq_len):
            input_i = input[:, :, i, :]
            hidden = self.metagru_layer(input_i, hidden)
            output = self.output_layer(hidden)
            h_outputs.append(output.unsqueeze(0))
        h_outputs = torch.cat(h_outputs, 0)

        input = h_outputs.squeeze(-1).permute(1, 2, 0)
        input = F.relu(input)
        output = self.predictor(input)
        return output, output

class MetaTCNGAT(nn.Module):
    def __init__(self, model_args, task_args):
        super(MetaTCNGAT, self).__init__()
        self.model_args = model_args
        self.task_args = task_args
        self.gat_in_channel = model_args['message_dim']
        self.gat_out_channel = model_args['hidden_dim']
        self.meta_dim = model_args['meta_dim']
        self.hidden_dim = model_args['hidden_dim']
        self.output_dim = model_args['output_dim']
        self.his_num = task_args['his_num']
        self.build()
    
    def build(self):
        self.mk_learner = STMetaLearner(self.model_args, self.task_args)
        self.meta_gat = MetaGATConv(self.gat_in_channel * self.his_num, self.gat_out_channel * self.his_num, heads=3, concat=False, meta_dim=self.meta_dim)

        self.tcn_1 = MetaTemporalConvNet(self.meta_dim, num_inputs=self.gat_out_channel, num_channels=[self.hidden_dim, self.output_dim], kernel_size=3)
        self.tcn_2 = MetaTemporalConvNet(self.meta_dim, num_inputs=self.gat_out_channel, num_channels=[self.hidden_dim, self.output_dim], kernel_size=2)

        self.predictor = nn.Linear(self.task_args['his_num'], self.task_args['pred_num'])
    
    def forward(self, data, A_wave):
        batch_size, node_num, seq_len, c_in = data.x.shape

        meta_knowledge = self.mk_learner(data)
        meta_graph_structure = torch.bmm(meta_knowledge, meta_knowledge.permute(0,2,1))
        meta_graph_structure = F.relu(meta_graph_structure)
        # softmax_func = nn.Softmax(dim=-1)
        # meta_graph_structure = softmax_func(torch.relu(meta_graph_structure))

        gat_inputs = torch.reshape(data.x, (-1, self.task_args['his_num'], self.gat_in_channel))
        gat_inputs = torch.reshape(gat_inputs, (-1, self.task_args['his_num'] * self.gat_in_channel))
        gat_outs = self.meta_gat(meta_knowledge, gat_inputs, data.edge_index)

        gat_outs = F.relu(torch.reshape(gat_outs, (-1, data.node_num, self.task_args['his_num'], self.model_args['hidden_dim'])))

        output_1 = self.tcn_1(meta_knowledge, gat_outs)
        output_2 = self.tcn_2(meta_knowledge, gat_outs)
        output = output_1 + output_2

        output = torch.reshape(output, (batch_size, node_num, seq_len))
        output = self.predictor(output)
        return output, meta_graph_structure
