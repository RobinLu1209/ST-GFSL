import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch import Tensor
from torch.autograd import Variable
import sys
import yaml
import scipy.sparse as sp
import torch
from scipy.sparse import linalg
from torch_geometric.nn import GATConv

class MetaConv2d_update(nn.Module):
    def __init__(self, meta_dim, in_channels, out_channels, kernel_size=3, dilation=1):
        super(MetaConv2d_update, self).__init__()
        self.meta_dim = meta_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
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
        input = input.permute(0,2,3,1)
        # print("input shape is", input.shape)
        # view batch conv2d as group conv2d
        batch_size, node_num, seq_len_in, _ = input.shape
        # print("input shape is", input.shape)
        input = torch.reshape(input, (batch_size*node_num, seq_len_in, self.in_channels)).permute(0, 2, 1)
        input = input.unsqueeze(2)
        group_input = torch.reshape(input, (-1, 1, seq_len_in)).unsqueeze(0)
        # print("group_input shape is", group_input.shape)
        group_output = F.conv2d(group_input, weight=w_meta, bias=b_meta, groups=batch_size*node_num, dilation=self.dilation)
        # print("group_output shape is", group_output.shape)
        group_output = torch.reshape(group_output, (1, batch_size, node_num, self.out_channels, -1)).squeeze(0)
        group_output = group_output.permute(0, 2, 1, 3)
        return group_output

class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncvl,vw->ncwl',(x,A))
        return x.contiguous()

class linear(nn.Module):
    def __init__(self,c_in,c_out):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)

    def forward(self,x):
        return self.mlp(x)

class meta_linear(nn.Module):
    def __init__(self,meta_dim,c_in,c_out):
        super(meta_linear,self).__init__()
        self.mlp = MetaConv2d_update(meta_dim,c_in,c_out,1)
        # self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)

    def forward(self,meta_knowledge, x):
        return self.mlp(meta_knowledge, x)

class meta_gcn(nn.Module):
    def __init__(self,meta_dim,c_in,c_out,dropout,support_len=3,order=2):
        super(meta_gcn,self).__init__()
        self.nconv = nconv()
        c_in = (order*support_len+1)*c_in
        self.mlp = meta_linear(meta_dim,c_in,c_out)
        self.dropout = dropout
        self.order = order

    def forward(self,meta_knowledge,x,support):
        out = [x]
        for a in support:
            print("a shape is", a.shape)
            x1 = self.nconv(x,a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1,a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out,dim=1)
        print("[mlp] input shape is", h.shape)
        h = self.mlp(meta_knowledge,h)
        print("[mlp] output shape is", h.shape)
        h = F.dropout(h, self.dropout, training=self.training)
        return h

class gcn(nn.Module):
    def __init__(self,c_in,c_out,dropout,support_len=3,order=2):
        super(gcn,self).__init__()
        self.nconv = nconv()
        c_in = (order*support_len+1)*c_in
        self.mlp = linear(c_in,c_out)
        self.dropout = dropout
        self.order = order

    def forward(self,x,support):
        out = [x]
        for a in support:
            x1 = self.nconv(x,a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1,a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out,dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h

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
        input = input.permute(0,2,3,1)
        batch_size, node_num, seq_len_in, _ = input.shape
        input = torch.reshape(input, (batch_size*node_num, seq_len_in, self.in_channels)).permute(0, 2, 1)
        group_input = torch.reshape(input, (-1, seq_len_in)).unsqueeze(0)

        # print("group_input shape is", group_input.shape, " device:", group_input.device)

        group_output = F.conv1d(group_input, weight=w_meta, bias=b_meta, groups=batch_size*node_num, stride=self.stride, padding=self.padding, dilation=self.dilation)
        group_output = torch.reshape(group_output, (1, batch_size, node_num, self.out_channels, -1)).squeeze(0)
        group_output = group_output.permute(0,2,1,3)
        return group_output

class meta_gwnet(nn.Module):
    def __init__(self, meta_dim, dropout=0.3, gcn_bool=True, in_dim=2,out_dim=6,residual_channels=32,dilation_channels=32,skip_channels=256,end_channels=512,kernel_size=2,blocks=4,layers=2):
        super(meta_gwnet, self).__init__()
        self.meta_dim = meta_dim
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.gcn_bool = gcn_bool

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()

        self.start_conv = MetaConv2d_update(meta_dim, in_dim, residual_channels, 1)
        # self.start_conv = nn.Conv2d(in_channels=in_dim,
        #                             out_channels=residual_channels,
        #                             kernel_size=(1,1))
        receptive_field = 1

        # All supports are double transition
        self.supports_len = 2

        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # dilated convolutions
                self.filter_convs.append(MetaConv2d_update(meta_dim, residual_channels, dilation_channels, kernel_size, new_dilation))

                self.gate_convs.append(MetaConv2d_update(meta_dim, residual_channels, dilation_channels, kernel_size, new_dilation))

                # 1x1 convolution for residual connection
                self.residual_convs.append(MetaConv1d(meta_dim, dilation_channels, residual_channels,1))

                # 1x1 convolution for skip connection
                self.skip_convs.append(MetaConv1d(meta_dim, dilation_channels, skip_channels, 1))

                self.bn.append(nn.BatchNorm2d(residual_channels))
                new_dilation *=2
                receptive_field += additional_scope
                additional_scope *= 2
                if self.gcn_bool:
                    self.gconv.append(meta_gcn(meta_dim, dilation_channels,residual_channels,dropout,support_len=self.supports_len))



        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                  out_channels=end_channels,
                                  kernel_size=(1,1),
                                  bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1,1),
                                    bias=True)

        self.receptive_field = receptive_field

    def forward(self, meta_knowledge, input, supports):

        input = input.permute(0,3,1,2)

        in_len = input.size(3)
        if in_len<self.receptive_field:
            x = nn.functional.pad(input,(self.receptive_field-in_len,0,0,0))
        else:
            x = input
        # print("[start conv] x shape is", x.shape)
        x = self.start_conv(meta_knowledge, x)
        # print("[after start conv] x shape is", x.shape)
        skip = 0       

        # WaveNet layers
        for i in range(self.blocks * self.layers):

            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|                |
            # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
            #                 |-- conv -- sigm --|     |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + ------------->	*skip*

            #(dilation, init_dilation) = self.dilations[i]

            #residual = dilation_func(x, dilation, init_dilation, i)
            residual = x
            # dilated convolution
            filter = self.filter_convs[i](meta_knowledge, residual)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](meta_knowledge, residual)
            gate = torch.sigmoid(gate)
            x = filter * gate

            # parametrized skip connection

            s = x
            s = self.skip_convs[i](meta_knowledge, s)
            try:
                skip = skip[:, :, :,  -s.size(3):]
            except:
                skip = 0
            skip = s + skip


            if self.gcn_bool and supports is not None:
                x = self.gconv[i](meta_knowledge, x,supports)
            else:
                x = self.residual_convs[i](meta_knowledge, x)

            x = x + residual[:, :, :, -x.size(3):]


            x = self.bn[i](x)

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        x = (x.squeeze(-1)).permute(0,2,1)
        return x

class gwnet(nn.Module):
    def __init__(self, dropout=0.3, gcn_bool=True, in_dim=2,out_dim=6,residual_channels=32,dilation_channels=32,skip_channels=256,end_channels=512,kernel_size=2,blocks=4,layers=2):
        super(gwnet, self).__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.gcn_bool = gcn_bool

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()

        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1,1))
        receptive_field = 1

        # All supports are double transition
        self.supports_len = 2

        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # dilated convolutions
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1,kernel_size),dilation=new_dilation))

                self.gate_convs.append(nn.Conv1d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size), dilation=new_dilation))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1)))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(residual_channels))
                new_dilation *=2
                receptive_field += additional_scope
                additional_scope *= 2
                if self.gcn_bool:
                    self.gconv.append(gcn(dilation_channels,residual_channels,dropout,support_len=self.supports_len))



        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                  out_channels=end_channels,
                                  kernel_size=(1,1),
                                  bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1,1),
                                    bias=True)

        self.receptive_field = receptive_field

    def forward(self, input, supports):

        input = input.permute(0,3,1,2)

        in_len = input.size(3)
        if in_len<self.receptive_field:
            x = nn.functional.pad(input,(self.receptive_field-in_len,0,0,0))
        else:
            x = input
        x = self.start_conv(x)
        skip = 0       

        # WaveNet layers
        for i in range(self.blocks * self.layers):

            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|                |
            # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
            #                 |-- conv -- sigm --|     |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + ------------->	*skip*

            #(dilation, init_dilation) = self.dilations[i]

            #residual = dilation_func(x, dilation, init_dilation, i)
            residual = x
            # dilated convolution
            # print("Conv2d input shape is ", residual.shape)
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)
            # print("Conv1d input shape is ", residual.shape)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter * gate

            # parametrized skip connection

            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :,  -s.size(3):]
            except:
                skip = 0
            skip = s + skip


            if self.gcn_bool and supports is not None:
                x = self.gconv[i](x,supports)
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]


            x = self.bn[i](x)

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        x = (x.squeeze(-1)).permute(0,2,1)
        return x

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
            mk_input = None
            print("sp and tp are all False.")
        
        meta_knowledge = self.mk_learner(mk_input)
        meta_knowledge = torch.reshape(meta_knowledge, (batch_size, node_num, self.meta_out))
        return meta_knowledge


class MetaGWN(nn.Module):
    def __init__(self, model_args, task_args) -> None:
        super(MetaGWN, self).__init__()
        self.model_args = model_args
        self.task_args = task_args
        self.meta_dim = model_args['meta_dim']
        self.message_dim = model_args['message_dim']
        self.his_num = task_args['his_num']
        self.pred_num = task_args['pred_num']
        self.build()

    def build(self):
        self.meta_learner = STMetaLearner(self.model_args, self.task_args)
        self.meta_gwnet = meta_gwnet(self.meta_dim)
    
    def forward(self, data, A_wave):
        """
        here: A_wave is supports
        """
        meta_knowledge = self.meta_learner(data)
        meta_graph_structure = torch.bmm(meta_knowledge, meta_knowledge.permute(0,2,1))
        meta_graph_structure = F.relu(meta_graph_structure)
        output = self.meta_gwnet(meta_knowledge, data.x, A_wave)
        return output, meta_graph_structure

