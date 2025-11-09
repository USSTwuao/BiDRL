import torch
import torch.nn.functional as F
import numpy as np
from torch import nn
import math


class SkipConnection(nn.Module):
    def __init__(self, module):
        super(SkipConnection, self).__init__()
        self.module = module

    def forward(self, input):
        return input + self.module(input)
    


class Normalization(nn.Module):
    def __init__(self, emb_dim, normalization='batch'):
        super(Normalization, self).__init__()

        normalizer_class = {
            'batch': nn.BatchNorm1d,
        }.get(normalization, None)

        self.normalizer = normalizer_class(emb_dim, affine=True)

    def init_Parameters(self):
        for name, param in self.named_Parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)
            print('stdv', stdv)

    def forward(self, input):

        if isinstance(self.normalizer, nn.BatchNorm1d):
            return self.normalizer(input.view(-1, input.size(-1))).view(*input.size())
        else:
            assert self.normalizer is None, "Unknown normalizer type"
            return input


class edge_and_node_embed(nn.Module):
    def __init__(
            self,
            edge_dim,
            node_dim,
            emb_dim
    ):
      super(edge_and_node_embed, self).__init__()
      self.edge_init_embed = nn.Linear(edge_dim, 16) if edge_dim is not None else None
      self.node_init_embed = nn.Linear(node_dim, emb_dim) if node_dim is not None else None
    
    def forward(self , x):
        x1=x[:, :, :2] 
        guodu_matrix=(x1[:, :, None, :] - x1[:, None, :, :]).norm(p=2, dim=-1) 
        adj = (guodu_matrix != 0).int()  
        dis_matrix=guodu_matrix.unsqueeze(-1) 
        dis_matrix = dis_matrix.float()
        edge_fea = self.edge_init_embed(dis_matrix) 
        node_fea = self.node_init_embed(x.float())
        return node_fea,edge_fea,adj
    


class Graph_Attention(nn.Module):
    def __init__(
            self,
            batch_size,
            emb_dim,
            dropout,
            alpha,
            concat=True,
    ):
        super(Graph_Attention, self).__init__()

        self.batch_size=batch_size
        self.emb_dim=emb_dim
        self.dropout=dropout
        self.alpha=alpha
        self.concat=concat

        self.w = nn.Parameter(torch.zeros(size=(emb_dim,emb_dim)))
        nn.init.kaiming_normal_(self.w.data, a=0, mode='fan_in', nonlinearity='relu')
        self.a = nn.Parameter(torch.zeros(size=(3*emb_dim,1)))
        nn.init.kaiming_normal_(self.a.data, a=0, mode='fan_in', nonlinearity='relu')
        self.b = nn.Parameter(torch.zeros(size=(16,emb_dim)))
        nn.init.kaiming_normal_(self.b.data, a=0, mode='fan_in', nonlinearity='relu')

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self ,node_fea ,edge_fea,adj):


        h = torch.matmul(node_fea,self.w) 
        N=h.size()[1]
        eij=torch.matmul(edge_fea,self.b) 
        a_input1 = torch.cat([h.repeat(1,1,N).view(self.batch_size,N*N,self.emb_dim),h.repeat(1,N,1)],dim=2).view(self.batch_size,N,N,2*self.emb_dim)
        a_input2 = torch.cat((a_input1 , eij),dim=3)
        e=self.leakyrelu(torch.matmul(a_input2,self.a).squeeze(-1))

        zero_vec = -1e12*torch.ones_like(e)
        attention = torch.where(adj>0 , e,zero_vec)
        
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention ,self.dropout ,training = self.training)
        h_prime = torch.matmul(attention ,h) 

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime
        
    def _repr_(self):
        return self.__class__.__name__+'(' + str(self.in_features) +'->'+ str(self.out_features)+')'
    


class Multi_graphattention_module(nn.Module):
    def __init__(self,batch_size,emb_dim, dropout, alpha, n_heads): 
        super(Multi_graphattention_module,self).__init__()
        self.dropout = dropout
        self.emb_dim=emb_dim
        self.attentions = nn.ModuleList([Graph_Attention(batch_size,emb_dim, dropout=dropout, alpha=alpha, concat=True) for _ in range(n_heads)])
        self.w1 = nn.Parameter(torch.zeros(size=(8*emb_dim, emb_dim)))
        nn.init.kaiming_normal_(self.w1.data, a=0, mode='fan_in', nonlinearity='relu')
        

    def forward(self ,x,y, adj):  
        z = x.clone()
        x = F.dropout(x, self.dropout ,training=self.training)
        y = F.dropout(y, self.dropout ,training=self.training) 
        x = torch.cat([att(x,y,adj) for att in self.attentions],dim=2) ，
        x = torch.matmul(x,self.w1)
        x = x + z
        x = F.dropout(x , self.dropout,training=self.training)
    
        return x
    


class MultiHead_Graph_AttentionLayer(nn.Module):
    def __init__(
            self,
            batch_size,
            n_heads,
            emb_dim,
            dropout,
            alpha,
            feed_forward_hidden=512,
            normalization='batch',
    ):
        super(MultiHead_Graph_AttentionLayer, self).__init__()
        self.model1=Multi_graphattention_module(batch_size,
                                            emb_dim, 
                                            dropout, 
                                            alpha, 
                                            n_heads
                )
        self.Layers=nn.Sequential(Normalization(emb_dim,normalization),
        SkipConnection(
            (nn.Sequential(
                nn.Linear(emb_dim, feed_forward_hidden),
                nn.ReLU(),
                nn.Linear(feed_forward_hidden, emb_dim))
            ) if feed_forward_hidden > 0 else nn.Linear(emb_dim, emb_dim)
            ),
            Normalization(emb_dim,normalization),
        )
    def forward(self,x,y,adj):
        out1=self.model1(x,y,adj)
        out=self.Layers(out1)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(
            self,
            n_heads,
            input_dim,
            emb_dim,
    ):   
        super(MultiHeadAttention, self).__init__()
        val_dim = emb_dim // n_heads
        key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = emb_dim
        self.embed_dim = emb_dim
        self.val_dim = val_dim
        self.key_dim = key_dim

        self.norm_factor = 1 / math.sqrt(key_dim)

        self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        nn.init.kaiming_normal_(self.W_query.data, a=0, mode='fan_in', nonlinearity='relu')
        self.W_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        nn.init.kaiming_normal_(self.W_key.data, a=0, mode='fan_in', nonlinearity='relu')
        self.W_val = nn.Parameter(torch.Tensor(n_heads, input_dim, val_dim))
        nn.init.kaiming_normal_(self.W_val.data, a=0, mode='fan_in', nonlinearity='relu')
        self.W_out = nn.Parameter(torch.Tensor(n_heads, key_dim, emb_dim))
        nn.init.kaiming_normal_(self.W_out.data, a=0, mode='fan_in', nonlinearity='relu')

    def forward(self , MGAL_out):
        MGAL_out_flat = MGAL_out.contiguous().view(-1,self.input_dim) 
        batch_size = MGAL_out.size(0)
        N = MGAL_out.size(1)
        shape = (self.n_heads , batch_size ,N ,-1)
        Q = torch.matmul(MGAL_out_flat,self.W_query) 
        Q = Q.view(shape) 
        K = torch.matmul(MGAL_out_flat,self.W_key) 
        K = K.view(shape)
        V = torch.matmul(MGAL_out_flat,self.W_val) 
        V = V.view(shape)

        compatibility = self.norm_factor * torch.matmul(Q, K.transpose(2, 3)) 
        indices = torch.arange(N)
        compatibility[:, :, indices, indices] = -1e12
        attn = F.softmax(compatibility, dim=-1)

        heads = torch.matmul(attn, V)
        
        out = torch.mm(
        heads.permute(1, 2, 0, 3).contiguous().view(-1, self.n_heads * self.val_dim),
        self.W_out.view(-1, self.embed_dim)
        ).view(batch_size, N, self.embed_dim)

        return out
    

class MultiHeadAttention_Layer(nn.Sequential):
    def __init__(
            self,
            n_heads,
            emb_dim,
            feed_forward_hidden=512,
            normalization='batch',
    ):
        super(MultiHeadAttention_Layer, self).__init__(
            SkipConnection(
                MultiHeadAttention(n_heads,
                                   emb_dim, 
                                   emb_dim
                )
        ),
        Normalization(emb_dim,normalization),
        SkipConnection(
            nn.Sequential(
                nn.Linear(emb_dim, feed_forward_hidden),
                nn.ReLU(),
                nn.Linear(feed_forward_hidden, emb_dim)
            ) if feed_forward_hidden > 0 else nn.Linear(emb_dim, emb_dim)
            ),
            Normalization(emb_dim,normalization),
        )


class GraphAttentionEncoder(nn.Module):
    def __init__(
            self,
            edge_dim,
            batch_size,
            dropout,
            alpha,
            node_dim,
            n_heads, #8
            emb_dim, #128
            normalization='batch',
            feed_forward_hidden=512,
    ):
        
        super(GraphAttentionEncoder, self).__init__()
        self.edge_dim = edge_dim
        self.node_dim = node_dim
        self.emb_dim = emb_dim

        self.layers_a =MultiHeadAttention_Layer(n_heads,
                                    emb_dim,
                                    feed_forward_hidden,
                                    normalization)
        
        self.layers_b = MultiHeadAttention_Layer(n_heads,
                                    emb_dim,
                                    feed_forward_hidden,
                                    normalization)
        self.layers_c = MultiHeadAttention_Layer(n_heads,
                                    emb_dim,
                                    feed_forward_hidden,
                                    normalization)

        self.model1=edge_and_node_embed(self.edge_dim,self.node_dim,self.emb_dim)
        self.model2=MultiHead_Graph_AttentionLayer(batch_size,
                                            n_heads,
                                            emb_dim,
                                            dropout,
                                            alpha,
                                            feed_forward_hidden,
                                            normalization)
    def forward(self,shuru):
        x,y,adj = self.model1(shuru)
        out1 = self.layers_a(x)
        out2 = self.layers_b(out1)
        out3 = self.layers_c(out2)
        out4 = out3.mean(dim=1)
        return out3,out4
        




        





        


        




        