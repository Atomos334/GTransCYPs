import torch
import torch.nn.functional as F 
from torch.nn import Linear, BatchNorm1d, ModuleList
from torch_geometric.nn import TransformerConv, global_add_pool, GlobalAttention
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.nn import TransformerConv, TopKPooling
from torch_geometric.nn import GINConv, global_mean_pool, global_max_pool
from torch_geometric.nn import SAGEConv, global_mean_pool, global_max_pool

class GTransCYPs(torch.nn.Module):
    def __init__(self, feature_size, model_params):
        super(GTransCYPs, self).__init__()
        embedding_size = model_params["model_embedding_size"]
        n_heads = model_params["model_attention_heads"]
        self.n_layers = model_params["model_layers"]
        dropout_rate = model_params["model_dropout_rate"]
        dense_neurons = model_params["model_dense_neurons"]
        edge_dim = model_params["model_edge_dim"]
        self.conv_layers = ModuleList()
        self.transf_layers = ModuleList()
        self.bn_layers = ModuleList()
        self.conv1 = TransformerConv(feature_size, 
                                    embedding_size, 
                                    heads=n_heads, 
                                    dropout=dropout_rate,
                                    edge_dim=edge_dim,
                                    beta=True) 
        self.transf1 = Linear(embedding_size*n_heads, embedding_size)
        self.bn1 = BatchNorm1d(embedding_size)

        for _ in range(self.n_layers):
            self.conv_layers.append(TransformerConv(embedding_size, 
                                                    embedding_size, 
                                                    heads=n_heads, 
                                                    dropout=dropout_rate,
                                                    edge_dim=edge_dim,
                                                    beta=True))
            self.transf_layers.append(Linear(embedding_size*n_heads, embedding_size))
            self.bn_layers.append(BatchNorm1d(embedding_size))
        self.attention_pool = GlobalAttention(gate_nn=Linear(embedding_size, 1))
        self.linears = ModuleList([Linear(embedding_size, dense_neurons),
                                   Linear(dense_neurons, int(dense_neurons/2)),
                                   Linear(int(dense_neurons/2), 1)])
    def forward(self, x, edge_attr, edge_index, batch_index):
        x = self.conv1(x, edge_index, edge_attr)
        x = torch.relu(self.transf1(x))
        x = self.bn1(x)
        local_representation = []
        for i in range(self.n_layers):
            x = self.conv_layers[i](x, edge_index, edge_attr)
            x = torch.relu(self.transf_layers[i](x))
            x = self.bn_layers[i](x)
            local_representation.append(x) 
        x = sum(local_representation)
        x = self.attention_pool(x, batch_index)

        for linear in self.linears[:-1]:
            x = torch.relu(linear(x))
            x = F.dropout(x, p=0.8, training=self.training)
        x = self.linears[-1](x)

        return x

class GIN(torch.nn.Module):
    def __init__(self, feature_size, model_params):
        super(GIN, self).__init__()
        embedding_size = model_params["model_embedding_size"]
        n_heads = model_params["model_attention_heads"]
        self.n_layers = model_params["model_layers"]
        dropout_rate = model_params["model_dropout_rate"]
        dense_neurons = model_params["model_dense_neurons"]
        edge_dim = model_params["model_edge_dim"]
        self.conv_layers = ModuleList([GINConv(Linear(feature_size, embedding_size))])

        for _ in range(self.n_layers - 1):
            self.conv_layers.append(GINConv(Linear(embedding_size, embedding_size)))
        self.linear1 = Linear(embedding_size * 2, dense_neurons)
        self.linear2 = Linear(dense_neurons, int(dense_neurons / 2))
        self.linear3 = Linear(int(dense_neurons / 2), 1)

    def forward(self, x, edge_attr, edge_index, batch_index):
        for i in range(self.n_layers):
            x = self.conv_layers[i](x, edge_index)
            x = F.relu(x)
        x = torch.cat([global_max_pool(x, batch_index), global_mean_pool(x, batch_index)], dim=1)
        x = torch.relu(self.linear1(x))
        x = F.dropout(x, p=0.8, training=self.training)
        x = torch.relu(self.linear2(x))
        x = F.dropout(x, p=0.8, training=self.training)
        x = self.linear3(x)

        return x
    
class GraphSAGE(torch.nn.Module):
    def __init__(self, feature_size, model_params):
        super(GraphSAGE, self).__init__()
        embedding_size = model_params["model_embedding_size"]
        dense_neurons = model_params["model_dense_neurons"]
        self.n_layers = model_params["model_layers"]
        dropout_rate = model_params["model_dropout_rate"]
        self.conv_layers = ModuleList([SAGEConv(feature_size, embedding_size)] +
                                      [SAGEConv(embedding_size, embedding_size) for _ in range(self.n_layers - 1)])
        self.bn_layers = ModuleList([BatchNorm1d(embedding_size) for _ in range(self.n_layers)])
        self.linear1 = Linear(embedding_size * 2, dense_neurons)
        self.linear2 = Linear(dense_neurons, int(dense_neurons / 2))
        self.linear3 = Linear(int(dense_neurons / 2), 1)

    def forward(self, x, edge_attr, edge_index, batch_index): 
        for conv_layer, bn_layer in zip(self.conv_layers, self.bn_layers):
            x = F.relu(bn_layer(conv_layer(x, edge_index)))

        x = torch.cat([global_max_pool(x, batch_index), global_mean_pool(x, batch_index)], dim=1)
        x = torch.relu(self.linear1(x))
        x = F.dropout(x, p=0.8, training=self.training)
        x = torch.relu(self.linear2(x))
        x = F.dropout(x, p=0.8, training=self.training)
        x = self.linear3(x)

        return x
