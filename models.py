import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, GATv2Conv
import torch.nn.functional as F
from torch_scatter import scatter_mean

# TODO:
class GATModel(nn.Module):
    def __init__(self, 
                 num_layers,
                 heads,
                 input_dim,
                 hidden_dim,
                 edge_dim,
                 dropout):
        super().__init__()
        self.gat_layers = nn.ModuleList()
        
        gat = GATConv(in_channels=input_dim, 
                      out_channels=hidden_dim[0],
                      heads=heads[0],
                      dropout=dropout,
                      edge_dim=edge_dim)
        self.gat_layers.append(gat)
        
        for i in range(1, num_layers):
            gat = GATConv(in_channels=hidden_dim[i-1]*heads[i-1],
                          out_channels=hidden_dim[i],
                          heads=heads[i],
                          dropout=dropout,
                          edge_dim=edge_dim)
            self.gat_layers.append(gat)
            
        final_out_dim = heads[-1] * hidden_dim[-1]
        self.dense = nn.Sequential(nn.Linear(final_out_dim, 2 * final_out_dim),
                                   nn.ReLU(),
                                   nn.Dropout(dropout),
                                   nn.Linear(2 * final_out_dim, 2))
    
    
    def forward(self, x, edge_index, edge_attr, batch=None):
        for layer in self.gat_layers:
            x = layer(x=x, edge_index=edge_index, edge_attr=edge_attr)
            x = F.relu(x)
            
        if batch is None:
            x = x.mean(dim=0, keepdims=True)
        else:
            x = scatter_mean(x, batch, dim=0)
            
        logits = self.dense(x)
        
        return logits
    
if __name__ == '__main__':
    num_layers = 3
    input_dim = 79
    hidden_dim = [128, 128, 128]
    heads = [8, 8, 8]
    edge_dim = 10
    dropout = 0.5
    model = GATModel(num_layers=num_layers,
                     heads=heads,
                     input_dim=input_dim,
                     hidden_dim=hidden_dim,
                     edge_dim=edge_dim,
                     dropout=dropout)
    print(model)