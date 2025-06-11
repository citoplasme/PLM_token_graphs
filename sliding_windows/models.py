import torch
import torch_geometric

# ===========================================================================
# ============================= Model definition ============================
# ===========================================================================

# Based on https://mlabonne.github.io/blog/posts/2022-03-09-Graph_Attention_Network.html
class GAT(torch.nn.Module):
  def __init__(
    self,
    node_feature_count : int,
    class_count : int,
    hidden_dimension : int = 64,
    edge_dimension : int = None,
    attention_heads : int = 8,
    number_of_hidden_layers : int = 0,
    dropout_rate : float = 0.6,
    global_pooling : str = 'mean'
  ):
    super(GAT, self).__init__()

    self.convs = torch.nn.ModuleList()
    self.convs.append(torch_geometric.nn.GATConv(in_channels = node_feature_count, out_channels = hidden_dimension, edge_dim = edge_dimension, heads = attention_heads, add_self_loops = False))
    for _ in range(number_of_hidden_layers):
      self.convs.append(torch_geometric.nn.GATConv(in_channels = hidden_dimension * attention_heads, out_channels = hidden_dimension, edge_dim = edge_dimension, heads = attention_heads, add_self_loops = False))

    self.linear = torch.nn.Linear(hidden_dimension * attention_heads, class_count)
    self.dropout_rate = dropout_rate
    self.global_pooling = global_pooling

  def forward(self, x, edge_index, edge_attr, batch):
    # 1. Obtain node embeddings
    for conv in self.convs:
      x = torch.nn.functional.dropout(x, p = self.dropout_rate, training = self.training)
      x = conv(x = x, edge_index = edge_index, edge_attr = edge_attr)
      #x = torch.nn.functional.relu(x)
      x = torch.nn.functional.leaky_relu(x)

    # 2. Readout layer
    if self.global_pooling == 'sum':
      x = torch_geometric.nn.global_add_pool(x, batch)
    elif self.global_pooling == 'max':
      x = torch_geometric.nn.global_max_pool(x, batch)
    else:
      x = torch_geometric.nn.global_mean_pool(x, batch)

    # 3. Apply a final classifier
    x = self.linear(x)
    return torch.nn.functional.log_softmax(x, dim = 1)

# Based on https://mlabonne.github.io/blog/posts/2022-03-09-Graph_Attention_Network.html
class GATv2(torch.nn.Module):
  def __init__(
    self,
    node_feature_count : int,
    class_count : int,
    hidden_dimension : int = 64,
    edge_dimension : int = None,
    attention_heads : int = 8,
    number_of_hidden_layers : int = 0,
    dropout_rate : float = 0.6,
    global_pooling : str = 'mean'
  ):
    super(GATv2, self).__init__()

    self.convs = torch.nn.ModuleList()
    self.convs.append(torch_geometric.nn.GATv2Conv(in_channels = node_feature_count, out_channels = hidden_dimension, edge_dim = edge_dimension, heads = attention_heads, add_self_loops = False))

    for _ in range(number_of_hidden_layers):
      self.convs.append(torch_geometric.nn.GATv2Conv(in_channels = hidden_dimension * attention_heads, out_channels = hidden_dimension, edge_dim = edge_dimension, heads = attention_heads, add_self_loops = False))

    self.linear = torch.nn.Linear(hidden_dimension * attention_heads, class_count)
    self.dropout_rate = dropout_rate
    self.global_pooling = global_pooling

  def forward(self, x, edge_index, edge_attr, batch):
    # 1. Obtain node embeddings
    for conv in self.convs:
      x = torch.nn.functional.dropout(x, p = self.dropout_rate, training = self.training)
      x = conv(x = x, edge_index = edge_index, edge_attr = edge_attr)
      #x = torch.nn.functional.relu(x)
      x = torch.nn.functional.leaky_relu(x)

    # 2. Readout layer
    if self.global_pooling == 'sum':
      x = torch_geometric.nn.global_add_pool(x, batch)
    elif self.global_pooling == 'max':
      x = torch_geometric.nn.global_max_pool(x, batch)
    else:
      x = torch_geometric.nn.global_mean_pool(x, batch)

    # 3. Apply a final classifier
    x = self.linear(x)
    return torch.nn.functional.log_softmax(x, dim = 1)

class GCN(torch.nn.Module):
  def __init__(
    self,
    node_feature_count : int,
    class_count : int,
    hidden_dimension : int = 64,
    number_of_hidden_layers : int = 0,
    dropout_rate : float = 0.6,
    global_pooling : str = 'mean'
  ):
    super(GCN, self).__init__()
    
    self.convs = torch.nn.ModuleList()
    self.convs.append(torch_geometric.nn.GCNConv(in_channels = node_feature_count, out_channels = hidden_dimension, add_self_loops = False))
    for _ in range(number_of_hidden_layers):
      self.convs.append(torch_geometric.nn.GCNConv(in_channels = hidden_dimension, out_channels = hidden_dimension, add_self_loops = False))
    self.linear = torch.nn.Linear(hidden_dimension, class_count)
    self.dropout_rate = dropout_rate
    self.global_pooling = global_pooling

  def forward(self, x, edge_index, edge_weight, batch):
    # 1. Obtain node embeddings 
    for conv in self.convs:
      x = torch.nn.functional.dropout(x, p = self.dropout_rate, training = self.training)
      x = conv(x = x, edge_index = edge_index, edge_weight = edge_weight)
      #x = torch.nn.functional.relu(x)
      x = torch.nn.functional.leaky_relu(x)

    # 2. Readout layer
    if self.global_pooling == 'sum':
      x = torch_geometric.nn.global_add_pool(x, batch)
    elif self.global_pooling == 'max':
      x = torch_geometric.nn.global_max_pool(x, batch)
    else:
      x = torch_geometric.nn.global_mean_pool(x, batch)

    # 3. Apply a final classifier
    x = self.linear(x)
    return torch.nn.functional.log_softmax(x, dim = 1)
