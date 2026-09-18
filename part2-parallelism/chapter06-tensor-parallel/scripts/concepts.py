import torch
import torch.distributed as dist
import torch.nn as nn

class ColumnParallelLinear(nn.Module):
    """
    Split the weight matrix W by columns.

    W_full shape: [in_features, out_features]
    W_local shape: [in_features, out_features // tp_size]

    Forward: Y_local = X @ W_local
    No communication needed in forward!
    """
    def forward(self, X):
        # Each GPU computes the location portion of the output
        return X @ self.weight # shape: [batch_size, out_features // tp_size]


class RowParallelLinear(nn.Module):
    """
    Split the weight matrix W by rows.

    W_full shape: [in_features, out_features]
    W_local shape: [in_features // tp_size, out_features]
    Forward: Y_partial = X_local @ W_local
             Y = all_reduce(Y_partial)
    """
    def forward(self, X_local):
        # Each GPU has part of input, computes partial output
        Y_partial = X_local @ self.weight
        # Sum across all gpus
        dist.all_reduce(Y_partial, op=dist.ReduceOp.SUM)
        return Y_partial



def tp_mlp_forward(X, W1_col, W2_row, tp_group):
    """
    Tensor-parallel MLP with minimal communication.

    W1 is column-parallel: [hidden, 4*hidden//tp_size]
    W2 is row-parallel: [4*hidden//tp_size, hidden]
    """

    # Step 1: Column-parallel first linear
    hidden = torch.relu(X @ W1_col)

    # Step 2: Row-parallel second linear
    output = hidden @ W2_row
    dist.all_reduce(output, op=dist.ReduceOp.SUM, group=tp_group)
    return output





"""TP vs DP: When to Use Which?

Factor	         Data Parallel	         Tensor Parallel
Granularity	      Whole model	          Single layer
Communication	Gradients only	      Activations every layer
Scalability    	100s of GPUs	         Usually ≤8 GPUs
Best for	     Batch scaling	           Large layers
Topology	     Cross-node OK	       Intra-node (NVLink)
"""



