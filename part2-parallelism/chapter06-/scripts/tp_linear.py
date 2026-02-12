import torch
import torch.distributed as dist
import torch.nn as nn


class ColumnParallelLinear(nn.Module):
    """
    Split the weight matrix W by columns
    W_full  shape: [in_features, out_features]
    W_local shape: [in_features, out_features/tp_size]

    Forward: Y_local = X @ W_local
    No communication needed in forward pass
    """

    def forward(self, X):
        # each GPU computes its own portion of the output
        return X @ self.weight



class RowParallelLinear(nn.Module):
    """
    Split the weight matrix W by rows

    W_full shape: [in_features, out_features]
    W_local shape: [in_features/tp_size, out_features]

    Forward: Y_partial = X_local @ W_local
    Y = all_reduce(Y_partial)
    """

    def forward(self, X_local):
        # Each GPU has part of the input, computes partial output
        Y_partial = X_local @ self.weight
        # sum across GPUs
        dist.all_reduce(Y_partial, op=dist.ReduceOp.SUM)
        return Y_partial


# Combining Column + Row: The MLP Recipe
def mlp_tp_forward(X, W1_col, W2_row, tp_group):
    """
    Tensor-parallel MLP with minimal communication
    W1 is column-parallel [hidden_size, 4*hidden_size//tp_size]
    W2 is row-parallel [4*hidden_size//tp_size, hidden_size]
    """

    # Step 1: Column parallel first linear
    hidden = torch.relu(X @ W1_col) # no communication needed

    # Step 2: Row parallel second linear
    output = hidden @ W2_row

    # Step 3: only 1 all_reduce across tp_group
    dist.all_reduce(output, op=dist.ReduceOp.SUM, group=tp_group)
    return output






