"""
Implementation of GNAN from the paper:
"Group-Node Attention for Community Evolution Prediction"

"""

import sys
import os.path
import time
import math
from datetime import datetime
import torch as t
import torch.nn.functional as tfunc
from torch.nn.parameter import Parameter
from torch.nn import ParameterList
from torch.utils.data import random_split, TensorDataset, ConcatDataset, DataLoader

from comm_dyn.data import calc_pos_weight, load_train_and_valid

class MultiheadAtt(t.nn.Module):
    """A multihead attention wrapper.
    """
    def __init__(self,
                 num_heads: int,
                 model_size: int,
                 group_size: int,
                 key_size: int,
                 value_size: int) -> None:
        super(MultiheadAtt, self).__init__()
        self.num_heads = num_heads
        self.model_size = model_size
        self.group_size = group_size
        self.key_size = key_size
        self.value_size = value_size

        self.Wq = []
        self.Wk = []
        self.Wv = []

        for _i in range(num_heads):
            self.Wq.append(Parameter(t.randn(group_size, key_size)))
            self.Wk.append(Parameter(t.randn(model_size, key_size)))
            self.Wv.append(Parameter(t.randn(model_size, value_size)))

        self.Wq = ParameterList(self.Wq)
        self.Wk = ParameterList(self.Wk)
        self.Wv = ParameterList(self.Wv)
        self.Wo = Parameter(t.randn(num_heads * value_size, model_size))

    def forward(self, group_feats, nodes_feats, training: bool = True):
        head_outputs = []

        for i in range(self.num_heads):
            Q = t.matmul(group_feats, self.Wq[i])
            K = t.matmul(nodes_feats, self.Wk[i])
            V = t.matmul(nodes_feats, self.Wv[i])
            # Note that the coefficients end up being a vector and thus have only 1 axis. Because of this,
            # we apply softmax to the first axis at shape/axis index 0.
            att_coefs = tfunc.softmax(t.mul(t.matmul(Q, t.transpose(K, 0, 1)),
                                            t.reciprocal(t.sqrt(t.tensor([float(self.key_size)])))),
                                      0)
            output = t.relu(t.matmul(att_coefs, V))
            head_outputs.append(output)

        return t.matmul(t.cat(head_outputs), self.Wo)

class MultiheadLinear(t.nn.Module):
    def __init__(self,
                 d_in: int,
                 head_size: int,
                 num_heads: int):
        super(MultiheadLinear, self).__init__()
        self.d_in = d_in
        self.head_size = head_size
        self.num_heads = num_heads
        self.d_out = head_size * num_heads

        self.heads = []
        for i_ in range(num_heads):
            self.heads.append(t.nn.Linear(d_in, head_size))

    def forward(self, X, training: bool = True):
        head_outputs = []
        for i in range(self.num_heads):
            head_outputs.append(tfunc.dropout(tfunc.relu(self.heads[i](X)),
                                              p=0.15, training=training))

        return t.cat(head_outputs, 1)

class GNAN(t.nn.Module):
    """Group-Node Attention for Community Evolution Prediction.
    """
    def __init__(self,
                 d_in: int,
                 model_size: int,
                 num_group_feats: int,
                 d_out: int,
                 num_member_layers: int = 1) -> None:
        super(GNAN, self).__init__()

        self.model_size = model_size

        # Add a dimension for node position
        self.linear_in = t.nn.Linear(d_in + 1, model_size)
        self.linear_group_feats = t.nn.Linear(num_group_feats, model_size)

        num_heads = 4
        query_size = num_group_feats
        key_size = model_size
        value_size = model_size

        self.linear_query = t.nn.Linear(num_group_feats, query_size)

        self.group_att = MultiheadAtt(num_heads,
                                      model_size,
                                      query_size,
                                      key_size,
                                      value_size)

        # Accept the concatenation of the attention layer output with the group features
        self.linear_out = t.nn.Linear(model_size + model_size, d_out)

    def forward(self, X):
        """Process a single group.
        X is a tuple with two 2d tensors of group member features
        and group features.
        """
        (X_m, X_dnbor, x_g) = X

        # Drop unused features
        x_g = x_g[2:]

        X_m_ = X_m.transpose(0, 1)
        X_m_prev_degs = X_m_[3:5]
        X_m_curr_degs = X_m_[8:]
        X_m = t.cat((X_m_prev_degs, X_m_curr_degs)).transpose(0, 1)

        X_dnbor_ = X_dnbor.transpose(0, 1)
        X_dnbor_prev_degs = X_dnbor_[3:5]
        X_dnbor_curr_degs = X_dnbor_[8:]
        X_dnbor = t.cat((X_dnbor_prev_degs, X_dnbor_curr_degs)).transpose(0, 1)

        num_members = X_m.shape[0]
        num_dnbors = X_dnbor.shape[0]

        # Add positional information
        X_m_ = t.cat((X_m, t.zeros(num_members, 1).fill_(0.0)), 1)
        X_dnbor_ = t.cat((X_dnbor, t.zeros(num_dnbors, 1).fill_(1.0)), 1)

        X_ = t.cat((X_m_, X_dnbor_), 0)

        # Reshape input to match model size
        Z = tfunc.dropout(tfunc.relu(self.linear_in(X_)),
                          p=0.15, training=self.training)

        # Pass group features through a dense layer
        z_g = tfunc.dropout(t.relu(self.linear_group_feats(x_g)),
                            p=0.15, training=self.training)

        z_q = tfunc.dropout(t.relu(self.linear_query(x_g)),
                            p=0.15, training=self.training)

        z = self.group_att(z_q, Z, training=self.training)

        # Concatenate context vector with group features
        z_ = t.cat((z, z_g))

        # Reshape hidden vector for group
        _y_pred = self.linear_out(z_)

        # We use the direct output from the final Linear layer because
        # we are using a loss function that will internally use a Sigmoid.
        y_pred = _y_pred

        return y_pred

def make_train_step(model, optimizer):

    def train_step(criterion, x, y):
        # Set model to train mode
        model.train()

        # Make prediction
        y_pred = model(x)

        # Compute loss
        loss = criterion(y_pred, y)

        return loss

    return train_step

def load_model(path, input_dim, hidden_dim, num_group_feats, output_dim):
    model = GNAN(input_dim, hidden_dim, num_group_feats, output_dim)
    checkpoint = t.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

orig_node_dim = 2
# Concatenate features for a node over two snapshots
num_snapshots = 2
input_dim = orig_node_dim * num_snapshots
hidden_dim = 16
num_group_feats = 7
output_dim = 6

def main(argv=None):
    if not argv:
        argv = sys.argv

    dataset_path = argv[1]
    model_path = argv[2]

    # Make directory for experiment
    os.makedirs(model_path, exist_ok=True)

    max_epochs = 1000
    patience = 5

    train_dataset, valid_dataset = load_train_and_valid(dataset_path)
    train_size = len(train_dataset)

    all_dataset = ConcatDataset([train_dataset, valid_dataset])

    batch_size = 32

    # Use a data loader in order to shuffle the dataset for each epoch.
    # We manage batching on our own; mini-batches are only used for updating
    # model parameters and not for parallelization.
    train_dataloader = DataLoader(train_dataset, batch_size=None, shuffle=True)

    model = GNAN(input_dim, hidden_dim, num_group_feats, output_dim)
    params = model.parameters()
    optimizer = t.optim.AdamW(params, lr=1e-3, weight_decay=1e-2)

    train_step = make_train_step(model, optimizer)

    best_total_loss = math.inf
    best_epoch = None
    last_total_loss = math.nan

    # Use label counts to weight loss in criterion and account for class imbalance.
    tv_labels = [y for (x, y) in all_dataset]
    criterion = t.nn.BCEWithLogitsLoss(pos_weight=calc_pos_weight(tv_labels),
                                       reduction='none')

    for epoch in range(max_epochs):
        start_time = time.time()
        print(f'Starting epoch {epoch}...')
        losses = []
        for (iter_count, (train_x, train_y)) in enumerate(train_dataloader):
            # Perform a single train step
            loss = train_step(criterion, train_x, train_y)
            mean_loss = t.mean(loss)
            losses.append(mean_loss)
            # Update model parameters after processing a batch_size worth of training instances
            if (iter_count % batch_size) == (batch_size - 1):
                batch_loss = sum(losses)
                batch_loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                losses = []

        # Calculate total loss from validation dataset
        total_loss = 0
        model.eval()
        for (valid_x, valid_y) in valid_dataset:
            with t.no_grad():
                y_pred = model(valid_x)
                loss = criterion(y_pred, valid_y)
                mean_loss = t.mean(loss)
                total_loss += mean_loss

        end_time = time.time()
        total_time = int(end_time - start_time)
        print(f'Epoch: {epoch}, Num. examples: {train_size}, Total loss: {total_loss:.2f}, Loss delta: {total_loss - last_total_loss:.2f} Time spent: {total_time} secs')

        last_total_loss = total_loss

        if ((best_total_loss > total_loss) and
            (best_total_loss - total_loss) > 1e-3):
            print("^- New best.")
            # Record new best
            best_total_loss = total_loss
            best_epoch = epoch
            # Save model
            now = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
            model_name = f'model_{now}.pt'

            model_save_path = os.path.join(model_path, model_name)
            t.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss},
                   model_save_path)

        else:
            # Check if we've ran out of patience
            if (epoch - best_epoch) == patience:
                # Break out of training loop
                print('Out of patience, exiting training loop.')
                break

    return 0

if __name__ == '__main__':
    sys.exit(main())
