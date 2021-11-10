import torch as t

def calc_pos_weight(labels: Tensor) -> Tensor:
    """Provided a 2d tensor of one- or multi-hot vectors,
    calculate the weight of positive examples as used in
    binary cross-entropy loss.
    """
    num_labels = len(labels)
    pos_totals = sum(labels)
    neg_totals = num_labels - pos_totals

    assert t.count_nonzero(pos_totals) == len(pos_totals), "There are unexpected zero values."

    pos_weight = neg_totals / pos_totals

def load_train_and_valid(dir_path):
    train_path = os.path.join(dir_path, 'train.pt')
    valid_path = os.path.join(dir_path, 'valid.pt')

    train_dataset = t.load(train_path)
    valid_dataset = t.load(valid_path)

    return (train_dataset, valid_dataset)
