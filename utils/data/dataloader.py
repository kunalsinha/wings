from wings.utils.data import Dataset
try:
    import cupy as np
except Exception:
    import numpy as np


class DataLoader:
    """
    Wraps a dataset and generates mini batches for the training
    process.
    """

    def __init__(self, dataset, batch_size, shuffle=True):
        """
        Create a mini-batch data generator for the given dataset.

        Args:
            dataset: an instance of data.dataset or a tuple (X, Y) where 
            X and Y are arrays of features and labels respectively.
            batch_size (int): mini-batch size.
            shuffle (bool): shuffle data before every epoch.
        """
        if isinstance(dataset, Dataset):
            self.features, self.labels = dataset.dataset()
        else:
            self.features, self.labels = dataset[0], dataset[1]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.N = len(self.features)

    def __iter__(self):
        """
        Returns an iterator on the dataset.
        """
        self.indices = np.array(list(range(self.N)))
        self.batch_num = 0
        # shuffle the indices before each epoch
        if self.shuffle:
            np.random.shuffle(self.indices)
        return self

    def __next__(self):
        """
        Returns the next mini-batch.
        """
        # start index of next mini-batch
        start = self.batch_num * self.batch_size
        if start >= self.N:
            raise StopIteration
        # end index of next mini-batch
        end = (self.batch_num + 1) * self.batch_size
        # check if end doesn't exceed the total number of examples
        end = min(end, self.N)
        self.batch_num += 1
        # get the shuffle indices
        idx = self.indices[start: end]
        return (self.features[idx], self.labels[idx])
