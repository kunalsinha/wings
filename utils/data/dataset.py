class Dataset:
    """
    Base class for all datasets. Custom datasets should extend this class.
    """

    def __init__(self):
        self.data = None
        self.labels = None

    def dataset(self):
        return (self.data, self.labels)

    def data(self):
        return self.data

    def labels(self):
        return self.labels
