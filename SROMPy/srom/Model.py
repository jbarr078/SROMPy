import abc

class Model(object):

    @abc.abstractmethod
    def evaluate(self, inputs):
        raise NotImplementedError