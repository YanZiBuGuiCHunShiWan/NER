import abc

class BaseTrainer(object):
    __metaclass__ = abc.ABCMeta
    """
    定义抽象方法，父类中无须实现抽象方法，交给子类去完成
    """
    @abc.abstractmethod
    def train(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def validate(self, *args, **kwargs):
        pass
