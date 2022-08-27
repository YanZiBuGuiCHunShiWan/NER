import abc


class BasePredictor(object):
    __metaclass__ = abc.ABCMeta
    """
    定义抽象方法，父类中无须实现抽象方法，交给子类去完成
    """
    @abc.abstractmethod
    def predict(self, *args, **kwargs):
        pass
