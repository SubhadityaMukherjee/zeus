import matplotlib.pyplot as plt

from zeus.callbacks import Callback


class PlotLoss(Callback):
    def __init__(self, every=2):
        self.every = every

    def on_train_epoch_end(self, model):
        model.train_loss_history.append(
            [model.metrics["train"][metric] for metric in model.metrics["train"]]
        )

    def list_appender(self, s):
        temp_lis = []
        for val in s:
            try:
                temp_lis.append(val[-1])
            except TypeError:
                temp_lis.append(val)
        return temp_lis

    def on_valid_epoch_end(self, model):
        model.valid_loss_history.append(
            [model.metrics["valid"][metric] for metric in model.metrics["valid"]]
        )
        if (model.current_epoch + 1) % self.every == 0:
            plt.plot(self.list_appender(model.train_loss_history), label="Train_loss")
            plt.plot(self.list_appender(model.valid_loss_history), label="Valid_loss")
            plt.legend()
            plt.show()
