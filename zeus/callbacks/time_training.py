import time

from torch.utils.tensorboard import SummaryWriter

from zeus.callbacks import Callback


class TrainingTime(Callback):
    def on_train_start(self, model):
        self.start = time.time()

    def on_train_end(self, model):
        self.end = time.time()
        hours, rem = divmod(self.end - self.start, 3600)
        minutes, seconds = divmod(rem, 60)
        print(
            "The Training took {:0>2}hours :{:0>2} minutes :{:05.2f} seconds".format(
                int(hours), int(minutes), seconds
            )
        )
