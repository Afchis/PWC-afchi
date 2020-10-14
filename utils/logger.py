import torch


class Logger():
    def __init__(self, len_train, len_valid):
        self.epoch = 0
        self.iter = 0
        self.len_train = len_train
        self.len_valid = len_valid

    def init(self):
        self.epoch += 1
        self.disp = {
            "train_iter" : 0,
            "train_loss" : 0,
            "train_metric" : 0,
            "valid_iter" : 0,
            "valid_loss" : 0,
            "valid_metric" : 0
        }

    def update(self, key, x):
        if key == "train_iter":
            self.iter += 1
            self.disp[key] += 1
        elif key == "valid_iter":
            self.disp[key] += 1
        else: 
            self.disp[key] += x

    def printer_train(self):
        print(" "*70, end="\r")
        print("Train prosess: [%0.2f" % (100*self.disp["train_iter"]/self.len_train) + chr(37) + "]", "Iter: %s" % self.iter,
              "Loss: %0.2f" % (self.disp["train_loss"]/self.disp["train_iter"]),
              "EPE: %0.2f" % (self.disp["train_metric"]/self.disp["train_iter"]), end="\r")

    def printer_valid(self):
        print(" "*70, end="\r")
        print("Valid prosess: [%0.2f" % (100*self.disp["valid_iter"]/self.len_valid) + chr(37) + "]",
              "Loss: %0.2f" % (self.disp["valid_loss"]/self.disp["valid_iter"]),
              "EPE: %0.2f" % (self.disp["valid_metric"]/self.disp["valid_iter"]), end="\r")

    def printer_epoch(self):
        head = "Epoch %s" % self.epoch
        print(" "*70, end="\r")
        print(head, "train:",
              "Loss: %0.2f" % (self.disp["train_loss"]/self.disp["train_iter"]),
              "EPE: %0.2f" % (self.disp["train_metric"]/self.disp["train_iter"]))
        print(" "*len(head), "valid:",
              "Loss: %0.2f" % (self.disp["valid_loss"]/self.disp["valid_iter"]),
              "EPE: %0.2f" % (self.disp["valid_metric"]/self.disp["valid_iter"]))

    def tensorboard_iter(self, writer, tb):
        if tb != "None" and self.iter % 25 == 0:
            train_loss = self.disp["train_loss"]/self.disp["train_iter"]
            writer.add_scalars('%s_iter' % tb, {'loss' : train_loss}, self.iter)

    def tensorboard_epoch(self, writer, tb):
        if tb != "None":
            train_loss = self.disp["train_loss"]/self.disp["train_iter"]
            train_metric = self.disp["train_metric"]/self.disp["train_iter"]
            valid_loss = self.disp["valid_loss"]/self.disp["valid_iter"]
            valid_metric = self.disp["valid_metric"]/self.disp["valid_iter"]
            writer.add_scalars('%s_epoch_loss' % tb, {'train_loss' : train_loss,
                                                      "valid_loss" : valid_loss}, self.epoch)
            writer.add_scalars('%s_epoch_metric' % tb, {'train_metric' : train_metric,
                                                        "valid_metric" : valid_metric}, self.epoch)

    def visual_train(self, vis, Visual, pred, label):
        if vis is True:
            if self.iter % 5 == 0:
                Visual(pred, label)