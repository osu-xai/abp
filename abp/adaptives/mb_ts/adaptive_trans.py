import torch
import numpy as np
import tqdm
from tensorboardX import SummaryWriter

from abp.utils import clear_summary_path
from abp.models import TransModel

import pickle
import os
import logging

logger = logging.getLogger('root')
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
IntTensor = torch.cuda.IntTensor if use_cuda else torch.IntTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor


class TransAdaptive(object):
    """ Adaptive that uses Transition Model """

    def __init__(self, name, network_config, reinforce_config):
        super(TransAdaptive, self).__init__()
        self.name = name
        self.network_config = network_config
        self.reinforce_config = reinforce_config

        # Global
        self.steps = 0
        self.epochs = 10000

        reinforce_summary_path = self.reinforce_config.summaries_path + "/" + self.name

        if self.network_config.restore_network:
            self.restore_state()
        else:
            clear_summary_path(reinforce_summary_path)

        self.summary = SummaryWriter(log_dir=reinforce_summary_path)
        self.trans_model = TransModel(self.name, self.network_config, use_cuda)

    # def __del__(self):
    #     self.save()
    #     self.summary.close()

    # Credit for functions of loss_batch, evaluate, and train
    #       goes to Aakash N S and his article at https://medium.com/dsnet/training-deep-neural-networks-on-a-gpu-with-pytorch-11079d89805
    def loss_batch(self, loss_func, xb, yb, opt=None, metric=None):
        # Generate predictions
        preds = self.trans_model.predict_batch(xb)
        # Calculate loss
        loss = loss_func(preds, yb)
                        
        if opt is not None:
            # Compute gradients
            loss.backward()
            # Update parameters             
            opt.step()
            # Reset gradients
            opt.zero_grad() 

        self.summary.add_scalars("MSE",{'Training Nexus HP MSE' : float(loss)}, self.steps)
        
        metric_result = None
        if metric is not None:
            # Compute the metric
            metric_result = metric(preds, yb)
        
        return loss.item(), len(xb), metric_result

    def evaluate(self, loss_fn, valid_dl, metric=None):
        self.trans_model.eval_mode()
        with torch.no_grad():
            # Pass each batch through the model
            results = [self.loss_batch(loss_fn, xb, yb, metric=metric) for xb,yb in valid_dl]
            # Separate losses, counts and metrics
            losses, nums, metrics = zip(*results)
            # Total size of the dataset
            total = np.sum(nums)
            # Avg. loss across batches 
            avg_loss = np.sum(np.multiply(losses, nums)) / total
            avg_metric = None
            if metric is not None:
                # Avg. of metric across batches
                avg_metric = np.sum(np.multiply(metrics, nums)) / total
        self.trans_model.train_mode()
        return avg_loss, total, avg_metric

    # This function expects that you put you're training set and validation set through the DataLoader() function from torch.utils.data.dataloader
    def train(self, epochs, loss_fn, train_dl, valid_dl, metric=None, opt_fn=None, report_file="test_loss.txt"):
        losses, metrics = [], []
        self.epochs = epochs
        
        # Instantiate the optimizer
        if opt_fn is None:
            print('opt_fn is None')
            opt_fn = torch.optim.SGD
            opt = torch.optim.SGD(self.trans_model.parameters(), lr=self.network_config.learning_rate)
        
        for epoch in tqdm.tqdm(range(epochs)):
            self.steps += 1
            # Training
            for xb, yb in train_dl:
                loss,_,_ = self.loss_batch(loss_fn, xb, yb, opt_fn)

            # Evaluation
            val_loss, total, val_metric = self.evaluate(loss_fn, valid_dl, metric)
            
            # Record the loss & metric
            losses.append(val_loss)
            metrics.append(val_metric)
            
            # Print progress
            f = open(report_file, "a+")
            if metric is None:
                f.write('Epoch [{}/{}], Loss: {:.4f}\n'
                    .format(epoch+1, epochs, val_loss))
            else:
                f.write('Epoch [{}/{}], Loss: {:.4f}, {}: {:.4f}\n'
                    .format(epoch+1, epochs, val_loss, metric.__name__, val_metric))
            f.close()

            self.summary.add_scalars("MSE",{'Overall Nexus HP MSE' : float(val_loss)}, epoch)

            if epoch % self.network_config.save_steps == 0 and epoch != 0:
                self.trans_model.save_network()
        
        self.trans_model.save_network()
        return losses, metrics

    def save(self):
        info = {
            "steps": self.steps
            # "epochs": self.network_config.epochs - self.steps
        }

        model_path = os.path.join(self.network_config.network_path, self.name + '.p')
        if os.path.exists(model_path):
            answer = input("A saved network already exists here do you wish to overwrite? (Y or n): ")
            if answer == 'Y':
                self.trans_model.save_network()
                with open(self.network_config.network_path + "/adaptive.info", "wb") as file:
                            pickle.dump(info, file, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                print("Not saving network. Exiting...")

    def restore_state(self):
        restore_path = self.network_config.network_path + "/adaptive.info"

        if self.network_config.network_path and os.path.exists(restore_path):
            logger.info("Restoring state from %s" % self.network_config.network_path)

            with open(restore_path, "rb") as file:
                info = pickle.load(file)

            self.steps = info["steps"]
            self.epochs = info["epochs"]
            logger.info("Continuing to train epochs left: %d\t with steps: %d" %
                        (self.epochs, self.steps))
