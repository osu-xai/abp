import torch
import numpy as np
import tqdm
from tensorboardX import SummaryWriter

from abp.utils import clear_summary_path
from abp.models import TransModel

# logger = logging.getLogger('root')
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

        reinforce_summary_path = self.reinforce_config.summaries_path + "/" + self.name

        # if not self.network_config.restore_network:
        #     clear_summary_path(reinforce_summary_path)
        # else:
        #     self.restore_state()

        # summary_writer = SummaryWriter(reinforce_summary_path)
        self.trans_model = TransModel(self.name, self.network_config, use_cuda)

    # Credit for functions of loss_batch, evaluate, and train
    #       goes to Aakash N S and his article at https://medium.com/dsnet/training-deep-neural-networks-on-a-gpu-with-pytorch-11079d89805
    def loss_batch(self, loss_func, xb, yb, opt=None, metric=None):
        # Generate predictions
        preds = self.trans_model.predict_batch(xb)
        # Calculate loss
        loss = loss_func(preds, yb)
                        
        if opt is not None:
            self.trans_model.fit(preds, yb, self.steps)
            self.steps += 1
        
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
    def train(self, epochs, lr, loss_fn, train_dl, valid_dl, metric=None, opt_fn=None, report_file="test_loss.txt"):
        losses, metrics = [], []
        
        # Instantiate the optimizer
        if opt_fn is None:
            print('opt_fn is None')
            opt_fn = torch.optim.SGD
            opt = torch.optim.SGD(self.trans_model.parameters(), lr=lr)
        
        for epoch in tqdm.tqdm(range(epochs)):
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

            if epoch % 1000 == 0 and epoch != 0:
                self.trans_model.save_network()
        
        self.trans_model.save_network()
        return losses, metrics
