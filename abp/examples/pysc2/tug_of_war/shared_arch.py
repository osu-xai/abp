import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from tensorboardX import SummaryWriter
import os
import uuid
import tqdm

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
IntTensor = torch.cuda.IntTensor if use_cuda else torch.IntTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor
  
def weights_initialize(module):
    if type(module) == nn.Linear:
        nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain('relu'))
        module.bias.data.fill_(0.01)


# Model 1
class Encoder(nn.Model):
    def __init__(self, state_length, player1_action_length):
        super(self, Encoder).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(state_length, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 12), nn.ReLU(),
            nn.Linear(12, 3))

        self.combine = nn.Linear(3 + player1_action_length)

        self.s1_latent_length = 3 + player1_action_length

        self.encoder.apply(weights_initialize)
        self.combine.apply(weights_initialize)

    def forward(self, x, action_p1):
        latent_layer = self.encoder(x)

        sa1 = latent_layer.concatenate(action_p1)

        latent_sa1 = F.relu(self.combine(sa1))

        return latent_sa1

# Model 2 that uses Model 1
class QFunction(nn.Model):
    def __init__(self, encoder_output, q_output_length):
        super(self, QFunction).__init__()

        self.encoder_output = encoder_output

        self.qfunction = nn.Sequential(
            nn.Linear(len(self.encoder_output), 2048), nn.ReLU(),
            nn.Linear(2048, 1024), nn.ReLU(),
            nn.Linear(1024, 256), nn.ReLU(),
            nn.Linear(256, q_output_length))
        
        self.qfunction.apply(weights_initialize)
    
    def forward(self, x):
        return qfunction(self.encoder_output)

# Model 3 that uses Model 1
class TransitionFunction(nn.Model):
    def __init__(self, encoder_output, player2_action_length, unit_observation_output_length, health_observation_output_length):
        super(self, TransitionFunction).__init__()

        self.encoder = encoder_output

        self.sat_latent_length = len(encoder) + player2_action_length

        self.decoder = nn.Sequential(
            nn.Linear(self.sat_latent_length, 12), nn.ReLU(),
            nn.Linear(12, 64), nn.ReLU(), 
            nn.Linear(64, 128), nn.ReLU(), 
            nn.Linear(128, 256))

        self.unit_transition_model = nn.Linear(256, unit_observation_output_length)
        self.health_transition_model = nn.Linear(256, health_observation_output_length)

        def forward(self, x, action_p2):

            sat_latent = encoder_output.concatenate(action_p2)

            sat = self.decoder(sat_latent) 

            return F.relu(self.unit_transition_model(sat)), F.relu(self.health_transition_model(sat))

def save(model):
    unique_id = str(uuid.uuid4())
    cwd = os.getcwd()
    path = cwd + '/models'
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    file_path = path + '/JOINT_ARCH_' + unique_id + '.pt'
    torch.save(model.state_dict(), file_path)

###################################################################################################################
# Everything past this point is pretty much fair game for you to manipulate in order to get the architecture working.
# I sort of was going into the weeds for this one and I'm unsure what changes need to be made in order to get all of these models working together
# Mainly your explanation of sticking the models together, not sure how I was suppose to go about that
# I assumed that maybe I just pass in the class of Model 1 (the encoder) to the other 2 models but passing classes is more object oriented stuff
# So I just made it so the output of the encoder is what is passed into the other two Models, you may modify as you wish the above
# Thank you so much for the help Zhengxian hope you have a great weekend :)
# 谢谢!

def loss_batch(model, loss_func, xb, yb_q, yb_ou, yb_oh, action_p1, action_p2, opt=None, metric=None):
    # Generate predictions
    preds_q, preds_ou, preds_oh = model(xb, action_p1, action_p2)
    # Calculate loss
    loss_q = loss_func(preds_q, yb_q)
    loss_ou = loss_func(preds_ou, yb_ou)
    loss_oh = loss_func(preds_oh, yb_oh)
                     
    if opt is not None:
        # Compute gradients
        loss_q.backward()
        loss_ou.backward()
        loss_oh.backward()
        # Update parameters             
        opt.step()
        # Reset gradients
        opt.zero_grad()
    
    metric_result = None
    if metric is not None:
        # Compute the metric
        metric_result_q = metric(preds_q, yb_q)
        metric_result_ou = metric(preds_ou, yb_ou)
        metric_result_oh = metric(preds_oh, yb_oh)
    
    return loss_q.item(),  loss_ou.item(), loss_oh.item(), len(xb), metric_result_q, metric_result_ou, metric_result_oh

def evaluate(model, loss_fn, valid_dl, action_p1, action_p2, metric=None):
    model.eval()
    with torch.no_grad():
        # Pass each batch through the model
        results = [loss_batch(model, loss_fn, xb, yb_q, yb_ou, yb_oh, action_p1, action_p2, metric=metric) for xb,yb_q,yb_ou,yb_oh,action_p1,action_p2 in valid_dl]
        # Separate losses, counts and metrics
        losses_q, losses_ou, losses_oh, nums, metrics_q, metrics_ou, metrics_oh = zip(*results)
        # Total size of the dataset
        total = np.sum(nums)
        # Avg. loss across batches 
        avg_loss_q = np.sum(np.multiply(losses_q, nums)) / total
        avg_loss_ou = np.sum(np.multiply(losses_ou, nums)) / total
        avg_loss_oh = np.sum(np.multiply(losses_oh, nums)) / total

        avg_loss = (avg_loss_q + avg_loss_ou + avg_loss_oh) / 3

        avg_metric = None
        if metric is not None:
            # Avg. of metric across batches
            avg_metric_q = np.sum(np.multiply(metrics_q, nums)) / total
            avg_metric_ou = np.sum(np.multiply(metrics_ou, nums)) / total
            avg_metric_oh = np.sum(np.multiply(metrics_oh, nums)) / total
            avg_metric = (avg_metric_q + avg_metric_ou + avg_metric_oh) / 3
    model.train()
    return avg_loss, total, avg_metric

def fit(epochs, lr, model1, model2, model3, loss_fn, train_dl, valid_dl, metric=None, opt_fn=None):
    losses, metrics = [], []
    
    # Instantiate the optimizer
    if opt_fn is None:
        print('opt_fn is None')
        opt_fn = torch.optim.SGD
        opt1 = torch.optim.SGD(model1.parameters(), lr=lr)
        opt2 = torch.optim.SGD(model2.parameters(), lr=lr)
        opt3 = torch.optim.SGD(model3.parameters(), lr=lr)
    
    for epoch in tqdm.tqdm(range(epochs)):
        # Training
        for xb, yb_q, yb_ou, yb_oh, action_p1, action_p2 in train_dl:
            print("Help")
            # action_p1 = ??
            # action_p2 = ??
            # loss_q, loss_ou, loss_oh,_,_,_ = loss_batch(model, loss_fn, xb, yb_q, yb_ou, yb_oh, action_p1, action_p2, opt_fn)

        # Evaluation
        val_loss, total, val_metric = evaluate(model1, loss_fn, valid_dl, metric)
        
        # Record the loss & metric
        losses.append(val_loss)
        metrics.append(val_metric)
        
        # Print progress
        if metric is None:
            print('Epoch [{}/{}], Loss: {:.4f}, Accuracy: {:.4f}'
                  .format(epoch+1, epochs, val_loss, val_metric))
        else:
            print('Epoch [{}/{}], Loss: {:.4f}, {}: {:.4f}'
                  .format(epoch+1, epochs, val_loss, metric.__name__, val_metric))
        
        if epoch % 1000 == 0 and epoch != 0:
            save(model)
        
    return losses, metrics

def accuracy(outputs, ground_true):
    preds = torch.sum( torch.eq( outputs, ground_true), dim=1 )
    return torch.sum(preds).item() / len(preds)

def main():
    # ??????????
    # [O_6, S1, S2]
    data = torch.load('all_experiences_100000.pt')
    xb = data
    yb = data
    # ??????????

    state_length = 66
    player1_action_length = 4
    model1 = Encoder(state_length, player1_action_length)
    
    q_output_len = 4
    model2 = QFunction(model1, q_output_len)

    player2_action_length = 4
    unit_observation_output_length = 48
    health_observation_output_length = 4
    model3 = TransitionFunction(model1, player2_action_length, unit_observation_output_length, health_observation_output_length)


    loss_fn = nn.SmoothL1Loss()
    optimizer1 = Adam(model1.parameters(), lr = 0.0001)
    optimizer2 = Adam(model2.parameters(), lr = 0.0001)
    optimizer3 = Adam(model3.parameters(), lr = 0.0001)

    fit(10000, 0.0001, model1, model2, model3, loss_fn, xb, yb, accuracy, optimizer)

if __name__ == "__main__":
    main()