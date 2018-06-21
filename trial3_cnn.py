import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from utils import datautils
from models.cnn_3ch_2017 import CNN

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

symbols = datautils.get_symbols_list()

BATCH_SIZE = 1
TRAIN_EPOCH = 1
LEARNING_RATE = 1e-4
NUM_COINS = len(symbols)

class RewardLoss(nn.Module):
    def __init__(self):
        super(RewardLoss, self).__init__()
        self.c = 0.0025
    
    def forward(self, x, y, last_w):
        prices = y
        changes = datautils.price_change(prices)
        # print(changes.shape)
        # print(last_w.shape)

        wt_prime = (changes * last_w) / torch.sum(changes * last_w, dim=1).view(-1, 1)
        mu = 1 - (torch.sum(torch.abs(wt_prime - x), dim=1) * self.c)
        portfolio_value = torch.sum(changes * x, dim=1) * mu
        # print(portfolio_value)

        reward = -torch.mean(torch.log(portfolio_value))
        
        return reward

def main():
    model = CNN(BATCH_SIZE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = RewardLoss()

    # 34913 - 50 + 1 = 34864
    # Train 32457 (0-32456)
    # Test 2457 (32456-34912)
    states = datautils.get_states_tensor(3)

    pvm = nn.functional.softmax(torch.ones(states.shape[0], NUM_COINS+1), dim=1)
    pvm[0] = torch.cat((torch.ones(1), torch.zeros(NUM_COINS)))

    train_ids = range(0, 32457 - BATCH_SIZE + 1)
    test_ids = range(32456, len(states) - BATCH_SIZE + 1)

    train_loss = 0
    train_step = 100
    for e in range(TRAIN_EPOCH):
        for i in train_ids:
            model.train()
            state = states[i:i+BATCH_SIZE]
            wt_old = pvm[i:i+BATCH_SIZE]
            input = (state, wt_old)

            output = model(input)
            pvm[i+1:i+BATCH_SIZE+1] = output.detach()

            loss = criterion(output, state, wt_old)
            train_loss += loss.item()

            if i % train_step == 0:
                print(i, "Loss =", train_loss / train_step)
                print(output)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Save Checkpoint
        checkpoint = {
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'pvm': pvm,
        }
        torch.save(checkpoint, './checkpoints/trial3_cnn.pth.tar')

if __name__ == "__main__":
    main()