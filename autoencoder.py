from curses.ascii import DC1
from logging.handlers import DatagramHandler
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import os

from module import Autoencoder

from tqdm import tqdm
import museval
import musdb
import librosa

class MUSDB(Dataset):
    def __init__(self, mus, input_dir, num):
        super().__init__()
        self.input = np.zeros((num, 513, 1953))
        self.target = np.zeros((num, 513, 1953))
        self.num = num

        for i, track in tqdm(enumerate(mus)):
            if i >= num:
                break
            src = track.audio[1000000:2000000, 0]
            target_spec = np.abs(librosa.stft(src, n_fft=1024, window='hann', win_length=1024,hop_length=512))
            if target_spec.shape[1] % 2 == 0:
                target_spec = target_spec[:, 0:-1]
            self.target[i] = target_spec
            
            input_file = input_dir + track.name + '.wav'
            # print(input_file)
            # input()
            y, sr = librosa.load(input_file, sr=44100)
            input_spec = np.abs(librosa.stft(y[0:1000000], n_fft=1024, window='hann', win_length=1024,hop_length=512))
            # self.input[i] = input_spec
            ########### todo
            if input_spec.shape[1] % 2 == 0:
                input_spec = input_spec[:, 0:-1]
            self.input[i] = input_spec
        
    def __len__(self):
        return self.num

    def __getitem__(self, index):
        input = self.input[index]
        target = self.target[index]
        return input, target


def train(data_loader, model, optimizer, loss_module, device, num_epochs=100):
    # Training loop
    for epoch in tqdm(range(num_epochs)):
        for inputs, targets in data_loader:
            ## Step 1: Move input data to device (only strictly necessary if we use GPU)
            inputs = inputs.to(device, dtype=torch.float)
            targets = targets.to(device, dtype=torch.float)
            
            # [8, 513, 1954] -> [8, 1, 513, 1954]
            # (n_samples, channels, height, width)
            inputs = inputs.unsqueeze(1)
            targets = targets.unsqueeze(1)
            
            ## Step 2: Run the model on the input data
            preds = model(inputs)
            preds = preds.squeeze(dim=1) # Output is [Batch size, 1], but we want [Batch size]
            targets = targets.squeeze(dim=1)

            ## Step 3: Calculate the loss
            # print(preds.shape)
            # print(targets.float().shape)
            # interp: [498, 1938]
            # torch.Size([16, 501, 1941])
            # torch.Size([16, 1, 513, 1953]) 
            # 这维数都不一样也行?? loss怎么算的
            if preds.shape[1] % 2 == 0:
                preds = preds[:, 0:-1, 0:-1]
            x = (targets.shape[1] - preds.shape[1]) // 2
            y = (targets.shape[2] - preds.shape[2]) // 2
            # print(preds.shape)
            targets = targets[:, x:targets.shape[1]-x, y:targets.shape[2]-y]
            loss = loss_module(preds, targets.float())
            optimizer.zero_grad() 
            loss.backward()
            
            ## Step 5: Update the parameters
            optimizer.step()

        # print statistics
        print('epoch: ', epoch, ' loss: ', loss.item())
    

def eval(model, data_loader, device):
    model.eval() # Set model to eval mode
    loss = 0.
    
    with torch.no_grad(): # Deactivate gradients for the following code
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            preds = model(inputs)

            SDR, ISR, SIR, SAR, _ = museval.metrics.bss_eval(targets, preds)
            ## todo


if __name__ ==  '__main__':
    mus = musdb.DB(root="./database/musdb18", subsets="train")
    input_dir = "./database/noise/"
    train_num = 50
    num_epoch = 100
    dataset = MUSDB(mus, input_dir, train_num)
    data_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    print("-------- finish data loading --------")

    model = Autoencoder()
    print(model)
    loss = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=0.001)

    # print(torch.cuda.device_count())
    torch.cuda.empty_cache()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    print("------ begin training ------")

    train(data_loader, model, optimizer, loss, device, num_epoch)

    state_dict = model.state_dict()
    torch.save(state_dict, "./model/model_4c1p1k_interp.tar")

    # eval(model, data_loader, device)