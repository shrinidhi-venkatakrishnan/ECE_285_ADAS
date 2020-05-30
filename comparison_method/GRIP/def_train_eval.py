import sys
import os
sys.path.append('..')
import time
import torch.utils.data as utils
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import numpy as np
from models import *
from sklearn.cluster import SpectralClustering , KMeans
from data_stream import *

from scipy.sparse.linalg import eigs
from torch.autograd import Variable

device = torch.device("cuda:0")
# s1 = True

DATA='NEW-LYFT'  

BATCH_SIZE= 64
if DATA=='NUSCENES':
    VEHICLES=80
    train_seq_len = 10
    pred_seq_len = 10  #change to 20 for NEW-LYFT
if DATA=='NEW-LYFT':
    VEHICLES=220
    train_seq_len = 10
    pred_seq_len = 20
    

FINAL_GRIP_OUTPUT_COORDINATE_SIZE = 256
FINAL_GRIP_OUTPUT_COORDINATE_SIZE_DECODER = 256
MODEL_LOC = '../../../resources/trained_models/GRIP'

def load_grip_batch(index, data_raw, batchsize):
    keys = list(data_raw[0].keys())
#     print("keys",keys)
    timesteps = len(keys) - 2
    print (timesteps)
    coordinates, n_agents = data_raw[0][keys[2]].shape
#     print("coordinates, agents", coordinates, n_agents)
    data = torch.zeros((batchsize, coordinates, n_agents, timesteps)).to(device)
    range_batch_start = batchsize*index
#     print(len(data_raw))
    range_batch_end = min(batchsize*(index+1), len(data_raw) - 1)
#     print("range_batch_start,range_batch_end", range_batch_start, range_batch_end)
    for i in range(range_batch_start, range_batch_end):
        keys = list(data_raw[i].keys())
        for t in range(timesteps):
            data[i%batchsize, :, :, t] = torch.from_numpy(data_raw[i][keys[2 + t]]).to(device)
    return data


def trainIters(n_epochs, train_dataloader, valid_dataloader, data, sufix, print_every=1, save_every=5, plot_every=1000, learning_rate=0.001):

    num_batches = int(len(train_dataloader)/BATCH_SIZE)
    array=[]
    batch_loss = []
#     num_batches = 4


    train_raw = train_dataloader
    pred_raw = valid_dataloader

    # Initialize encoder, decoders for both streams

    grip_batch_train = load_grip_batch(0, train_dataloader, BATCH_SIZE)
    grip_batch_val = load_grip_batch(0, valid_dataloader, BATCH_SIZE)
    print('data finished')
    grip_model = GRIPModel(grip_batch_train.shape[1], grip_batch_train.shape[3]).to(device)
    encoder_stream = Encoder ( FINAL_GRIP_OUTPUT_COORDINATE_SIZE , grip_batch_train.shape[2]).to ( device )
    decoder_stream = Decoder (FINAL_GRIP_OUTPUT_COORDINATE_SIZE_DECODER, grip_batch_val.shape[0], grip_batch_val.shape[2], grip_batch_val.shape[3]).to ( device )
    print(encoder_stream)
    print(decoder_stream)
    encoder_stream_optimizer = optim.RMSprop(encoder_stream.parameters(), lr=learning_rate)
    decoder_stream_optimizer = optim.RMSprop(decoder_stream.parameters(), lr=learning_rate)

    for epoch in range(0, n_epochs):

        print_loss_total_stream = 0  # Reset every plot_every

        for bch in range(num_batches):
            print('# {}/{} epoch {}/{} batch'.format(epoch, n_epochs, bch, num_batches))
            grip_batch_train = load_grip_batch ( bch, train_dataloader , BATCH_SIZE )
            grip_batch_test = load_grip_batch (bch, valid_dataloader , BATCH_SIZE )

            input_to_LSTM = grip_model ( grip_batch_train )
            
            loss_stream, output_stream_decoder = train_stream(input_to_LSTM, grip_batch_test, encoder_stream, decoder_stream, encoder_stream_optimizer, decoder_stream_optimizer)
            print(loss_stream)
            batch_loss.append(loss_stream)
            print_loss_total_stream += loss_stream

        print( 'stream average loss:', print_loss_total_stream/num_batches)
        array.append(print_loss_total_stream/num_batches)
        if (epoch + 1) % save_every == 0:
            save_model(encoder_stream, decoder_stream, grip_model, data, sufix)
    if n_epochs > 0:
        compute_accuracy_stream(train_dataloader, 	valid_dataloader, grip_model, encoder_stream, decoder_stream, n_epochs)
    save_model(encoder_stream, decoder_stream, grip_model, data, sufix )
    np.save('batch_losses.npy',batch_loss)
    print('loss over epochs',array)
    return encoder_stream, decoder_stream, grip_model

def eval(epochs, train_dataloader, valid_dataloader, data, sufix, learning_rate=1e-3, loc=MODEL_LOC):
    
    encoderloc = os.path.join(loc, 'encoder_stream_grip_{}{}.pt'.format(data, sufix))
    decoderloc = os.path.join(loc, 'decoder_stream_grip_{}{}.pt'.format(data, sufix))
    griploc = os.path.join(loc, 'grip_model_{}{}.pt'.format(data, sufix))
    train_raw = train_dataloader
    pred_raw = valid_dataloader

    # Initialize encoder, decoders for both streams

    grip_batch_train = load_grip_batch(0, train_dataloader, BATCH_SIZE)
    grip_batch_val = load_grip_batch(0, valid_dataloader, BATCH_SIZE)
#     print(grip_batch_train.shape)
#     print(grip_batch_val.shape)
    grip_model = GRIPModel(grip_batch_train.shape[1], grip_batch_train.shape[3]).to(device)
    encoder_stream = Encoder ( FINAL_GRIP_OUTPUT_COORDINATE_SIZE , grip_batch_train.shape[2]).to ( device )
    decoder_stream = Decoder (FINAL_GRIP_OUTPUT_COORDINATE_SIZE_DECODER, grip_batch_val.shape[0], grip_batch_val.shape[2], grip_batch_val.shape[3]).to ( device )
    encoder_stream_optimizer = optim.RMSprop(encoder_stream.parameters(), lr=learning_rate)
    decoder_stream_optimizer = optim.RMSprop(decoder_stream.parameters(), lr=learning_rate)
    encoder_stream.load_state_dict(torch.load(encoderloc,map_location='cuda:0'))
    encoder_stream.eval()
    decoder_stream.load_state_dict(torch.load(decoderloc,map_location='cuda:0'))
    decoder_stream.eval()
    grip_model.load_state_dict(torch.load(griploc,map_location='cuda:0'))
    grip_model.eval()
    compute_accuracy_stream(train_dataloader, valid_dataloader, grip_model, encoder_stream, decoder_stream, epochs)


def save_model( encoder_stream2, decoder_stream2, grip, data, sufix, loc=MODEL_LOC):
    torch.save(grip.state_dict(), os.path.join(loc, 'grip_model_{}{}.pt'.format(data, sufix)))
    torch.save(encoder_stream2.state_dict(), os.path.join(loc, 'encoder_stream_grip_{}{}.pt'.format(data, sufix)))
    torch.save(decoder_stream2.state_dict(), os.path.join(loc, 'decoder_stream_grip_{}{}.pt'.format(data, sufix)))
    print('model saved at {}'.format(loc))


def train_stream(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer):

    Hidden_State , _ = encoder.loop(input_tensor)
    stream2_out,_, _ = decoder.loop(Hidden_State)
   
    l = nn.MSELoss()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    scaled_train = scale_train(stream2_out, target_tensor)
    
    ## New loss calculation
    mask = np.ones(target_tensor.shape)
    mask[target_tensor.cpu().detach().numpy() == 0] = 0
    scaled_train = scaled_train*torch.tensor(mask).to(device)
    loss = l(scaled_train.float(), target_tensor)
    div = torch.tensor(scaled_train.shape[1]*scaled_train.shape[2])
    loss = loss/div
    loss.backward()
    
#     loss = l(stream2_out, target_tensor)
# loss = -log_likelihood(mu_1, mu_2, log_sigma_1, log_sigma_2, rho, target_tensor)
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item(), stream2_out

def scale_train(train_tensor, target_tensor):
    train_tensor_x = train_tensor[:,0,:,:].clone()
    train_tensor_y= train_tensor[:,1,:,:].clone()
    target_tensor_x= target_tensor[:,0,:,:].clone()
    target_tensor_y= target_tensor[:,1,:,:].clone()

    train_tensor[:,0,:,:] = torch.mean(target_tensor_x) + (train_tensor_x - torch.mean(train_tensor_x))*( (torch.std(target_tensor_x)/ torch.std(train_tensor_x)))
    train_tensor[:,1,:,:] = torch.mean(target_tensor_y) + (train_tensor_y- torch.mean(train_tensor_y))*( (torch.std(target_tensor_y)/ torch.std(train_tensor_y)))

    return train_tensor


def compute_accuracy_stream(train_dataloader, label_dataloader, grip_model, encoder, decoder, n_epochs):
    ade = 0
    fde = 0
    count = 0

#     num_batches = int(len(train_dataloader)/BATCH_SIZE)
#     num_batches = int(1000/64)
    num_batches = 1
       
    mse2=np.zeros((1,VEHICLES,pred_seq_len))
    ade_mat=np.zeros((1,VEHICLES,pred_seq_len))

    for bch in range ( num_batches ):
        print ( '# {}/{} batch'.format ( bch , num_batches ) )
        grip_batch_train = load_grip_batch (bch, train_dataloader , BATCH_SIZE )
        grip_batch_test = load_grip_batch (bch, label_dataloader , BATCH_SIZE )

        input_to_LSTM = grip_model ( grip_batch_train )
        Hidden_State , _ = encoder.loop ( input_to_LSTM )
        stream2_out , _ , _ = decoder.loop ( Hidden_State )
        scaled_train = scale_train ( stream2_out , grip_batch_test)
        
        ade_bch, mse = MSE(scaled_train/torch.max(scaled_train), grip_batch_test/torch.max(grip_batch_test)) * (torch.max(grip_batch_test)).cpu().detach().numpy()
        print(mse2.shape)
        print(ade_mat.shape)
        
        mse2=np.concatenate((mse2,mse),axis=0)
        print('mse concat',mse2.shape)
        ade_mat=np.concatenate((ade_mat,ade_bch),axis=0)
        
    print("max1",np.max(mse2))
    mse2=mse2[1:] #16 220 20
    non_zeros=np.count_nonzero(mse2)
    
    mse2=np.sum(mse2,axis=0)
    mse2=np.sum(mse2,axis=0)
    mse2=mse2/non_zeros
    rmse=np.sqrt(mse2)
    print('newest rmse',rmse)
    
#     
#     mse2=np.mean(mse2,axis=0)
#     print("max2",np.max(mse2))
#     print('mse concat',mse2.shape)
#     mse2=np.mean(mse2,axis=0)
#     print('mse concat',mse2.shape)
#     rmse=np.sqrt(mse2)
    
    ade=ade_mat[1:]
    ade=np.mean(ade,axis=0)
    ade=np.mean(ade,axis=0)
    fde=ade[-1]
    print ('Epoch batch Average ADE:',ade, '-------------FDE:',fde, '-------------RMSE', rmse )


def MSE(y_pred, y_gt, device=device):
    y_pred = y_pred.cpu().detach().numpy() #[ 16 2 220 20]
    y_gt = y_gt.cpu().detach().numpy()  
    mask = np.ones(y_gt.shape)
    mask[y_gt == 0] = 0 
    y_pred = y_pred*mask
    
    root_error = np.linalg.norm(y_pred - y_gt, axis=1)#16 220 20 of root of squared error  # 64 80 10
    print('root error shape',root_error.shape)
    
    # MSE Calculation
    accuracy=np.zeros(np.shape(y_pred))
    accuracy=y_pred-y_gt
    arr = np.power(accuracy,2) # 16 2 220 20
    x_y=np.sum(arr,axis=1)  #16 220 20
    return root_error,x_y  #root_error_agents is ade return 16 x 220 x 20 of squared error

