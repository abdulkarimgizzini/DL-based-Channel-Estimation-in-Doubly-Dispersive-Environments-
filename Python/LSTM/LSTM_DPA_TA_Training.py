import time
import copy
import numpy as np
import torch
import sys
import torch.nn as nn
import torch.utils.data as data
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import scipy.io as sio
from scipy.io import loadmat

class LSTM_MLP(nn.Module):
    def __init__(self, input_size, lstm_size):
        super(LSTM_MLP, self).__init__()
        self.input_size = input_size
        self.lstm_size = lstm_size
        self.lstmcell = nn.LSTMCell(input_size=self.input_size,
                                    hidden_size=self.lstm_size)
        self.out = nn.Sequential(
             nn.Linear(128, 96)
        )

    def forward(self, x_cur, h_cur=None, c_cur=None):
        batch_size, _ = x_cur.size()
        if h_cur is None and c_cur is None:
            h_cur = torch.zeros(batch_size, self.lstm_size, device=x_cur.device)
            c_cur = torch.zeros(batch_size, self.lstm_size, device=x_cur.device)
        h_next, c_next = self.lstmcell(x_cur, (h_cur, c_cur))
        out = self.out(h_next)
        #out = h_next

        return out, h_next, c_next


def calc_error(pred, target):
    error = np.sqrt(np.sum((pred - target) ** 2))
    step_error = error / pred.shape[0]
    avg_error = step_error / pred.shape[1] / pred.shape[2]
    return avg_error, step_error, error


def calc_nmse(pred, target):
    nmse = np.sum(np.abs((pred - target))**2/np.abs(target)**2) / pred.size
    return nmse


if __name__ == '__main__':
    # --------------------- parameter ----------------------------- #
    train_rate = 0.75
    val_rate = 0.25
    initial = False
    r_cnt = 1
    # ----------------------train parameter-------------------------- #
    LR = 0.01
    EPOCH = 50
    BATCH_SIZE = 128
    input_size = 104
    lstm_size = 128
    clip = 1e-4
    weight_decay = 0
    step_size = 10
    gamma = 0.8
    scheme = 'DPA_TA'
    DL_Type = 'LSTM'
    mobility = 'H'
    modu = 'QPSK'
    # ----------------------- load data -------------------------------- #
    mat= loadmat('D:\ChPrediction\{}\{}\{}_{}_Training_Dataset_40.mat'.format(mobility, modu,scheme, DL_Type))

    #mat = loadmat('D:/PredictionDataset40.mat')
    Testing_Dataset = mat['Channel_Error_Correction_Dataset']
    Testing_Dataset = Testing_Dataset[0, 0]

    input_data = Testing_Dataset['Train_X']
    label_data = Testing_Dataset['Train_Y']
    print('Training Input Dataset: ', input_data.shape)
    print('Training Label Dataset: ', label_data.shape)


    # Reshape Input and Label Data
    input_data_Re = input_data.reshape(-1, 2)
    label_data_Re = label_data.reshape(-1, 2)
    print('Reshaped Training Input Dataset: ', input_data_Re.shape)
    print('Reshaped Training Label Dataset: ', label_data_Re.shape)

    # Normalization
    scaler = StandardScaler()
    input_data_sclar = scaler.fit_transform(input_data_Re)#.reshape(input_data.shape)
    label_data_sclar = scaler.fit_transform(label_data_Re )#.reshape(label_data.shape)

    # Reshape after normalization
    input_data_sclar = input_data_sclar.reshape(input_data.shape)
    label_data_sclar = label_data_sclar.reshape(label_data.shape)
    print('Reshaped Normalized Training Input Dataset: ', input_data_sclar.shape)
    print('Reshaped Normalized Training Label Dataset: ', label_data_sclar.shape)

    # Training and Validation Datasets splits
    nums = input_data.shape[0]
    train_nums = int(train_rate * nums)
    val_nums = int(nums * val_rate)
    print('dataset size: ', nums, ', train set size: ', train_nums, ', val set size: ', val_nums)

    # Assign training data set and validation data set
    Train_X = input_data_sclar[:train_nums]
    Train_Y = label_data_sclar[:train_nums]
    Val_X = input_data_sclar[-val_nums:]
    Val_Y = label_data_sclar[-val_nums:]
    print('Train_X :' , Train_X.shape)
    print('Train_Y :', Train_Y.shape)
    print('Val_X :', Val_X.shape)
    print('Val_Y :', Val_Y.shape)
    train_input = torch.from_numpy(Train_X).type(torch.FloatTensor)
    train_label = torch.from_numpy(Train_Y).type(torch.FloatTensor)
    val_input = torch.from_numpy(input_data_sclar[-val_nums:]).type(torch.FloatTensor)
    val_label = torch.from_numpy(label_data_sclar[-val_nums:]).type(torch.FloatTensor)

    # ----------------------------- load model ---------------------------- #
    dir_name = 'D:/ChPrediction/'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if initial is False:
        # ---------------- generate batch dataset ------------------- #
        dataset = data.TensorDataset(train_input, train_label)

        loader = data.DataLoader(
            dataset=dataset,      # torch TensorDataset format
            batch_size=BATCH_SIZE,      # mini batch size
            shuffle=True,
            num_workers=8 if torch.cuda.is_available() else 0
        )

        # ---------------------- train the model ------------------------ #
        r_min_err = float('inf')
        for r in range(r_cnt):
            # ---------------- instantiate a model and optimizer ------------------- #
            model = LSTM_MLP(input_size, lstm_size).to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=weight_decay)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
            criterion = nn.MSELoss()

            LOSS_TRAIN = []
            LOSS_VAL = []
            nmse_val = []
            STEP = 0

            min_err = float('inf')
            time_train = 0

            for epoch in range(EPOCH):
                # ---------------------- train ------------------------ #
                start = time.time()
                with torch.set_grad_enabled(True):
                    scheduler.step()
                    model.train()
                    for step, (train_batch, label_batch) in enumerate(loader):
                        train_batch, label_batch = train_batch.to(device), label_batch.to(device)
                        optimizer.zero_grad()

                        output = torch.zeros_like(label_batch)
                        for t in range(train_batch.size(1)):
                            if t == 0:
                                out_t, hn, cn = model(train_batch[:, t, :])
                            else:
                                out_t, hn, cn = model(train_batch[:, t, :], hn, cn)
                            output[:, t, :] = out_t
                        loss = criterion(output, label_batch)
                        loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), clip)
                        optimizer.step()

                        avg_err, s_err, error = calc_error(output.detach().cpu().numpy(), label_batch.detach().cpu().numpy())
                        if step % 200 == 0:
                            print('Epoch: ', epoch, '| Step: ', step, '| loss: ', loss.item(), '| err: ', avg_err)
                            LOSS_TRAIN.append(loss)

                time_train += time.time() - start

                # ---------------------- validation ------------------------ #
                with torch.set_grad_enabled(False):
                    model.eval()
                    val_input, val_label = val_input.to(device), val_label.to(device)
                    output = torch.zeros_like(val_label)
                    for t in range(val_input.size(1)):
                        if t == 0:
                            val_t, hn, cn = model(val_input[:, t, :])
                        else:
                            val_t, hn, cn = model(val_input[:, t, :], hn, cn)
                        output[:, t, :] = val_t

                    loss = criterion(output, val_label)

                    avg_err, s_err, error = calc_error(output.detach().cpu().numpy(), val_label.detach().cpu().numpy())
                    print('Epoch: ', epoch, '| val err: ', avg_err)
                    LOSS_VAL.append(loss)

                    out1 = scaler.inverse_transform(output.detach().cpu().numpy().reshape(-1, 2)).reshape(output.shape)
                    val_label1 = scaler.inverse_transform(val_label.detach().cpu().numpy().reshape(-1, 2)).reshape(val_label.shape)
                    nmse = calc_nmse(out1, val_label1)
                    nmse_val.append(nmse)

                    if avg_err < min_err:
                        min_err = avg_err
                        best_model_wts = copy.deepcopy(model.state_dict())

            if min_err < r_min_err:
                r_min_err = min_err
                r_best_model_wts = best_model_wts

        model.load_state_dict(r_best_model_wts)
        torch.save(model.to('cpu'), dir_name + '{}_{}_{}_{}_Trained_Model_40.pkl'.format(mobility, modu, scheme,DL_Type))

        plt.figure(1)
        x = range(EPOCH)
        plt.semilogy(x, LOSS_TRAIN, 'r-', label='loss_train')
        plt.semilogy(x, LOSS_VAL, 'b-', label='loss_val' )
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        NMSE = np.array(nmse_val)
        sio.savemat('nmse_'+str(lstm_size)+'_'+str(EPOCH)+ '.mat', {'nmse': NMSE})

    else:
        model = torch.load(dir_name+'.pkl')

plt.show()