import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os
import copy
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import argparse

seed = 1
np.random.seed(seed)

class Encoder(nn.Module):
    """encoder in DA_RNN."""

    def __init__(self, T,
                 input_size,
                 encoder_num_hidden,
                 pay_attn,
                 device,
                 parallel=False):
        """Initialize an encoder in DA_RNN."""
        super(Encoder, self).__init__()
        self.encoder_num_hidden = encoder_num_hidden
        self.input_size = input_size
        self.parallel = parallel
        self.T = T
        self.pay_attn = pay_attn
        self.device = device

        # Fig 1. Temporal Attention Mechanism: Encoder is LSTM
        self.encoder_lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.encoder_num_hidden,
            num_layers=2, dropout=0.5
        )

        # Construct Input Attention Mechanism via deterministic attention model
        # Eq. 8: W_e[h_{t-1}; s_{t-1}] + U_e * x^k
        self.encoder_attn = nn.Linear(
            in_features=2 * self.encoder_num_hidden + self.T,
            out_features=1
        )

        self.bn1 = nn.BatchNorm1d(self.input_size)

    def forward(self, X, take_att_weights=False):
        """forward.

        Args:
            X: input data

        """
        X_tilde = Variable(X.data.new(
            X.size(0), self.T, self.input_size).zero_())
        X_encoded = Variable(X.data.new(
            X.size(0), self.T, self.encoder_num_hidden).zero_())

        # Eq. 8, parameters not in nn.Linear but to be learnt
        # v_e = torch.nn.Parameter(data=torch.empty(
        #     self.input_size, self.T).uniform_(0, 1), requires_grad=True)
        # U_e = torch.nn.Parameter(data=torch.empty(
        #     self.T, self.T).uniform_(0, 1), requires_grad=True)

        # h_n, s_n: initial states with dimention hidden_size
        h_n = self._init_states(X)
        s_n = self._init_states(X)

        if take_att_weights:
            att_weights = torch.zeros(X.shape[0], self.T, X.shape[2]).type(torch.FloatTensor).to(device=self.device)

        for t in range(self.T):
            # batch_size * input_size * (2 * hidden_size + T - 1)
            x = torch.cat((h_n[-1].repeat(self.input_size, 1, 1).permute(1, 0, 2),
                        s_n[-1].repeat(self.input_size, 1, 1).permute(1, 0, 2),
                        X.permute(0, 2, 1)), dim=2)

            x = self.encoder_attn(x.view(-1, self.encoder_num_hidden * 2 + self.T))

            # get weights by softmax
            alpha = F.softmax(x.view(-1, self.input_size), dim=1)
            if take_att_weights:
                att_weights[:, t, :] = alpha

            # get new input for LSTM
            x_tilde = torch.mul(alpha, X[:, t, :])

            # Fix the warning about non-contiguous memory
            # https://discuss.pytorch.org/t/dataparallel-issue-with-flatten-parameter/8282
            self.encoder_lstm.flatten_parameters()

            # encoder LSTM
            _, final_state = self.encoder_lstm(self.bn1(x_tilde).unsqueeze(0), (h_n, s_n))
            h_n = final_state[0]
            s_n = final_state[1]

            X_tilde[:, t, :] = x_tilde
            X_encoded[:, t, :] = h_n[-1]

        if take_att_weights:
            return X_tilde, X_encoded, att_weights
        else:
            return X_tilde, X_encoded

    def _init_states(self, X):
        """Initialize all 0 hidden states and cell states for encoder.

        Args:
            X

        Returns:
            initial_hidden_states
        """
        # https://pytorch.org/docs/master/nn.html?#lstm
        return Variable(X.data.new(2, X.size(0), self.encoder_num_hidden).zero_())


class Decoder(nn.Module):
    """decoder in DA_RNN."""

    def __init__(self, T, decoder_num_hidden, encoder_num_hidden, input_feature_dimension, output_timesteps, device):
        """Initialize a decoder in DA_RNN."""
        super(Decoder, self).__init__()
        self.decoder_num_hidden = decoder_num_hidden
        self.encoder_num_hidden = encoder_num_hidden
        self.T = T
        self.output_timesteps = output_timesteps
        self.device = device

        self.attn_layer = nn.Sequential(
            nn.Linear(2 * decoder_num_hidden + encoder_num_hidden, encoder_num_hidden),
            nn.Tanh(),
            nn.Linear(encoder_num_hidden, 1)
        )
        self.lstm_layer = nn.LSTM(input_size=input_feature_dimension, hidden_size=decoder_num_hidden, num_layers=2, dropout=0.5)
        self.fc = nn.Sequential(
            nn.Linear(encoder_num_hidden + input_feature_dimension, input_feature_dimension),
            nn.BatchNorm1d(input_feature_dimension)
        )
        # nn.Linear(encoder_num_hidden + input_feature_dimension, input_feature_dimension)
        self.fc[0].weight.data.normal_()

        # self.fc_inter_pred = nn.Linear(encoder_num_hidden + decoder_num_hidden, decoder_num_hidden)
        # self.final_lstm = nn.LSTM(input_size=12800, hidden_size=decoder_num_hidden)
        self.fc_final =  nn.Sequential(
            nn.Linear(decoder_num_hidden + encoder_num_hidden, input_feature_dimension),
            nn.ReLU()
        )
        # nn.Linear(decoder_num_hidden + encoder_num_hidden, input_feature_dimension)


    def forward(self, X_encoded, y_prev, take_att_weights=False):
        """forward."""
        d_n = self._init_states(X_encoded)
        c_n = self._init_states(X_encoded)

        if take_att_weights:
            att_weights = torch.zeros(y_prev.shape[0], self.T, y_prev.shape[1]).type(torch.FloatTensor).to(device=self.device)

        for t in range(self.T):
            x = torch.cat((d_n[-1].repeat(self.T, 1, 1).permute(1, 0, 2),
                           c_n[-1].repeat(self.T, 1, 1).permute(1, 0, 2),
                           X_encoded), dim=2)

            beta = F.softmax(self.attn_layer(x.view(-1, 2 * self.decoder_num_hidden + self.encoder_num_hidden)).view(-1, self.T), dim=1)
            if take_att_weights:
                att_weights[:, t, :] = beta

            # Eqn. 14: compute context vector
            # batch_size * encoder_hidden_size
            context = torch.bmm(beta.unsqueeze(1), X_encoded)[:, 0, :]

            # Eqn. 15
            # batch_size * 1
            y_tilde = self.fc(torch.cat((context, y_prev[:, t]), dim=1))

            # Eqn. 16: LSTM
            self.lstm_layer.flatten_parameters()
            _, final_states = self.lstm_layer(
                y_tilde.unsqueeze(0), (d_n, c_n))

            d_n = final_states[0]  # 1 * batch_size * decoder_num_hidden
            c_n = final_states[1]  # 1 * batch_size * decoder_num_hidden

        # Eqn. 22: final output
        y_pred = self.fc_final(torch.cat((d_n[-1], context), dim=1))

        if take_att_weights:
            return y_pred, att_weights
        else:
            return y_pred

    def _init_states(self, X):
        """Initialize all 0 hidden states and cell states for encoder.

        Args:
            X
        Returns:
            initial_hidden_states

        """
        # hidden state and cell state [num_layers*num_directions, batch_size, hidden_size]
        # https://pytorch.org/docs/master/nn.html?#lstm
        return Variable(X.data.new(2, X.size(0), self.decoder_num_hidden).zero_())


class DA_rnn(nn.Module):
    """da_rnn."""

    def apply_max_norm(self, max_norm=3):
        with torch.no_grad():
            for param in self.parameters():
                norm = param.norm(2, dim=0, keepdim=True)
                desired = torch.clamp(norm, 0, max_norm)
                param *= (desired / (1e-6 + norm))

    def __init__(self, X, y, T,
                 encoder_num_hidden,
                 decoder_num_hidden,
                 batch_size,
                 learning_rate,
                 epochs,
                 output_timesteps,
                 train_size,
                 lr_decay,
                 lr_decay_iteration_frequency,
                 enc_attn=True,
                 parallel=False,
                 lookback=10):
        """da_rnn initialization."""
        super(DA_rnn, self).__init__()
        self.encoder_num_hidden = encoder_num_hidden
        self.decoder_num_hidden = decoder_num_hidden
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.parallel = parallel
        self.shuffle = True
        self.epochs = epochs
        self.T = T
        self.X = X
        self.y = y
        self.output_timesteps = output_timesteps
        self.lr_decay = lr_decay
        self.lr_decay_iteration_frequency = lr_decay_iteration_frequency
        self.lookback = lookback

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print("==> Use accelerator: ", self.device)

        self.Encoder = Encoder(input_size=X.shape[2],
                               encoder_num_hidden=encoder_num_hidden,
                               T=T, pay_attn=enc_attn, device=self.device).to(self.device)
        self.Decoder = Decoder(encoder_num_hidden=encoder_num_hidden,
                               decoder_num_hidden=decoder_num_hidden,
                               T=T, input_feature_dimension=1,
                               output_timesteps=output_timesteps,
                               device=self.device).to(self.device)

        # Loss function
        self.criterion = nn.MSELoss()

        if self.parallel:
            self.encoder = nn.DataParallel(self.encoder)
            self.decoder = nn.DataParallel(self.decoder)

        self.encoder_optimizer = optim.Adam(
            params=filter(lambda p: p.requires_grad, self.Encoder.parameters()),
            lr=self.learning_rate,
            weight_decay=1e-5
        )
        self.decoder_optimizer = optim.Adam(
            params=filter(lambda p: p.requires_grad, self.Decoder.parameters()),
            lr=self.learning_rate,
            weight_decay=1e-5
        )

        # Training set
        self.train_timesteps = train_size

        # Mean normalization on target only using train set
        # self.y = self.y - np.mean(self.y[:self.train_timesteps])

        self.input_size = 1

    def do_train(self, label_scaler):
        """training process."""
        iter_per_epoch = int(np.ceil(self.train_timesteps * 1. / self.batch_size))
        self.iter_losses = np.zeros(self.epochs * iter_per_epoch)
        self.epoch_losses = np.zeros(self.epochs)

        f_final_train_attns = []
        f_final_eval_train_attns = []
        f_final_eval_test_attns = []

        t_final_train_attns = []
        t_final_eval_train_attns = []
        t_final_eval_test_attns = []

        n_iter = 0

        best_test_loss = float('inf')
        real = label_scaler.inverse_transform(self.y.reshape(-1, 1)).reshape(-1)

        for epoch in range(self.epochs):
            if self.shuffle:
                ref_idx = np.random.permutation(self.train_timesteps)
            else:
                ref_idx = np.array(range(self.train_timesteps))

            idx = 0

            self.train()

            while (idx < self.train_timesteps):
                # get the indices of X_train
                indices = ref_idx[idx:(idx + self.batch_size)]
                # x = np.zeros((self.T - 1, len(indices), self.input_size))
                # x = np.zeros((len(indices), self.T - 1, self.input_size))
                _y_prev = np.zeros((len(indices), self.T, self.input_size))

                x = self.X[indices]
                y_prev = self.y[indices]
                y_gt = self.y[indices + self.batch_size]

                # format into 3D tensor
                for bs in range(len(indices)):
                    # x[bs, :, :] = self.X[indices[bs]:(indices[bs] + self.T - 1), :]
                    _y_prev[bs, :] = self.y[indices[bs]: (indices[bs] + self.T)].reshape(self.T, -1)

                if epoch == epochs-1:
                    loss, feature_attention_weights, temporal_attention_weights = self.train_forward(x, _y_prev, y_gt, take_att_weights=True)
                    f_final_train_attns.append(feature_attention_weights.cpu().data.numpy())
                    t_final_train_attns.append(temporal_attention_weights.cpu().data.numpy())
                else:
                    loss = self.train_forward(x, _y_prev, y_gt, take_att_weights=False)

                self.iter_losses[int(epoch * iter_per_epoch + idx / self.batch_size)] = loss

                idx += self.batch_size
                n_iter += 1

                if n_iter % self.lr_decay_iteration_frequency == 0 and n_iter != 0:
                    for param_group in self.encoder_optimizer.param_groups:
                        param_group['lr'] = param_group['lr'] * (1.0 - self.lr_decay)
                    for param_group in self.decoder_optimizer.param_groups:
                        param_group['lr'] = param_group['lr'] * (1.0 - self.lr_decay)

                self.epoch_losses[epoch] = np.mean(self.iter_losses[range(
                    epoch * iter_per_epoch, (epoch + 1) * iter_per_epoch)])

            if epoch % 10 == 0 or epoch == epochs-1:
                print("Epochs: ", epoch, " Iterations: ", n_iter,
                      " Loss: ", self.epoch_losses[epoch])

            if epoch % 10 == 0 or epoch == epochs-1:
                self.eval()
                with torch.no_grad():
                    if epoch == epochs-1:
                        y_train_pred, feature_attention_weights, temporal_attention_weights = self.test(on_train=True, take_att_weights=True)
                        f_final_eval_train_attns.append(feature_attention_weights)
                        t_final_eval_train_attns.append(temporal_attention_weights)

                        y_test_pred, feature_attention_weights, temporal_attention_weights = self.test(on_train=False, take_att_weights=True)
                        f_final_eval_test_attns.append(feature_attention_weights)
                        t_final_eval_test_attns.append(temporal_attention_weights)
                    else:
                        y_train_pred = self.test(on_train=True)
                        y_test_pred = self.test(on_train=False)

                y_train_pred = self.test(on_train=True)
                y_test_pred = self.test(on_train=False)
                y_pred = np.concatenate((y_train_pred, y_test_pred))
                directory = 'plots/epoch_' + str(epoch)
                os.makedirs(directory, exist_ok=True)

                y_pred = label_scaler.inverse_transform(y_pred.reshape(-1, 1)).reshape(-1)
                real = label_scaler.inverse_transform(self.y.reshape(-1, 1)).reshape(-1)

                plt.figure(figsize=(10, 6))
                line = plt.plot(range(len(real[:self.train_timesteps])), real[:self.train_timesteps], alpha=0.5, label='True Features', color='green')
                title = 'Predicted - Train | Sequence'
                line2 = plt.plot(range(len(y_pred[:self.train_timesteps])), y_pred[:self.train_timesteps], alpha=0.5, label='Predicted Features', color='red')
                plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
                plt.title(title)
                plt.savefig(directory + '/seq_pred_train' , bbox_inches='tight')
                plt.close()

                plt.figure(figsize=(10, 6))
                line = plt.plot(range(len(real[self.train_timesteps:])), real[self.train_timesteps:], alpha=0.5, label='True Features', color='green')
                title = 'Predicted - Test | Sequence'
                line2 = plt.plot(range(self.T, len(y_pred[self.train_timesteps:])+self.T), y_pred[self.train_timesteps:], alpha=0.5, label='Predicted Features', color='red')
                plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
                plt.title(title)
                plt.savefig(directory + '/seq_pred_test' , bbox_inches='tight')
                plt.close()

                # eval_indices = np.random.randint(0, y_pred.shape[0], size=10)
                # eval_indices.sort()
                # plt.ioff()
                # for seq_i in eval_indices:
                #     plt.figure(figsize=(10, 6))
                #     line = plt.plot(range(len(self.y[seq_i])), self.y[seq_i], alpha=0.25, label='True Features', color='green')
                #     if seq_i < self.train_timesteps:
                #         title = 'Predicted - Train | Sequence ' + str(seq_i)
                #     else:
                #         title = 'Predicted - Test | Sequence ' + str(seq_i)
                #     line2 = plt.plot(range(len(y_pred[seq_i])), y_pred[seq_i], alpha=0.25, label='Predicted Features', color='red')
                #     plt.legend([line[0], line2[0]], ['True Features', 'Predicted Features'], loc='upper left', bbox_to_anchor=(1, 1))
                #     plt.title(title)
                #     plt.savefig(directory + '/seq_pred_' + str(seq_i), bbox_inches='tight')
                #     plt.close()



            y_pred_test = self.test(on_train=False)
            y_pred_test = label_scaler.inverse_transform(y_pred_test.reshape(-1, 1)).reshape(-1)
            test_loss = mean_absolute_error(real[self.train_timesteps+self.T:], y_pred_test)
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                print('Best Loss at epoch', str(epoch), '|', str(best_test_loss))
                state = {
                    'epoch': epoch,
                    'state_dict': self.state_dict(),
                    'optimizer_encoder': self.encoder_optimizer.state_dict(),
                    'optimizer_decoder': self.decoder_optimizer.state_dict(),
                    'X': self.X,
                    'y': self.y,
                    'train_count': self.train_timesteps
                }
                torch.save(self.state_dict(), 'saved_model_checkpoint.pth')

                plt.figure(figsize=(10, 6))
                line = plt.plot(range(len(real[self.train_timesteps:])), real[self.train_timesteps:], alpha=0.5, label='True Features', color='green')
                title = 'Predicted - Test | MAE: ' + str(test_loss)
                line2 = plt.plot(range(self.T, len(y_pred_test)+self.T), y_pred_test, alpha=0.5, label='Predicted Features', color='red')
                plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
                plt.title(title)
                plt.savefig(directory + '/seq_pred_test_best' , bbox_inches='tight')
                plt.close()

            # # Save files in last iterations
            # if epoch == self.epochs - 1:
            #     np.savetxt('../loss.txt', np.array(self.epoch_losses), delimiter=',')
            #     np.savetxt('../y_pred.txt',
            #                np.array(self.y_pred), delimiter=',')
            #     np.savetxt('../y_true.txt',
            #                np.array(self.y_true), delimiter=',')

        return f_final_train_attns, f_final_eval_train_attns, f_final_eval_test_attns, t_final_train_attns, t_final_eval_train_attns, t_final_eval_test_attns

    def train_forward(self, X, y_prev, y_gt, take_att_weights=False):
        """
        Forward pass.

        Args:
            X:
            y_prev:
            y_gt: Ground truth label

        """
        # zero gradients
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        e_out = self.Encoder(Variable(torch.from_numpy(X).type(torch.FloatTensor).to(self.device)), take_att_weights=take_att_weights)
        if take_att_weights:
            input_weighted, input_encoded, feature_attention_weights = e_out
        else:
            input_weighted, input_encoded = e_out

        out = self.Decoder(input_encoded, Variable(torch.from_numpy(y_prev).type(torch.FloatTensor).to(self.device)), take_att_weights=take_att_weights)
        if take_att_weights:
            y_pred, temporal_attention_weights = out
        else:
            y_pred = out

        y_true = Variable(torch.from_numpy(y_gt).type(torch.FloatTensor).to(self.device))

        # y_true = y_true.view(-1, 1)
        loss = self.criterion(y_pred, y_true.unsqueeze(1))
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.apply_max_norm()

        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        if take_att_weights:
            return loss.item(), feature_attention_weights, temporal_attention_weights
        else:
            return loss.item()

    def test(self, on_train=False, take_att_weights=False):
        """test."""

        if on_train:
            y_pred = np.zeros([self.train_timesteps])
            if take_att_weights:
                t_attns = np.zeros([self.train_timesteps, self.T, self.X.shape[1]])
                f_attns = np.zeros([self.train_timesteps, self.T, self.X.shape[2]])
        else:
            y_pred = np.zeros([self.X.shape[0] - self.train_timesteps])
            if take_att_weights:
                t_attns = np.zeros([self.X.shape[0] - self.train_timesteps, self.T, self.X.shape[1]])
                f_attns = np.zeros([self.X.shape[0] - self.train_timesteps, self.T, self.X.shape[2]])

        i = 0
        while i < len(y_pred):
            batch_idx = np.array(range(len(y_pred)))[i: (i + self.batch_size)]
            y_history = np.zeros((len(batch_idx), self.T, self.input_size))
            if on_train:
                X = self.X[batch_idx, :, :]
                for bs in range(len(batch_idx)):
                    y_history[bs, :] = np.expand_dims(self.y[batch_idx[bs]: (batch_idx[bs] + self.T)], axis=1)
            else:
                X = self.X[self.train_timesteps:, :, :][batch_idx]
                for bs in range(len(batch_idx)):
                    y_history[bs, :] = np.expand_dims(self.y[(batch_idx[bs] + self.train_timesteps): (batch_idx[bs] + self.train_timesteps + self.T)], axis=1)

            y_history = Variable(torch.from_numpy(y_history).type(torch.FloatTensor).to(self.device))
            x_raw = Variable(torch.from_numpy(X).type(torch.FloatTensor).to(self.device))

            out = self.Encoder(x_raw, take_att_weights=take_att_weights)
            if take_att_weights:
                input_encoded = out[1]
                f_attns[i:(i + self.batch_size)] = out[2].cpu().data.numpy()
            else:
                _, input_encoded = out

            out = self.Decoder(input_encoded, y_history, take_att_weights=take_att_weights)
            if take_att_weights:
                y_pred[i:(i + self.batch_size)] = np.squeeze(out[0].cpu().data.numpy())
                t_attns[i:(i + self.batch_size)] = out[1].cpu().data.numpy()
            else:
                y_pred[i:(i + self.batch_size)] = np.squeeze(out.cpu().data.numpy())

            i += self.batch_size

        if take_att_weights:
            return y_pred, f_attns, t_attns
        else:
            return y_pred

#%% Command Line Inputs
parser = argparse.ArgumentParser(description="DA-LSTM inferencing for \"Spatial-Temporal Analysis of Groundwater Well Features from Neural Network Prediction of Hexavalent Chromium Concentration\".")
parser.add_argument("-t", "--target", help = "Name of target well to forecast and explain. Target has to be available in the data after processing.", required=True)
parser.add_argument("-s", "--start", help = "Start date of data to be modeled (in Year-Month-Day format). If not available, will pick the earliest date.", required=True)
parser.add_argument("-e", "--end", help = "End date of data to be modeled (in Year-Month-Day format). If not available, will pick the latest date.", required=True)
parsed = parser.parse_args()

date1 = parsed.start
date2 = parsed.end
specific_well = parsed.target

# If interative python is preferred:
# date1 = '2015-01-01'
# date2 = '2019-12-31'
# specific_well = '199-D5-127'

#%% Hyperparameters used during training
batchsize = 32
nhidden_encoder = 16
nhidden_decoder = 16
ntimestep = 5
lr = 0.001
epochs = 100
output_timesteps = 100
lr_decay = 0.1
lr_decay_iteration_frequency = 1000

#%% Read and process inputs
def data_interpolation(available_data, start, end, rollingWindow=45, feature='Concentration'):
    wells = available_data['WellName'].unique()

    dfList = []
    for well in wells:
        selection = available_data[available_data['WellName'] == well][feature]
        selection.index = pd.to_datetime(selection.index)
        selection = selection.reindex(pd.date_range(start, end))

        selection = selection.resample('D').mean()

        if (selection[~selection.isna()].shape[0] > 1):
            selection = selection.interpolate(method='polynomial', order=1)

        selection.interpolate(method='linear', order=10, inplace=True)
        selection.name = well

        dfList.append(selection)

    finals = pd.concat(dfList ,axis=1)

    print('Final Well Count without NaNs before rolling mean fill:', finals.shape[1])
    rez = finals.rolling(window=rollingWindow).mean().fillna(method='bfill').fillna(method='ffill')
    rez = rez.loc[:, ~rez.isna().any()]
    print('Final Well Count without NaNs after rolling mean fill:', rez.shape[1])
    return rez

data = pd.read_csv('../input/100HRD.csv')

data = data[data['STD_CON_LONG_NAME'] == 'Hexavalent Chromium']
data = data.rename(columns={'SAMP_SITE_NAME': 'WellName', 'STD_VALUE_RPTD': 'Concentration', 'SAMP_DATE_TIME': 'Date'})
data = data[['Date', 'WellName', 'Concentration']]
data['Date'] = pd.to_datetime(data['Date'])
data['Date'] = data['Date'].dt.strftime('%Y-%m-%d')

locs = pd.read_csv('../input/100AreaWellLocs.csv')
attri = pd.read_csv('../input/100AreaWellAttri.csv')
attri_gw = attri[attri['WELL_TYPE'] == 'GROUNDWATER WELL']
groundwater_wells = attri_gw['WELL_NAME'].unique()

data = data[data['WellName'].isin(groundwater_wells)]

selects = []
for well in data['WellName'].unique():
    selection = data[data['WellName'] == well]
    selection = selection.groupby('Date')['Concentration'].mean().to_frame().reset_index()
    selection['WellName'] = well
    selects.append(selection)
data = pd.concat(selects)
data = data.sort_values(by='Date')
data = data.set_index('Date')

if date1 < data.index[0]:
    date1 = data.index[0]
if date2 > data.index[-1]:
    date2 = data.index[-1]

data = data_interpolation(data, date1, date2, rollingWindow=45, feature='Concentration')

if specific_well not in data.columns:
    raise Exception(specific_well + ' not available after processing. Here is a list of available wells: ' + str(list(data.columns)))

y_raw = copy.deepcopy(data[specific_well])
Xx = data.drop(columns=[specific_well], inplace=False)

#%% Train/Test Split and Scaling
train_split = int(Xx.shape[0] * 0.8)

with open('scalers.pkl', 'rb') as f:
    scalers = pickle.load(f)

scaler = scalers['scaler']
label_scaler = scalers['label_scaler']

x_base_scaled = scaler.fit_transform(Xx.iloc[:train_split])
x_test_scaled = scaler.transform(Xx.iloc[train_split:])
x_base_scaled_pd = pd.DataFrame(x_base_scaled, columns=Xx.columns, index=Xx.iloc[:train_split].index)
x_test_scaled_pd = pd.DataFrame(x_test_scaled, columns=Xx.columns, index=Xx.iloc[train_split:].index)
full_data = pd.concat([x_base_scaled_pd, x_test_scaled_pd])

y_raw_train = label_scaler.fit_transform(y_raw.iloc[:train_split].values.reshape(-1, 1))
y_raw_test = label_scaler.transform(y_raw.iloc[train_split:].values.reshape(-1, 1))
y_raw_train = pd.DataFrame(y_raw_train, index=y_raw.iloc[:train_split].index, columns=[y_raw.iloc[:train_split].name])
y_raw_test = pd.DataFrame(y_raw_test, index=y_raw.iloc[train_split:].index, columns=[y_raw.iloc[train_split:].name])
y_raw = pd.concat([y_raw_train, y_raw_test])

#%% Sequencing
lookback = 50
x_sequences = []
y_sequences = []
for i in range(full_data.shape[0]):
    endIndex = i + lookback
    if endIndex >= full_data.shape[0]:
        break
    sequenceX = full_data[i:endIndex]
    sequenceY = endIndex
    x_sequences.append(sequenceX)
    y_sequences.append(sequenceY)
x_test = np.array(x_sequences)
y_array = np.array(y_sequences)
y_test_date = [y_raw.index[i] for i in y_array]
y_test = np.array([y_raw.iloc[i, 0] for i in y_array], dtype='float64')

X = x_test
y = y_test

#%% Trim x to have account for y-history usage
X = X[lookback:]

#%% Model Inferencing
model = DA_rnn(
    X,
    y,
    X.shape[1],
    nhidden_encoder,
    nhidden_decoder,
    batchsize,
    lr,
    epochs,
    output_timesteps,
    train_split - lookback,
    lr_decay,
    lr_decay_iteration_frequency,
    enc_attn=True,
    lookback=lookback
)

print("==> Start...")
state = torch.load('saved_model_checkpoint.pth')
model.load_state_dict(state)
model.eval()
y_pred, f_final_test_attns, t_final_test_attns = model.test(on_train=False, take_att_weights=True)

directory = '..\\Target_' + specific_well + '\\DA-LSTM_Results\\'
os.makedirs(directory, exist_ok=True)

y_pred = label_scaler.inverse_transform(y_pred.reshape(-1, 1)).reshape(-1)
real = label_scaler.inverse_transform(y.reshape(-1, 1)).reshape(-1)

pd_comp = pd.DataFrame(columns=['COLLECTION_DATE', 'Actual_Value'])
pd_comp['COLLECTION_DATE'] = np.array(y_test_date)[-y_pred.shape[0]:]
pd_comp['Actual_Value'] = real[train_split:]
pd_comp["Predicted_Value"] = y_pred
pd_comp.set_index('COLLECTION_DATE', inplace=True)
pd_comp.sort_index(inplace=True)

mae = mean_absolute_error(real[train_split:], y_pred)
mse = mean_squared_error(real[train_split:], y_pred, squared=True)
rmse = mean_squared_error(real[train_split:], y_pred, squared=False)
r2 = r2_score(real[train_split:], y_pred)

fig3 = plt.figure()
plt.figure(figsize=(10, 6))
line = plt.plot(pd_comp['Actual_Value'], label='Actual Value', color='green')
title = '199-D5-127: DA-LSTM On Testing,' + ' MAE: ' + str(np.round(mae,3)) + ' MSE: ' + str(np.round(mse,3)) + ' RMSE: ' + str(np.round(rmse,3)) + ' R2: ' + str(np.round(r2,3)) 
line2 = plt.plot(pd_comp['Predicted_Value'], label='Prediction', color='red')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.title(title)
plt.ylabel("Cr(VI) µg/L")
plt.xlabel("Dates of Testing Data")
plt.savefig(directory + 'seq_pred_test' , bbox_inches='tight')
plt.close()

np.save(directory + 'f_final_att_tests.npy', np.array(f_final_test_attns))
np.save(directory + 't_final_att_tests.npy', np.array(t_final_test_attns))
pd_comp.to_csv(directory + 'DA-LSTM_predicted.csv')

print('Finished Inference')