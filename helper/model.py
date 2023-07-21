import numpy as np
import pandas as pd
import sklearn.model_selection
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from typing import Tuple

import math
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable

from tqdm.auto import tqdm
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import seaborn as sb
from matplotlib.pyplot import *
import config
import data as data_helper

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.inspection import PartialDependenceDisplay

def show_me(fig_size: Tuple[int, int] = (8, 20)):
    plt.figure(figsize=fig_size)
    plt.imshow(mpimg.imread('../US.png'))

# the same as attn_4 but with a softmax (nearly same results)
class attn(nn.Module):  # Enc_modifie modifie car comprends un softmax
    def __init__(self, timesteps, n_hidden):
        super().__init__()
        self.linear = nn.Linear(n_hidden, n_hidden)
        self.linear_out = nn.Linear(n_hidden * 2, n_hidden)

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, hidden, enc_outputs):
        #enc_outputs = [batch, timesteps, n_hidden]
        #hidden = [[1, batch, n_hidden], [1, batch, n_hidden]]
        x = self.linear(enc_outputs)
        #x = [batch, timesteps, n_hidden]
        hidden = hidden[0]+hidden[1]
        #hidden = [1, batch, n_hidden]
        hidden = hidden.squeeze(0).unsqueeze(1)
        #hidden = [batch, 1, n_hidden]
        hidden_ = hidden.transpose(-2, -1)
        #hidden = [batch, n_hidden, 1]

        x = x.bmm(hidden_)
        #x = [batch, timesteps, 1]

        x = self.tanh(x)
        x = x.squeeze(-1)
        #x = [batch, timesteps]
        x = self.softmax(x)
        scores = x.unsqueeze(-1)
        #scores = [batch, timesteps, 1]

        output = enc_outputs * scores
        #output = [batch, timesteps, n_hidden]
        loss = F.mse_loss(enc_outputs, output)
        out = torch.sum(output, dim=1)
        #out = [batch, n_hidden]

        hidden = hidden.squeeze(1)
        #out = [batch, n_hidden]

        in_linear = torch.cat((out, hidden), -1)
        #in_linear = [batch, 2*n_hidden]

        out = self.linear_out(in_linear)
        #out = [batch, n_hidden]
        out = self.tanh(out)
        #out = [batch, n_hidden]

        return out, loss, scores

class model_(nn.Module):
    def __init__(self, input_size: int, hidden_size=config.HIDDEN_SIZE, output_size=config.OUTPUT_SIZE):
        super().__init__()
        self.lstm = nn.ModuleList()
        self.dropout = nn.Dropout(p=0.2)
        self.attn = attn(config.WINDOW_SIZE, hidden_size)

        for i in range(3):  # number of layers of the model (def : 3)
            input_size = input_size if i == 0 else hidden_size
            lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
            self.lstm.append(lstm)

        hid_final = 1024
        self.linear = nn.Linear(hidden_size, hid_final)
        self.linear2 = nn.Linear(hid_final, output_size)

        # dropout as a parameter, because we will have to neutralize it for the gradients visualization part.
    def forward(self, input_seq, dropout=True):
        # to get the gradients, we have to copy the inputs in a layer of training parameters.
        self.inputs_copy = Variable(input_seq.data.clone(), requires_grad=True)
        input_values = self.inputs_copy
        #input_values = [batch, timesteps, n_features]

        outputs_values = []
        for i in range(len(self.lstm)):
            output_values, hidden = self.lstm[i](input_values)
            #output_values = [batch, timesteps, n_hidden]
            #hidden = [[1, batch, n_hidden], [1, batch, n_hidden]]

            # for residual connexion
            # if(output_values.size() == input_values.size() and cf.args.residual):
            #     output_values = output_values + input_values

            outputs_values.append(output_values)
            input_values = output_values
            if(dropout):
                input_values = self.dropout(input_values)

        output, loss, scores = self.attn(hidden, input_values)
        #scores = [batch, timesteps, 1]
        #output = [batch, n_hidden]

        output = self.linear(output)
        #output = [batch, 1024]
        predictions = self.linear2(output)
        #predictions = [batch, predict_size]

        return predictions, loss, scores


# custom loss function to divide the loss
# (worked for me once on a tensorflow project, allowing me reach the same results in 15 mn of training
# with 5-6 successive division of the loss function with Adam optimizer rather than 6 hours with Nesterov)
# class rmseLoss(torch.nn.Module):
#     def __init__(self, eps=1e-6):
#         super().__init__()
#         self.mse = nn.MSELoss()
#         self.eps = eps
#     def forward(self, yhat, y):
#         # return np.mean((yhat - y) ** 2) ** 0.5
#         return torch.sqrt(self.mse(yhat, y) + self.eps)

# class rmseLoss(torch.nn.Module):
#     def __init__(self, coeff=1):
#         super(rmseLoss,self).__init__()
#         self.coeff = coeff
#     def forward(self, x, y):
#         return torch.mean((y - x)**2)**0.5

class rmseLoss(torch.nn.Module):
    def __init__(self, coeff=1):
        super(rmseLoss, self).__init__()
        self.coeff = coeff

    def forward(self, y, y_hat):
        return torch.sqrt(torch.mean(torch.square(y - y_hat)))
        # return np.sqrt(np.mean(np.square(y - y_hat)))
        # return torch.mean((0.5/self.coeff) * torch.mean((y - x) ** 2))


def diplay_loss_graph(epochs: list[int], train_loss: list[any], test_loss: list[any]):
    # display the test loss graph :
    if len(epochs) > 1:
        fig, ax1 = plt.subplots()

        color = 'tab:blue'
        ax1.set_xlabel('epochs')
        ax1.set_ylabel('Train loss', color=color)
        ax1.plot(np.linspace(epochs[0], epochs[-1], len(train_loss)), train_loss,
                 np.linspace(epochs[0], epochs[-1], len(train_loss)), train_loss, 'o')
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        color = 'tab:orange'
        # we already handled the x-label with ax1
        ax2.set_ylabel('Test loss', color=color)
        ax2.plot(epochs, test_loss, color="orange")
        ax2.plot(epochs, test_loss, 'o', color="red")
        ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        fig.set_figheight(10)
        fig.set_figwidth(30)
        plt.show()


# compute the rmse on the data
def evaluate(model, X_test, show_progress=True):
    results_, labels_ = [], []

    with torch.no_grad():
        for batch, vals in enumerate(tqdm(X_test, disable=not show_progress)):
            # Solve the information leckage problem
            X, Y = vals[0], vals[1]

            if results_:
                results__ = results_[-X.shape[1]:]
                for index in range(len(X)):
                    X[index][-1] = results__[index]
                    
                if len(results_) > (7 * 24) * (index + 1):
                    for index_ in range(len(X[index])):
                        X[index][index_][-2] = results_[-((7 * 24) + (index_ + 1))]
                        
            seq = torch.FloatTensor(X).to(device=config.DEVICE)
            # seq = [batch, timesteps, n_features] or [batch, timesteps, n_features-1] according to if we include or not the target feature
            labels = torch.FloatTensor(Y).to(device=config.DEVICE)
            with torch.no_grad():
                model.eval()
                y_pred = model(seq)[0]

            y_pred = y_pred.cpu().numpy()
            labels = labels.cpu().numpy()
            #y_pred, labels =  [batch, predict_size]

            y_pred = np.reshape(y_pred, (-1))
            labels = np.reshape(labels, (-1))

            results_.extend(y_pred)
            labels_.extend(labels)

    return np.mean(np.subtract(results_, labels_) ** 2) ** 0.5


def train_model(input_size: any, X_train: list[any], Y_train: list[any], X_test: list[any], Y_test: list[any], is_training, batch_size, num_epoch, prc_epoch, model_path, show_loss_graph, show_progress=True) -> nn.Module:
    # Create the model
    model = model_(input_size)
    model.to(device=config.DEVICE)

    # Configure the loos function
    loss = rmseLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=5, gamma=0.1)

    # Train the model
    if is_training:
        print("Training...")
        epochs, train_loss_, test_loss_ = [], [], []
        for i in range(num_epoch):
            train_loss = 0
            # Reset the yield Rather then calling it multiple times
            generateurTrain = data_helper.get_batch_data(X_train, Y_train, batch_size=batch_size)
            generateurTest = data_helper.get_batch_data(X_test, Y_test, batch_size=batch_size)

            for batch, vals in enumerate(tqdm(range((len(X_train) // batch_size) - int((1 - prc_epoch) * (len(X_train) // batch_size))), disable=not show_progress)):
                vals = next(generateurTrain)

                optimizer.zero_grad()
                model.train()
                scaled_data = vals[0]
                scaled_label = vals[1]

                seq = torch.FloatTensor(scaled_data).to(device=config.DEVICE)
                labels = torch.FloatTensor(scaled_label).to(device=config.DEVICE)
                y_pred = model(seq)[0]

                single_loss = loss(labels, y_pred)
                single_loss.backward()

                optimizer.step()
                train_loss += single_loss.item()

            train_loss /= batch
            train_loss_.append(train_loss)

            test_loss = evaluate(model, generateurTest, show_progress)
            epochs.append(i + 1)
            test_loss_.append(test_loss)

            scheduler.step()
            print(f"Train_loss {train_loss:10.8f}, Test loss : {test_loss:.8f}, Epoch {i+1}")
            if model_path:
                torch.save(model, model_path + "_epoch_" + str(i+1))
                print(f"epoch {str(i+1)} saved")

        if model_path:
            torch.save(model, model_path)
            print("model saved")

        if show_loss_graph:
            print(model)
            diplay_loss_graph(epochs, train_loss_, test_loss_)
            
    else:
        import __main__
        setattr(__main__, "model_", model_)
        setattr(__main__, "attn", attn)
        model = torch.load(model_path)
        print("the model ", model_path, "has beed loaded !")
    return model


# compute the predicted values on the provided data
# compute the loss with the real values
# display the results and the loss
# return the real and predicted values
def display_results(model, X, Y, scaler, scale_range: Tuple[int, int], display_range: Tuple[int, int], show_progress : bool=True):
    loss_res = []
    results_, labels_ = [], []
    with torch.no_grad():
        for index in tqdm(range(len(X)), disable=not show_progress):
            # Stop the information leckage
            if results_:
                results__ = results_[-X.shape[1]:]
                for index_ in range(len(X[index])):
                    if index_ < len(results__):
                        X[index][-(index_+1)][-1] = results__[-(index_+1)]
                
                if len(results_) > (7 * 24) * (index + 1):
                    for index_ in range(len(X[index])):
                        X[index][index_][-2] = results_[-((7 * 24) + (index_ + 1))]

            seq = torch.FloatTensor([X[index]]).to(device=config.DEVICE)
            labels = torch.FloatTensor([Y[index]]).to(device=config.DEVICE)

            with torch.no_grad():
                model.eval()
                y_pred = model(seq)[0]

            y_pred = y_pred.cpu().numpy()
            labels = labels.cpu().numpy()
            #y_pred, labels =  [batch, predict_size]

            y_pred = np.reshape(scaler.inverse_transform(
                np.reshape(y_pred, (-1, 1))), (-1))
            labels = np.reshape(scaler.inverse_transform(
                np.reshape(labels, (-1, 1))), (-1))

            results_.extend(y_pred)
            labels_.extend(labels)

            single_loss = (y_pred-labels) ** 2
            loss_res.extend(single_loss)

        results_ = np.reshape(MinMaxScaler(feature_range=(np.min(scale_range[0]), np.max(scale_range[1]))).fit_transform(np.reshape(results_, (-1, 1))), (-1))
        
        if display_range:
            plt.figure(figsize=(40, 11))
            items = display_range[1] - display_range[0]

            # predict curve
            plt.plot(np.array(range(items)), results_[display_range[0]:display_range[1]], color="red", lw=2, label="Predicted value")

            # real curve
            plt.plot(np.array(range(items)), labels_[display_range[0]:display_range[1]], color="green", lw=2, label="Actual Value")

            plt.legend(loc='upper right')
            plt.show()
            
    RMSE_score, R2_score, MAE, MIAE, MAAE = np.mean(loss_res) ** 0.5, round(r2_score(labels_, results_), 2) * 100, np.mean(np.abs(np.subtract(labels_, results_))), np.min(np.abs(np.subtract(labels_, results_))), np.max(np.abs(np.subtract(labels_, results_)))
    print(f"Rmse lsoss -> {RMSE_score}")
    print(f"Mean absolute error -> {MAE}")
    print(f"R2 score -> {R2_score}")

    return labels_, results_, RMSE_score, R2_score, MAE, MIAE, MAAE

# compute the mean of the gradients on the whole provided dataset
def generate_global_heatmap(model: any, X: list[any], title: str):
    attn = []
    scores = []

    generateurTest = data_helper.get_batch_data(X, np.array([None] * len(X)), batch_size=config.BATCH_SIZE)
    for _ in tqdm(range(0, len(X), config.BATCH_SIZE)):
        val = next(generateurTest)[0]
        if len(val) >= config.BATCH_SIZE:
            model.train()
            input_val = torch.FloatTensor(val).to(device=config.DEVICE)

            res = model(input_val, dropout=False)
            loss = res[1]
            scores_ = res[2]
            res = res[0]

            loss.backward(retain_graph=True)
            attn.append(model.inputs_copy.grad.data.cpu().detach().numpy())
            scores.append(scores_.cpu().detach().numpy())

    # run completed
    attn, scores = np.reshape(attn, (-1, np.shape(attn)[-2], np.shape(attn)[-1])), np.reshape(scores, (-1, np.shape(scores)[-2], np.shape(scores)[-1]))
    attn, scores = np.mean(attn, axis=0), np.mean(scores, axis=0)

    meanAttn_ = np.transpose(attn)
    meanScores_ = np.transpose(scores)

    cmap = sb.diverging_palette(240, 10, sep=2, as_cmap=True)
    plt.figure(figsize=(40, 15))
    sb.heatmap(meanAttn_, cmap=cmap)
    plt.title(f"{title} - Feature Wise")
    plt.show()

    plt.figure(figsize=(40, 15))
    sb.heatmap(meanScores_, cmap=cmap)
    plt.title(f"{title} - Data Wise")
    plt.show()

    return attn, scores


# How can we seperate a good predictions and bad predictions
# We can get the list of all the RMSE error, and get the min and max and devide the half
def split_good_bad_data(X: list[any], Y: list[any], model: any, scaler: any):
    loss_res = []
    results_, labels_ = [], []
    with torch.no_grad():
        for index in tqdm(range(len(X))):
            # Stop the information leckage
            if results_:
                results__ = results_[-X.shape[1]:]
                for index_ in range(len(X[index])):
                    if index_ < len(results__):
                        X[index][-(index_+1)][-1] = results__[-(index_+1)]

                if len(results_) > (7 * 24) * (index + 1):
                    for index_ in range(len(X[index])):
                        X[index][index_][-2] = results_[-((7 * 24) + (index_ + 1))]

            seq = torch.FloatTensor([X[index]]).to(device=config.DEVICE)
            labels = torch.FloatTensor([Y[index]]).to(device=config.DEVICE)

            with torch.no_grad():
                model.eval()
                y_pred = model(seq)[0]

            y_pred = y_pred.cpu().numpy()
            labels = labels.cpu().numpy()
            #y_pred, labels = [batch, predict_size]

            y_pred = np.reshape(scaler.inverse_transform(np.reshape(y_pred, (-1, 1))), (-1))
            labels = np.reshape(scaler.inverse_transform(np.reshape(labels, (-1, 1))), (-1))

            results_.extend(y_pred)
            labels_.extend(labels)

            single_loss = np.mean(np.abs(labels - y_pred))
            loss_res.append(single_loss)

    good_pred, bad_pred = [], []
    for index_ in range(len(loss_res)):
        if loss_res[index_] < 0.15:
            good_pred.append(X[index_])
        else:
            bad_pred.append(X[index_])

    return good_pred, bad_pred

# compute the importance for each and every data points
def compute_grads(scaler, model, X, duree=1):
    # debInf = 0
    results = []
    for _ in range(duree):
        val = X
        data = [val]

        model.train()
        input_val = torch.FloatTensor(data).to(device=config.DEVICE)

        res = model(input_val, dropout=False)
        loss = res[1]
        scores = res[2]
        res = res[0]

        loss.backward(retain_graph=True)
        results.append(torch.squeeze(res).view(-1).data.cpu().numpy())

        # results = np.reshape(scaler.inverse_transform(np.reshape(results, (-1, 1))), (-1))
        results = np.reshape(results, (-1))
        # labels = np.reshape(scaler.inverse_transform(np.reshape(X[debInf+cf.args.windows_size+cf.args.predict_delay:debInf+cf.args.windows_size+cf.args.predict_delay+duree,cf.args.num_channel_predict], (-1, 1))), (-1))

    # print(results, "-->", labels)
    return results, model.inputs_copy.grad.data.cpu().detach().numpy(), scores.cpu().detach().numpy()


def disp_grads_rk(val):
    res = val[0]
    
    attn = val[1]
    scores = val[2]
    attn = attn[0, :, :]

    scores = scores[0, :, :]
    attn_ = np.transpose(attn)
    scores_ = np.transpose(scores)

    cmap = sb.diverging_palette(240, 10, sep=2, as_cmap=True)
    plt.figure(figsize=(40, 10))
    sb.heatmap(attn_, cmap=cmap)
    plt.show()

    plt.figure(figsize=(40, 10))
    sb.heatmap(scores_, cmap=cmap)
    plt.show()
    
    return res[0]

# Base line model or persistance model or standard model
def model_base_line(data_x: pd.DataFrame, data_y: pd.DataFrame, scaler: any):
    X, Y = data_x['base_line_demand'].tolist(), data_y.tolist()
    X = np.reshape(scaler.inverse_transform(np.reshape(X, (-1, 1))), (-1))
    Y = np.reshape(scaler.inverse_transform(np.reshape(Y, (-1, 1))), (-1))
    
    items = 7 * 24
    multiplicatior = int(len(X) / items)
    return np.append(np.tile(X[0: items], multiplicatior), X[0: (len(X) - (multiplicatior * items))]) , Y

def diplay_skill_grph(actual_values: list[any], perfect_score: any, model_prected_values: list[any], model_score: any, base_line_predicted_values: list[any], base_line_model_score: any, index_to_diplay: Tuple[int, int]):
    plt.figure(figsize=(40, 10))
    items = index_to_diplay[1] - index_to_diplay[0]

    # real curve
    plt.plot(np.array(range(items)),
             actual_values[index_to_diplay[0]:index_to_diplay[1]], color="green", lw=2, label="Actual Value")

    # predict curve
    plt.plot(np.array(range(items)),
             model_prected_values[index_to_diplay[0]:index_to_diplay[1]], color="red", lw=2, label="Predicted Value")

    # Base line prediction
    plt.plot(np.array(range(items)),
             base_line_predicted_values[index_to_diplay[0]:index_to_diplay[1]], color="blue", lw=2, label="Base-Line Value")

    plt.legend(loc='upper right')
    plt.show()

    skill_score = (model_score - base_line_model_score) / (perfect_score - base_line_model_score)
    print(f"The skill score -> {skill_score}")
    return skill_score

def cross_validation(X: list[any], Y: list[any], k: int, y_scaler: any, scaler_range: Tuple[int, int]) -> list[any]:
    split = len(X) // k
    datasets_x, datasets_y = [], []
    matrix = []
    curr_model = None

    # Create the folds, split the Dataset
    for fold in range(k):
        datasets_x.append(X[split * fold: (fold + 1) * split] if fold < (k - 1) else X[split * fold:])
        datasets_y.append(Y[split * fold: (fold + 1) * split] if fold < (k - 1) else Y[split * fold:])
    
    # Split each fold to a train and test set
    for ds_index in range(len(datasets_x)):
        X_train_cross_validation, X_test_cross_validation, Y_train_cross_validation, Y_test_cross_validation = sklearn.model_selection.train_test_split(datasets_x[ds_index], datasets_y[ds_index], test_size=0.1, random_state=0)
        
        # Train the model
        curr_model = train_model(np.shape(X[0])[-1], X_train_cross_validation, Y_train_cross_validation, X_test_cross_validation, Y_test_cross_validation, True, config.BATCH_SIZE, config.NUM_EPOCHS, config.PRC_EPOCH, None, False, False)
            
        # Test the model, (get, min, max, and average, and performance score, MSE or R2 score)
        actual_cv, predicted_cv, RSME_cv, R2_cv, MAE_cv, MIN_CV, MAX_cv = display_results(curr_model, X_test_cross_validation, Y_test_cross_validation, y_scaler, scaler_range, None, False)
        
        # print(f"Fold - {ds_index + 1}, Mean Error - {MAE_cv}, Mini Error - {MIN_CV}, Max Error - {MAX_cv}")
        matrix.append([MAE_cv, MIN_CV, MAX_cv, RSME_cv, R2_cv, MAE_cv])
    
    return matrix
    
def plot_cross_validation(results: list[any]):
    # line plot of k mean values with min/max error bars
    plt.errorbar(range(len(results)), [i[0] for i in results], yerr=[[i[1] for i in results], [i[2] for i in results]], fmt='o')
    # plot the ideal case in a separate color
    plt.plot(range(len(results)), [results[i][0] for i in range(len(results))], color='r')
    # show the plot
    plt.figure(figsize=(40, 20))
    plt.show()
    

def train_plyrg_model(degree: int, X: any, Y: any):
    polyreg = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    polyreg.fit(X, Y)
    return polyreg

def generate_feature_graph(model: any, X: any, n_cols: int, n_rows: int, columns: list[str]):
    # n_cols = 3
    # n_rows = int(len(X.columns)/n_cols)
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(40, 40))
    PartialDependenceDisplay.from_estimator(model, X, columns, ax=ax, n_cols=columns)
    fig.suptitle('Partial Dependency Plots')
    fig.tight_layout()
