import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

import datetime


class ForecastModel(nn.Module):

    def __init__(self):
        super(ForecastModel, self).__init__()

        self.lstm = nn.LSTM(input_size=1, hidden_size=128, batch_first=True)
        self.linear = nn.Linear(in_features=128, out_features=1, bias=True)

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq.view(1, len(input_seq), -1))  # input shape = (1, 365, 1) - output shape = (1, 365, 128)
        # print(lstm_out.shape)
        outputs = self.linear(lstm_out.squeeze(0))  # input shape = (365, 128) - output shape = (365, 1)
        # print(outputs)
        outputs = torch.sigmoid(outputs)
        # print(outputs)

        return outputs[-1]


def data_loader(path):
    df = pd.read_csv(path)

    '''Data Prep'''

    df['arrival_date_month'] = pd.to_datetime(df.arrival_date_month, format='%B').dt.month
    df['arrival_date'] = pd.to_datetime(
        df['arrival_date_year'].map(str) + '/' + df['arrival_date_month'].map(str) + '/' + df[
            'arrival_date_day_of_month'].map(str))
    df['length_of_stay'] = df['stays_in_week_nights'] + df['stays_in_weekend_nights']

    resort = df.loc[(df['hotel'] == 'Resort Hotel') & (df['is_canceled'] == 0)]
    resort = resort.loc[resort.index.repeat(resort.length_of_stay)]
    resort['i'] = resort.groupby(resort.index).cumcount() + 1
    resort['arrival_date'] += pd.TimedeltaIndex(resort['i'], unit='D')
    resort['month_id'] = resort['arrival_date'].map(str).str.slice(stop=7)

    resort_final = resort.loc[:, ['month_id', 'arrival_date', 'length_of_stay']]
    resort_final = resort_final.rename(columns={'arrival_date': 'inhouse_date'})
    resort_final['length_of_stay'] = 1

    daily = resort_final.loc[(resort_final['month_id'] != '2015-07') & (resort_final['month_id'] != '2017-09')].groupby(
        'inhouse_date').sum()
    daily = daily.rename(columns={'length_of_stay': 'rooms'})
    daily['occupancy'] = daily['rooms'] / 200

    # Split data into training set and validation set
    train = daily['2015-08-01':'2017-05-30']
    valid = daily['2017-05-31':'2017-08-31']

    return train, valid


def create_inout_sequence(input_data, tw):
    inout_seq = []
    for i in range(len(input_data) - tw):
        train_seq = input_data[i: i + tw]
        train_label = input_data[i + tw: i + tw + 1]
        inout_seq.append((train_seq, train_label))
    return inout_seq


if __name__ == "__main__":
    data_path = '/Users/danielle13/Desktop/Research/data/hotel_bookings.csv'
    model_path = '/Users/danielle13/Desktop/Research/ckpt/hotel_lstm.pt'
    do_train = False
    train_window = 365

    # Create train - test data
    train_data, test_data = data_loader(data_path)

    train_inputs = torch.tensor(train_data['occupancy'].values, dtype=torch.float32)
    # print(train_inputs.shape)
    train_inout_seq = create_inout_sequence(train_inputs, train_window)
    # print(train_inout_seq)

    # Model, loss function and optimizer
    model = ForecastModel()
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    if do_train:
        # Train the model
        epochs = 5
        for i in range(epochs):
            for seq, labels in train_inout_seq:
                optimizer.zero_grad()

                # forward pass
                y_pred = model(seq)

                # compute loss and backward to update weight
                loss = loss_function(y_pred, labels)
                loss.backward()
                optimizer.step()

            print(f'epoch: {i:3} loss: {loss.item():10.8f}')

        torch.save(model.state_dict(), model_path)

    else:
        # Test the model
        fut_pred = 93
        test_inputs = train_inputs[-train_window:].tolist()
        preds = list()

        model.load_state_dict(torch.load(model_path))
        model.eval()

        for i in range(fut_pred):
            seq = torch.tensor(test_inputs[-train_window:], dtype=torch.float32)
            with torch.no_grad():
                pred = model(seq)
                preds.append(pred.item())
                test_inputs.append(test_data['occupancy'].values[i])

        # print(test_inputs[train_window:])
        # print(preds)
        predictions = np.array(test_inputs[train_window:])
        x = np.array([datetime.date(2017, 5, 31) + datetime.timedelta(days=i) for i in range(fut_pred)])

        plt.figure(figsize=(12, 4))
        plt.title('Long Short-Term Memory Method')
        plt.autoscale(axis='x', tight=True)
        plt.plot(train_data['occupancy'].values, label='Train')
        plt.plot(test_data['occupancy'].values, label='Valid')
        plt.plot(x, predictions, label='LSTM')
        plt.legend(loc='best')
        plt.show()
