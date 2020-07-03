import sys
import csv
import math
import numpy as np
import pandas as pd

data_path   = sys.argv[1]
test_path   = sys.argv[2]
result_path = sys.argv[3]

# features wanted in training
wanted_features = [5, 6, 7, 8, 9, 10, 11, 14, 15, 16, 17]
num_of_features = len(wanted_features)
index_of_label  = 4

# Load csv file
data = pd.read_csv(data_path, encoding = 'big5')

# Preproccess the csv file
data = data.iloc[:, 3:]
data[data == 'NR'] = 0
raw_data = data.to_numpy()

# smoothing method
def smooth_data(dim = (0,0), array = []):
    for feature_index in range(dim[0]):
        tmp_index = 0
        end_index = 0
        while end_index < dim[1]:
            if array[feature_index][end_index - 1] > 0:
                tmp_index = end_index
            else:
                if array[feature_index][end_index] > 0:
                    array[feature_index][tmp_index:end_index] = \
                        (array[feature_index][tmp_index - 1] + array[feature_index][end_index]) / 2
                    tmp_index = end_index
            end_index += 1

# Extract features
# Transform data(4320*18) into month_data(12*18*480)
month_data = []
for month in range(12):
    sample = np.empty([18, 480])
    for day in range(20):
        sample[:, day * 24 : (day + 1) * 24] = raw_data[(20 * month + day) * 18 : (20 * month + day + 1) * 18, :]      
    # smooth train data
    smooth_data((18, 480), sample)
    # take cos(degree) of wind direction
    sample[15:16] = np.cos(np.pi / 180 * sample[15:16])
    month_data.append(sample[wanted_features])

# Declare x for previous 9-hr data, and y for 10th-hr pm2.5
# Trim every month's data into 471 data blocks (trim every 10 hours in 480 hours)
# Flatten the previous 9-hr data
x = np.empty([num_of_months * 471, num_of_features * 9], dtype=float)
y = np.empty([num_of_months * 471, 1], dtype=float)
for month in range(len(month_data)):
    for day in range(20):
        for hour in range(24):
            if day == 19 and hour > 14:
                continue
            x[471 * month + day * 24 + hour, :] = month_data[month][:, day * 24 + hour : day * 24 + hour + 9].reshape(1, -1)
            y[471 * month + day * 24 + hour, 0] = month_data[month][index_of_label, day * 24 + hour + 9]

# Normalize every data (12 * 471)
mean_x = np.mean(x, axis=0)
std_x = np.std(x, axis=0)
for i in range(len(x)):
    for j in range(len(x[0])):
        if std_x[j] != 0:
            x[i][j] = (x[i][j] - mean_x[j]) / std_x[j]

np.save('npy/mean.npy', mean_x)
np.save('npy/std.npy', std_x)

# Split Training Data Into "train_set" and "validation_set"
split_ratio = 0.8
x_train_set = x[: math.floor(len(x) * split_ratio), :]
y_train_set = y[: math.floor(len(x) * split_ratio), :]
x_valid_set = x[math.floor(len(x) * split_ratio) :, :]
y_valid_set = y[math.floor(len(x) * split_ratio) :, :]

# Training with Adam
dim = x_train_set.shape[1] + 1 # dimension of weight vector including bias
w = np.zeros([dim, 1])
x_train_set = np.concatenate((np.ones([x_train_set.shape[0], 1]), x_train_set), axis = 1).astype(float)
eps = 0.00000001 # avoid divider being 0
train_loss = []

learning_rate = 0.005
iteration = 10000

V_dw = np.zeros([dim, 1])
S_dw = np.zeros([dim, 1])
beta_1 = 0.99
beta_2 = 0.99

for t in range(1, iteration):
    loss = np.sqrt(np.sum(np.power(np.dot(x_train_set, w) - y_train_set, 2)) / x_train_set.shape[0])
    train_loss.append(loss) #root mean square error
    
    if t % 100 == 0:
        print('epoch {}, loss={:.5f}'.format(t, loss), end='\r')
    if t % 5000 == 0:
        learning_rate /= 2
    gradient = 2 * np.dot(x_train_set.transpose(), np.dot(x_train_set, w) - y_train_set)
    
    V_dw = beta_1 * V_dw + (1 - beta_1) * gradient
    S_dw = beta_2 * S_dw + (1 - beta_2) * (gradient ** 2)
    V_dw_corrected = V_dw / (1 - (beta_1 ** t))
    S_dw_corrected = S_dw / (1 - (beta_2 ** t))
    w = w - learning_rate * V_dw_corrected / (np.sqrt(S_dw_corrected) + eps)

print('Training loss: {:.5f}'.format(train_loss[-1]))

np.save('npy/weight.npy', w)

# test validation set
if split_ratio < 1:
    x_valid_set = np.concatenate((np.ones([x_valid_set.shape[0], 1]), x_valid_set), axis = 1).astype(float)

    # Predicting validation set
    ans_y = np.dot(x_valid_set, w)

    # Print out result and error
    error = 0.0
    for i in range(len(ans_y)):
        error += (ans_y[i][0] - y_valid_set[i][0]) ** 2 / len(ans_y)
    print('Error of validation set:{:.5f}'.format(math.sqrt(error)))

# Testing
testdata = pd.read_csv(test_path, header = None, encoding = 'big5')
test_data = testdata.iloc[:, 2:]
test_data[test_data == 'NR'] = 0
test_data = test_data.to_numpy().astype(float)

# smooth test data
smooth_data(test_data.shape, test_data)
# take cos(degree) of wind direction
for i in range(240):
    test_data[15 + 18 * i : 16 + 18 * i] = np.cos(np.pi / 180 * test_data[15 + 18 * i : 16 + 18 * i])

test_x = np.empty([240, num_of_features * 9], dtype = float)
for i in range(240):
    if i != 0: wanted_features = [x + 18 for x in wanted_features]
    test_x[i, :] = test_data[wanted_features, :].reshape(1, -1)

mean_x = np.load('npy/mean.npy')
std_x = np.load('npy/std.npy')

for i in range(len(test_x)):
    for j in range(len(test_x[0])):
        if std_x[j] != 0:
            test_x[i][j] = (test_x[i][j] - mean_x[j]) / std_x[j]
test_x = np.concatenate((np.ones([240, 1]), test_x), axis = 1).astype(float)

# Predict
w = np.load('npy/weight.npy')
ans_y = np.dot(test_x, w)

# Write ans to file
with open(result_path, mode='w', newline='') as submit_file:
    csv_writer = csv.writer(submit_file)
    header = ['id', 'value']
    csv_writer.writerow(header)
    for i in range(240):
        row = ['id_' + str(i), ans_y[i][0]]
        csv_writer.writerow(row)