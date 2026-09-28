import os
# import torch
import pickle
import numpy as np
import datetime
import re
import torch

def now():
    return '{date:%Y-%m-%d_%H:%M:%S}'.format(date=datetime.datetime.now())
def isarray(x):
    return isinstance(x,(list,np.array))
def mkdir(*args):
    dirs = os.path.join('',*args)
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    return dirs

def find_numbers_in_string(s):
    pattern = r"([-+]?\d*\.\d+|[-+]?\d+)([eE][-+]?\d+)?"
    return re.findall(pattern, s)

def extract_numbers(matches):
    numbers = []
    for match in matches:
        number = match[0] + (match[1] or '')
        if 'e' in number or 'E' in number:
            numbers.append(float(number))
        elif '.' in number:
            numbers.append(float(number))
        else:
            numbers.append(int(number))
    return numbers

def FLOAT(x):
    return extract_numbers(find_numbers_in_string(x))[-1]

def save_to_pickle(data,path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def load_from_pickle(pickle_file,*paras):
    with open(pickle_file,'rb') as f:
        result = pickle.load(f)
    return result

def get_points_from_file(filename):
    data = load_from_pickle(filename)
    points = data.data_derivatives['listData']['points']
    values = data.data_derivatives['listData']['GammaT']
    return points, values

def store_points_to_file(foldername, points, values):
    points_file = os.path.join(foldername, 'points.pickle')
    values_file = os.path.join(foldername, 'values.pickle')
    save_to_pickle(points, points_file)
    save_to_pickle(values, values_file)

def model_predict(model, inputs, input_type='array'):
    # Convert inputs to tensor
    if input_type == 'grid':
        X, Y = np.meshgrid(inputs[0], inputs[1])
        inputs = np.stack((X.ravel(), Y.ravel()), axis=-1)
    inputs = torch.tensor(inputs, dtype=torch.float32)
    
    # Evaluate model
    model.eval()
    with torch.no_grad():
        outputs = model(inputs)
    
    # If grid input, reshape output
    if input_type == 'grid':
        outputs = outputs.view(X.shape)
        
    return outputs

def relative_l2_error(output, target):
    error = output - target
    return torch.norm(error)**2 / torch.norm(target)


def relative_l2_error_max(output, target):
    error = output - target
    return torch.square(error) / torch.norm(target)

def now():
    return '{date:%Y-%m-%d_%H:%M:%S}'.format(date=datetime.datetime.now())