import numpy as np
import gpytorch
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import argparse
import ast
import pickle
import os
from NDFitter.utils import *
from NDFitter.paths import project_path
import torch

class PolynomialKernel(gpytorch.kernels.Kernel):
    def __init__(self, degree, **kwargs):
        super().__init__(**kwargs)
        self.degree = degree

    def forward(self, x1, x2, **params):
        return (1 + x1.matmul(x2.transpose(-1, -2)))**self.degree
    
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        # self.covar_module = gpytorch.kernels.ScaleKernel(PolynomialKernel(degree=2))
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=1))
        # self.covar_module = gpytorch.kernels.PolynomialKernel(3,)
        
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def train_gp(X, y, model_name,lr,epochs):
    likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.GreaterThan(5e-2))
    model = ExactGPModel(X, y, likelihood)

    if torch.cuda.is_available():
        model = model.cuda()
        X = X.cuda()
        y = y.cuda()
        likelihood = likelihood.cuda()

    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    training_iter = epochs
    for i in range(training_iter):
        optimizer.zero_grad()
        output = model(X)
        loss = -mll(output, y)
        loss.backward()
        optimizer.step()

    # Save the model and likelihood state
    torch.save({
        'model_state_dict': model.state_dict(),
        'likelihood_state_dict': likelihood.state_dict()
    }, model_name + '.pth')

    # # Save the scaler
    # with open(scaler_name + '.pkl', 'wb') as f:
    #     pickle.dump(scaler, f)

    return model, likelihood



def load_and_predict(model_name, scaler_name, X_new):
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(None, None, likelihood)

    checkpoint = torch.load(model_name + '.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    likelihood.load_state_dict(checkpoint['likelihood_state_dict'])

    # Load the scaler
    with open(scaler_name + '.pkl', 'rb') as f:
        scaler = pickle.load(f)

    X_new_scaled = torch.tensor(scaler.transform(X_new), dtype=torch.float32)

    model.eval()
    likelihood.eval()

    # Make predictions
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        preds = model(X_new_scaled)

    return preds.mean, preds.variance  # Return predictions

def main():
    parser = argparse.ArgumentParser(description="Gaussian Process Regression")
    parser.add_argument("--data_folder", help="Data folder, relative to the project root or absolute", type=project_path, required=True)
    # parser.add_argument("--values_file", help="The pickle file containing corresponding values", type=str, required=True)
    # parser.add_argument("--model_name", help="The output model name", type=str, default="gp_model")
    # parser.add_argument("--scaler_name", help="The output scaler name", type=str, default="scaler")
    parser.add_argument("--x0", help="minimum strain", type=float, default=0.5)
    parser.add_argument("--xN", help="maximum strain", type=float, default=8.2)
    parser.add_argument("--nx", help="strain steps", type=int, default=309)
    parser.add_argument("--grid", action='store_true')
    parser.add_argument("--no-grid", dest='grid', action='store_false')
    parser.add_argument("--lr",help='learning rate, 0.1 as default',type=float,default=0.1)
    parser.add_argument("--epochs",help='number of epochs',type=int,default=50)
    parser.add_argument("--add",help="if add previous grid",action='store_true')
    parser.add_argument("--no-add",help="if not add previous grid",action='store_false')
    parser.set_defaults(grid=True)
    parser.set_defaults(add=False)
    # parser.add_argument("--output", help="output folder",type=str,default=parser.parse_args().data_folder)
    args = parser.parse_args()
    print(args.x0,args.xN)
    outputfolder = os.path.join(args.data_folder, now())
    mkdir(outputfolder)

    pointsfile = os.path.join(args.data_folder,'points.pickle')
    valuesfile = os.path.join(args.data_folder,'values.pickle')
    points = np.array(load_from_pickle(pointsfile))
    points = points.reshape(-1, 1)
    values = np.array(load_from_pickle(valuesfile))

    print(points[:5])

    scaler = StandardScaler()
    X = torch.tensor(scaler.fit_transform(points),dtype=torch.float32)
    y = torch.tensor(values,dtype=torch.float32)

    

    if torch.cuda.is_available():
        X = X.cuda()
        y = y.cuda()

    # Save the scaler
    scalar_file = os.path.join(outputfolder,'scalar.pkl')
    with open(scalar_file, 'wb') as f:
        pickle.dump(scaler, f)

    # Train and save the model
    model_file = os.path.join(outputfolder,'gpytorch_model')
    gp_model,likelihood = train_gp(X, y, model_file,lr=args.lr,epochs=args.epochs)

    gp_model.eval()
    likelihood.eval()

    res_points = points
    res_values = (y-gp_model(X).mean).cpu().detach().numpy()
    with open(os.path.join(outputfolder,'fitted_data.pickle'),'wb') as f:
        pickle.dump(gp_model(X).mean.cpu().detach().numpy(),f)
    res_pointsfile,res_valuesfile = os.path.join(outputfolder,'points.pickle'), os.path.join(outputfolder,'values.pickle')

    with open(res_pointsfile,'wb') as f:
        pickle.dump(res_points, f)
    with open(res_valuesfile,'wb') as f:
        pickle.dump(res_values, f)
    # plot_gp(gp_model, X, y)

    # Example prediction
    if args.grid == True:
        # Define the grid for prediction
        nx = args.nx
        x_base = torch.linspace(args.x0, args.xN, nx).view(-1,1)
        test_points = x_base
        scaled_test_points = scaler.transform(test_points)
        scaled_test_points_torch = torch.tensor(scaled_test_points,dtype=torch.float32)

        # if torch.cuda.is_available():
        #     scaled_test_points_torch=scaled_test_points_torch.cuda()
        # # Predict the values on the grid
        # with torch.no_grad(), gpytorch.settings.fast_pred_var():
        #     prediction = gp_model.cpu()(scaled_test_points_torch.cpu())
        # prediction_grid = prediction.mean.view(args.ne,args.nT)
        
        # Predict the values on the grid
        dvd = 1000
        predictions = []
        # print(len(scaled_test_points_torch))
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            for i in range(int(len(scaled_test_points_torch)/dvd)+1):
                if (i+1)*dvd>len(scaled_test_points_torch):
                    end = len(scaled_test_points_torch)
                else:
                    end = (i+1)*dvd
                if i*dvd>=len(scaled_test_points_torch):
                    break
                scaled_torch_temp = scaled_test_points_torch[i*dvd:end]
                # print(scaled_test_points)
                if torch.cuda.is_available():
                    scaled_torch_temp=scaled_torch_temp.cuda()
                temp_prediction = gp_model(scaled_torch_temp)
                # print(temp_prediction.mean)
                predictions.append(temp_prediction.mean.cpu().numpy())
        prediction = np.concatenate(predictions)
        prediction_grid = prediction.reshape(nx)
        
        fig, axe = plt.subplots()
        axe.plot(x_base,prediction_grid)
        axe.scatter(points,values,s=1)
        axe.set_xlabel('Strain(%)')
        axe.set_ylabel('Calibration')
        grid_fig_file = os.path.join(outputfolder,'grid_%d.png'%args.nx)
        fig.savefig(grid_fig_file)
        
        grid_file = os.path.join(outputfolder,'grid_%d.pickle'%args.nx)

        if args.add and os.path.exists(os.path.join(args.data_folder,'grid_%d.pickle'%args.nx)):
            print('______add______')
            to_add_grid = load_from_pickle(os.path.join(args.data_folder,'grid_%d.pickle'%args.nx))
            prediction_grid += to_add_grid
        with open(grid_file, 'wb') as f:
            pickle.dump(prediction_grid, f)
        
        

if __name__ == "__main__":
    main()
