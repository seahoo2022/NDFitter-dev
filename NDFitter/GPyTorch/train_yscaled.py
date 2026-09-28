import numpy as np
import gpytorch
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import argparse
import ast
import pickle
import os
from gpytorch.kernels import RBFKernel, MaternKernel
from NDFitter.utils import *
from NDFitter.paths import project_path
import torch

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            RBFKernel(
                ard_num_dims=2,
                eps=1e-6
                )
                )
        
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def train_gp(X, y, model_name,lr,epochs):
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
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
    parser.add_argument("--e0", help="minimum strain", type=float, default=0)
    parser.add_argument("--eN", help="maximum strain", type=float, default=-0.7)
    parser.add_argument("--T0", help="minimum Temp", type=float, default=0.5)
    parser.add_argument("--TN", help="maximum Temp", type=float, default=8.2)
    parser.add_argument("--ne", help="strain steps", type=int, default=2801)
    parser.add_argument("--nT", help="Temp steps", type=int, default=1541)
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

    outputfolder = os.path.join(args.data_folder, now())
    mkdir(outputfolder)

    pointsfile = os.path.join(args.data_folder,'points.pickle')
    valuesfile = os.path.join(args.data_folder,'values.pickle')
    points = np.array(load_from_pickle(pointsfile))
    values = np.array(load_from_pickle(valuesfile))

    print(points[:5])

    scalerX = StandardScaler()
    scalery = StandardScaler()
    X = torch.tensor(scalerX.fit_transform(points),dtype=torch.float32)
    y = torch.tensor(scalery.fit_transform(values.reshape(-1,1)),dtype=torch.float32).view(-1)

    

    if torch.cuda.is_available():
        X = X.cuda()
        y = y.cuda()

    # Save the scaler
    scalar_file = os.path.join(outputfolder,'scalarX.pkl')
    with open(scalar_file, 'wb') as f:
        pickle.dump(scalerX, f)
    
    scalar_file = os.path.join(outputfolder,'scalary.pkl')
    with open(scalar_file, 'wb') as f:
        pickle.dump(scalery, f)

    # Train and save the model
    model_file = os.path.join(outputfolder,'gpytorch_model')
    gp_model,likelihood = train_gp(X, y, model_file,lr=args.lr,epochs=args.epochs)

    gp_model.eval()
    likelihood.eval()

    res_points = points
    res_values = values-scalery.inverse_transform(gp_model(X).mean.cpu().detach().numpy().reshape(-1,1)).reshape(-1)
    res_pointsfile,res_valuesfile = os.path.join(outputfolder,'points.pickle'), os.path.join(outputfolder,'values.pickle')

    with open(res_pointsfile,'wb') as f:
        pickle.dump(res_points, f)
    with open(res_valuesfile,'wb') as f:
        pickle.dump(res_values, f)
    # plot_gp(gp_model, X, y)

    # Example prediction
    if args.grid == True:
        # Define the grid for prediction
        nT,ne = args.nT,args.ne
        T_base = torch.linspace(args.T0, args.TN, nT)
        e_base = torch.linspace(args.e0, args.eN, ne)
        e, T = torch.meshgrid(e_base, T_base)
        test_points = np.vstack((e.ravel(),T.ravel())).transpose()
        scaled_test_points = scalerX.transform(test_points)
        scaled_test_points_torch = torch.tensor(scaled_test_points,dtype=torch.float32)

        # if torch.cuda.is_available():
        #     scaled_test_points_torch=scaled_test_points_torch.cuda()
        # # Predict the values on the grid
        # with torch.no_grad(), gpytorch.settings.fast_pred_var():
        #     prediction = gp_model.cpu()(scaled_test_points_torch.cpu())
        # prediction_grid = prediction.mean.view(args.ne,args.nT)
        
        # Predict the values on the grid
        dvd = 100
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
                predictions.append(scalery.inverse_transform(temp_prediction.mean.cpu().numpy().reshape(-1,1)).reshape(-1))
        prediction = np.concatenate(predictions)
        prediction_grid = prediction.reshape((ne,nT))
        
        fig, axe = plt.subplots()
        axe.contourf(e, T, prediction_grid, levels=200, cmap=plt.get_cmap('Oranges'),alpha=1)
        axe.set_xlabel('strain(%)')
        axe.set_ylabel('T(K)')
        grid_fig_file = os.path.join(outputfolder,'grid_%dx%d.png'%(args.ne,args.nT))
        fig.savefig(grid_fig_file)
        
        grid_file = os.path.join(outputfolder,'grid_%dx%d.pickle'%(args.ne,args.nT))

        if args.add and os.path.exists(os.path.join(args.data_folder,'grid_%dx%d.pickle'%(args.ne,args.nT))):
            print('______add______')
            to_add_grid = load_from_pickle(os.path.join(args.data_folder,'grid_%dx%d.pickle'%(args.ne,args.nT)))
            prediction_grid += to_add_grid
        with open(grid_file, 'wb') as f:
            pickle.dump(prediction_grid, f)
        
        

if __name__ == "__main__":
    main()
