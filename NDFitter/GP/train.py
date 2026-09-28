import numpy as np
import GPy
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import argparse
import ast
import pickle
import os
from NDFitter.utils import *
from NDFitter.paths import project_path

def train_gp(X, y, model_name):
    kernel = GPy.kern.RBF(input_dim=2, variance=1., lengthscale=1., ARD=True)
    model = GPy.models.GPRegression(X, y, kernel)
    model.optimize(messages=True)
    model.save_model(model_name, compress=True, save_data=True)

    return model

def plot_gp(model, X, y):
    model.plot()
    plt.title("Gaussian Process Regression - Training")
    plt.savefig("gp_regression_training.png")

def load_and_predict(model_name, scaler_name, X_new):
    model = GPy.models.GPRegression.load_model(model_name + '.zip')

    # Load the scaler
    with open(scaler_name + '.pkl', 'rb') as f:
        scaler = pickle.load(f)

    X_new_scaled = scaler.transform(X_new)
    mean, var = model.predict(X_new_scaled)

    return scaler.inverse_transform(mean), var  # Return predictions in original scale

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
    parser.add_argument('--grid', action='store_true')
    parser.add_argument('--no-grid', dest='grid', action='store_false')
    parser.set_defaults(grid=True)
    # parser.add_argument("--output", help="output folder",type=str,default=parser.parse_args().data_folder)
    args = parser.parse_args()

    outputfolder = os.path.join(args.data_folder, now())
    mkdir(outputfolder)

    pointsfile = os.path.join(args.data_folder,'points.pickle')
    valuesfile = os.path.join(args.data_folder,'values.pickle')
    points = np.array(load_from_pickle(pointsfile))
    values = np.array(load_from_pickle(valuesfile))

    scaler = StandardScaler()
    X = scaler.fit_transform(points)
    y = values.reshape(-1, 1)

    # Save the scaler
    scalar_file = os.path.join(outputfolder,'scalar.pkl')
    with open(scalar_file, 'wb') as f:
        pickle.dump(scaler, f)

    # Train and save the model
    model_file = os.path.join(outputfolder,'gp_model.pkl')
    gp_model = train_gp(X, y, model_file)

    # plot_gp(gp_model, X, y)

    # Example prediction
    if args.grid == True:
        print(args.grid)
        # Define the grid for prediction
        x_grid = np.linspace(args.e0,args.eN, num=args.ne)
        y_grid = np.linspace(args.T0,args.TN, num=args.nT)
        X, Y = np.meshgrid(x_grid, y_grid)
        test_points = np.column_stack((X.ravel(), Y.ravel()))
        scaled_test_points = scaler.transform(test_points)

        # Predict the values on the grid
        prediction, _ = gp_model.predict(scaled_test_points)

        prediction_grid = prediction.reshape((args.ne,args.nT))
        grid_file = os.path.join(outputfolder,'grid_%dx%d.pickle'%(args.ne,args.nT))
        with open(grid_file, 'wb') as f:
            pickle.dump(prediction_grid, f)
        
        fig, axe = plt.subplots()
        axe.contourf(X, Y, prediction_grid, levels=200, cmap=plt.get_cmap('Oranges'),alpha=1)
        axe.set_xlabel('strain(%)')
        axe.set_ylabel('T(K)')
        grid_fig_file = os.path.join(outputfolder,'grid_%dx%d.png'%(args.ne,args.nT))
        fig.savefig(grid_fig_file)

    res_points = points
    res_values = values-gp_model.predict(X)
    res_pointsfile,res_valuesfile = os.path.join(outputfolder,'points.pickle'), os.path.join(outputfolder,'values.pickle')

    with open(res_pointsfile,'wb') as f:
        pickle.dump(res_points, f)
    with open(res_valuesfile,'wb') as f:
        pickle.dump(res_values, f)

if __name__ == "__main__":
    main()
