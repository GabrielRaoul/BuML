import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
# Get cut-down GDAL that rasterio uses
from osgeo import gdal
# ... and suppress errors
gdal.PushErrorHandler('CPLQuietErrorHandler')
import rasterio as rio
import os

def linear_dataset_plot(X,y):
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(X, y)
    plt.xlabel("X = Input data")
    plt.ylabel("Y = Labels")
    plt.title("Plot of dataset")
    plt.subplot(1, 2, 2)
    plt.scatter(X, y, marker='.')
    plt.xlabel("X = Input data")
    plt.ylabel("Y = Labels")
    plt.title("Scatterplot of dataset")
    plt.show()
  
def datasplit_plot(X_train, y_train, X_test, y_test):
    fig = plt.figure(layout="constrained")
    gs = GridSpec(2, 2, figure=fig)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(X_train, y_train, label='Training Data', marker='.', linestyle='-', color='blue')
    ax1.set_title("Training data")
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(X_test, y_test, label='Testing Data', marker='.', linestyle='-', color='orange')
    ax2.set_title("Testing data")
    ax3 = fig.add_subplot(gs[1, :])
    ax3.scatter(X_train, y_train, label='Training Data', marker='.', linestyle='-', color='blue')
    ax3.scatter(X_test, y_test, label='Testing Data', marker='+', linestyle='-', color='orange')
    ax3.set_title("Combined data")
    fig.suptitle("Datasplit")
    plt.legend()
    plt.show()

def plot_loss(history):
    plt.figure(figsize=(20,5))
    plt.plot(history.history['loss'], 'g', marker='o', label='Training Loss')
    plt.plot(history.history['val_loss'], 'b', marker='o', label='Validation Loss')
    mi = min(min(history.history['loss']),min(history.history['val_loss']))
    ma = max(max(history.history['loss']),max(history.history['val_loss']))
    plt.ylim([.9*mi, 1.05*ma])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()
    
def plot_data(x_data, y_data, y_pred = None, title=None):
    plt.figure(figsize=(15,5))
    plt.xlim([min(x_data)-10,max(x_data)+10])
    if y_pred is not None:
      plt.ylim([min(min(y_data),min(y_pred))-10, max(max(y_data),max(y_pred))+10])
      plt.plot(x_data, y_pred, color='k', label='Model Predictions')
    else:
      plt.ylim([min(y_data)-10, max(y_data)+10])
      
    plt.scatter(x_data, y_data, label='Ground Truth', color='green', alpha=0.5)
    
    plt.xlabel('X = Input data')
    plt.ylabel('Y = Labels')
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.show()
    
def plot_error(y_test, y_pred, title=None):
    plt.figure(figsize=(7,7))
    mi = min(min(y_test),min(y_pred))
    ma = max(max(y_test),max(y_pred))
    plt.plot([mi,ma],[mi,ma], color='grey', alpha=.4)
    plt.plot(y_test, y_pred, linestyle=':', color='k', marker='o', mec='#1a7a7d', mfc = '#29bdc1')
    mae = np.mean(np.abs(y_test - y_pred))
    rmse = np.sqrt(np.mean((y_test - y_pred)**2))
    plt.text(0.05, 0.95, f'MAE:   {mae:.4f}\nRMSE: {rmse:.4f}',
            horizontalalignment='left',verticalalignment='top', 
            transform=plt.gca().transAxes, fontsize=12,
            bbox=dict(facecolor='#29bdc1', alpha=1))
    plt.xlim([mi, ma])
    plt.ylim([mi, ma])
   
    plt.xlabel(r'$y_{test}$',fontsize=14)
    plt.ylabel(r'$y_{pred}$',fontsize=14)
    plt.title(title)
    plt.grid(True)
    plt.show()
    
def plot_whole_data_with_prediction(X, y, X_test, y_pred, title=None):
    plt.figure(figsize=(15,5))
    plt.xlim([min(X)-10,max(X)+10])
    plt.ylim([min(min(y),min(y_pred))-10, max(max(y),max(y_pred))+10])
    plt.plot(X, y, label='Ground Truth')
    plt.plot(X_test, y_pred, label='Predictions')
    
    plt.xlabel('X = Input data')
    plt.ylabel('Y = Labels')
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.show()
    
def plot_sat(filepath, cmap='viridis'):
  s2 = rio.open(filepath)
  data = s2.read()
  if data.shape[0] > 3:
    plt.figure(figsize=(2*data.shape[0],3))
  else:
    plt.figure(figsize=(6,4))
  plt.suptitle(f"Bands of '{os.path.basename(filepath)}'\n{data.shape=}")
  last = None
  for i in range(data.shape[0]):
    plt.subplot(1, data.shape[0]+1, i+1)
    plt.title(f"Band {i+1}")
    plt.imshow(data[i,:,:], cmap=cmap)
    plt.axis('off')
  plt.show()