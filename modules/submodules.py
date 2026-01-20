
from copy import deepcopy
from sklearn.linear_model import LinearRegression
import torch
import numpy as np
import torch.nn as nn
from pyro.ops.stats import crps_empirical
from sklearn.feature_selection import mutual_info_regression
from utils import DEVICE, printProgressBar


class FeatureSelectionNetwork(nn.Module):
    """
    Source: https://arxiv.org/pdf/2010.13631v3
    """
    def __init__(self, input_size:int, hidden_size:int):
        super(FeatureSelectionNetwork, self).__init__()
        self.nlin_transform = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, input_size)
        )
        self.feature_norm = nn.Softmax(dim=1)

    def forward(self, X:torch.tensor):
        """
        X: torch.tensor
        """

        Z = self.nlin_transform(X)  # B, in_size
        Zmn = torch.mean(Z, dim=0, keepdim=True)  # 1, in_size
        m = self.feature_norm(Zmn)  # 1, in_size
        
        return m

class EarlyStopping:
    def __init__(self, tolerance=5, min_delta=0):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False
        self.best_score = torch.tensor(float('inf'))
        self.best_epoch = 0

    def __call__(self, validation_loss, epoch):
        if (validation_loss - self.best_score) > self.min_delta:
            self.counter +=1
            if self.counter >= self.tolerance:  
                self.early_stop = True
        else:
            self.best_score = validation_loss
            self.best_epoch = epoch
            self.counter = 0

class CRPSLoss(nn.Module):

    def __init__(self, ignore_index=-999, reduce_mean=True):
        self.ignore_index = ignore_index
        self.reduce_mean = reduce_mean
        super(CRPSLoss, self).__init__()

    def forward(self, ypred:torch.tensor, ytrue:torch.tensor, lats:torch.tensor):
        """
        ytrue: B, D1, ..., Dk
        ypred: B, N, D1, ..., Dk
        """
        mask = torch.where(ytrue[0].flatten()==self.ignore_index, False, True)
        crps = crps_empirical(ypred.flatten(2).permute(1, 0, 2)[:, :, mask], ytrue.flatten(1)[:, mask])
        if lats is not None:
            crps *= lats.flatten(1)[:, mask]
        if self.reduce_mean:
            loss = torch.mean(crps)
        else:
            loss = crps
        return loss

class MSELoss(nn.Module):

    def __init__(self, ignore_index=-999, reduce_mean=True):
        self.ignore_index = ignore_index
        self.reduce_mean = reduce_mean
        super(MSELoss, self).__init__()

    def forward(self, ypred:torch.tensor, ytrue:torch.tensor, lats:torch.tensor):
        """
        ytrue: B, D1, ..., Dk
        ypred: N, B, D1, ..., Dk
        """
        mask = torch.where(ytrue[0].flatten()==self.ignore_index, False, True)
        mse = torch.mean(torch.square(ypred.flatten(1)[:, mask] - ytrue.flatten(1)[:, mask]), 0)
        if lats is not None:
            mse *= lats.flatten(1)[:, mask]
        if self.reduce_mean:
            loss = torch.mean(mse)
        else:
            loss = mse
        return loss

class MLP_regression(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):

        super(MLP_regression, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def init_weights(self):

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        
        x = self.act(self.fc1(x))
        x = self.fc2(x)

        return x
    
    def val_step(self, xval, yval, criterion):
        self.eval()
        y = self(xval)
        loss = criterion(y, yval)
        return loss

    def train_model(self, x, y, epochs=100, lr=1e-4, xval=None, yval=None):

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        earlystop = EarlyStopping(epochs//5, 1e-5)
        best_state = None
        val_loss = torch.tensor(float('inf'))
        for epoch in range(epochs):
            optimizer.zero_grad()
            output = self(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            if xval is not None:
                val_loss = self.val_step(xval, yval, criterion)
                earlystop(val_loss, epoch)
                if earlystop.counter == 0:
                    best_state = deepcopy(self.state_dict())
                if earlystop.early_stop:
                    self.load_state_dict(best_state)
                    printProgressBar(epoch+1, epochs, f' - early stopping at epoch {epoch+1}, train loss = {loss:.4f}, val loss = {val_loss:.4f}', printEnd=None, prefix='Progress:', suffix='Complete', length=50)
                    break
                if epoch == epochs-1:
                    self.load_state_dict(best_state)
                    printProgressBar(epoch+1, epochs, f' - epoch = {epoch+1}, train loss = {loss:.4f}, val loss = {val_loss:.4f}', prefix='Progress:', suffix='Complete', length=50)
                    break
            printProgressBar(epoch+1, epochs, f' - epoch = {epoch+1}, train loss = {loss:.4f}, val loss = {val_loss:.4f}', prefix='Progress:', suffix='Complete', length=50)
    def inference(self, x):
        self.eval()
        y = self(x)
        return y.detach().cpu().numpy()

class PMI:

    def __init__(self, DEVICE):

        self.scores = {}
        self.DEVICE = DEVICE
    
    def compute_residual(self, X, Y, Xval=None, Yval=None):        
        
        mlp = MLP_regression(X.shape[1], int(X.shape[1]*2), Y.shape[1]).to(self.DEVICE)
        if Xval is not None and Yval is not None:
            mlp.train_model(torch.as_tensor(X.astype(np.float32), device=self.DEVICE), torch.as_tensor(Y.astype(np.float32), device=self.DEVICE), lr=1e-2, epochs=2000,
                            xval=torch.as_tensor(Xval.astype(np.float32), device=self.DEVICE), yval=torch.as_tensor(Yval.astype(np.float32), device=self.DEVICE))
        else:
            mlp.train_model(torch.as_tensor(X.astype(np.float32), device=self.DEVICE), torch.as_tensor(Y.astype(np.float32), device=self.DEVICE), lr=1e-2, epochs=2000)
        E_y_x = mlp.inference(torch.as_tensor(X.astype(np.float32), device=self.DEVICE))  # Returns ndarray
        del mlp
        return Y - E_y_x
        
    def fit(self, X, y, indices, Xval=None, yval=None):

        S = []
        C = np.copy(X)
        if Xval is not None and yval is not None:
            Sval = []
            Cval = np.copy(Xval)

        while C.shape[1] > 0: 
            if S == []:
                mi = mutual_info_regression(X, y)
            else:
                mi = mutual_info_regression(u, v.ravel())
            S.append(C[:, mi.argmax()])
            C = np.delete(C, mi.argmax(), 1)
            if C.shape[1] != 0:
                if Xval is not None and yval is not None:
                    Sval.append(Cval[:, mi.argmax()])
                    Cval = np.delete(Cval, mi.argmax(), 1)
                    u = self.compute_residual(np.asarray(S).reshape(-1, len(S)), C, Xval=np.asarray(Sval).reshape(-1, len(S)), Yval=Cval)
                    v = self.compute_residual(np.asarray(S).reshape(-1, len(S)), y[:, None], Xval=np.asarray(Sval).reshape(-1, len(S)), Yval=yval[:, None])
                else:
                    u = self.compute_residual(np.asarray(S).reshape(-1, len(S)), C)
                    v = self.compute_residual(np.asarray(S).reshape(-1, len(S)), y[:, None])
            self.scores[indices[mi.argmax()]] = mi.max()
            indices = np.delete(indices, mi.argmax())


class PartialCorrelation:

    def __init__(self, DEVICE):
        self.scores = {}
        self.DEVICE = DEVICE

    def compute_residuals(self, X, Y, Xval=None, Yval=None):
        """
        Compute residuals of Y given X.
        """
        reg = LinearRegression()
        reg.fit(X, Y)
        E_y_x = reg.predict(X)
        del reg
        return Y - E_y_x

    def fit(self, X, y, indices, Xval=None, yval=None):
        """
        X: np.ndarray, shape (n_samples, n_features)
        y: np.ndarray, shape (n_samples,)
        indices: list of feature names
        """
        S = []
        C = np.copy(X)

        while C.shape[1] > 0:
            if S == []:
                corr = np.corrcoef(X.T, y)[:-1, -1]
            else:
                corr = np.corrcoef(u.T, v[:, 0])[:-1, -1]
            S.append(C[:, corr.argmax()])
            C = np.delete(C, corr.argmax(), 1)
            if C.shape[1] != 0:
                u = self.compute_residuals(np.asarray(S).reshape(-1, len(S)), C)
                v = self.compute_residuals(np.asarray(S).reshape(-1, len(S)), y[:, None])
            self.scores[indices[np.abs(corr).argmax()]] = corr.max()
            indices = np.delete(indices, np.abs(corr).argmax())
