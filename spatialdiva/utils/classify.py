import numpy as np
import pandas as pd 
import scanpy as sc 
import anndata as ann 
import sklearn 
from sklearn.metrics import balanced_accuracy_score, f1_score, \
    roc_auc_score, classification_report
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.distributions as dist
    
    
def eval_predictions(train_embeddings, train_labels, holdout_labels, holdout_embeddings, multiclass = False,    
                     type = "categorical"):
    
    if type == "categorical":
        # Convert the labels to encoded representations
        le = sklearn.preprocessing.LabelEncoder()
        le.fit(train_labels)
        train_labels_encoded = le.transform(train_labels)
        
        # Standardize the embeddings
        scaler = sklearn.preprocessing.StandardScaler()
        train_embeddings = scaler.fit_transform(train_embeddings)
        holdout_embeddings = scaler.transform(holdout_embeddings)
    
        # Get a linear sklearn classifier and train on the training data
        clf = sklearn.linear_model.LogisticRegression(random_state=42)
        clf.fit(train_embeddings, train_labels_encoded)
        holdout_preds = clf.predict(holdout_embeddings)
        holdout_probs = clf.predict_proba(holdout_embeddings)
        holdout_labels_pred = le.inverse_transform(holdout_preds)
    
        # Acurracy, f1, and auc to start
        accuracy = balanced_accuracy_score(holdout_labels, holdout_labels_pred)
        f1 = f1_score(holdout_labels, holdout_labels_pred, average = "micro")
        bal_accuracy = balanced_accuracy_score(holdout_labels, holdout_labels_pred)
        if multiclass:
            roc_auc = roc_auc_score(
                holdout_labels,
                holdout_probs,
                multi_class = "ovo"
            )
        else:
            roc_auc = roc_auc_score(
                holdout_labels,
                holdout_probs[:,1]
            )
            
        # Create dictionary of results 
        base_results_dict = dict()
        base_results_dict.update({
                "accuracy" : accuracy,
                "f1_score" : f1,
                "auc" : roc_auc,
                "balanced_accuracy" : bal_accuracy
        })
        base_results_df = pd.DataFrame(base_results_dict, index=[0])

        # Get classification summary dict for celltypes, and append batch
        class_report_dict = classification_report(
            holdout_labels,
            holdout_labels_pred,
            output_dict = True
        )
        class_results_df = pd.DataFrame(class_report_dict)

        return base_results_df, class_results_df
    
    elif type == "continuous":
        # Get a linear sklearn regressor and train on the training data
        clf = sklearn.linear_model.LinearRegression()
        clf.fit(train_embeddings, train_labels)
        holdout_preds = clf.predict(holdout_embeddings)
        
        # Get the mean squared error
        mse = ((holdout_labels - holdout_preds) ** 2).mean()
        
        return mse  
    
    elif type == "proportional":
        # Create a linear model using torch and model the 
        # proportions using a dirichlet distribution 
        class ProportionalModel(nn.Module):
            def __init__(self):
                super(ProportionalModel, self).__init__()
                self.linear = nn.Linear(train_embeddings.shape[1], train_labels.shape[1])
            
            def forward(self, x):
                # Pass through linear layer
                x = self.linear(x)
                
                # Ensure predictions are positive and softmaxed
                x_positive = F.softplus(x)
                x_softmax = F.softmax(x_positive, dim = 1)
                
                return x_softmax
            
            def negative_log_likelihood(self, y, y_pred):
                # Define dirichlet distribution
                dirichlet = dist.Dirichlet(y_pred)
                
                # Compute negative log-likelihood
                nll = -dirichlet.log_prob(y).mean()
                
                return nll
            
        # Initialize model,  and optimizer
        model = ProportionalModel()
        optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
        
        # Move to cuda if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device).double()
        
        # Convert data to tensors
        train_embeddings_tensor = torch.tensor(train_embeddings, dtype = torch.float32).to(device)
        test_embeddings_tensor = torch.tensor(holdout_embeddings, dtype = torch.float32).to(device)
        train_labels_tensor = torch.tensor(train_labels, dtype = torch.float32).to(device)
        test_labels_tensor = torch.tensor(holdout_labels, dtype = torch.float32).to(device)
        
        # Create dataloaders
        train_dataset = torch.utils.data.TensorDataset(train_embeddings_tensor, train_labels_tensor)
        test_dataset = torch.utils.data.TensorDataset(test_embeddings_tensor, test_labels_tensor)
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 32, shuffle = True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 32 , shuffle = False)
        
        # Train model
        for epoch in range(100):
            for batch in train_loader:
                optimizer.zero_grad()
                x, y = batch
                x = x.to(device).double()
                y = y.to(device).double()
                y_pred = model(x)
                loss = model.negative_log_likelihood(y, y_pred)
                loss.backward()
                optimizer.step()
            
        # Make predictions on test data
        model.eval()
        test_nlls = []
        with torch.no_grad():
            for batch in test_loader:
                x, y = batch
                x = x.to(device).double()
                y = y.to(device).double()
                y_pred = model(x)
                test_nlls.append(model.negative_log_likelihood(y, y_pred))
        
        # Compute mean negative log-likelihood on test data
        test_nll = torch.stack(test_nlls).mean()
        
        # Detach, convert to numpy
        test_nll = test_nll.cpu().detach().numpy()
        test_nll_squeezed = np.squeeze(test_nll)
        
        return test_nll_squeezed
    
    else:
        raise ValueError(f"Type {type} not recognized")
