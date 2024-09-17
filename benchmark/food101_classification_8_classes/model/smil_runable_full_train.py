import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from sklearn.cluster import KMeans
import numpy as np
import random

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Example: Set seed to 42
set_seed(42)

# Helper function to initialize weights
def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

### 1. **Main Network (Classifier)**

class MainNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MainNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
        self.apply(weights_init)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

### 2. **Reconstruction Network with Variance Output**

class ReconstructionNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_priors):
        super(ReconstructionNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_logvar = nn.Linear(hidden_dim, num_priors)  # Only outputs log variance
        
        self.apply(weights_init)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        logvar = self.fc_logvar(h)
        var = torch.exp(logvar)  # Convert log variance to variance
        return var

    def reconstruct(self, x, modality_priors):
        """
        Reconstruct the missing modality using variance and modality priors.
        """
        var = self.forward(x)
        
        # Sample weights from Gaussian with mean I (identity matrix) and variance output by the network
        mean = torch.ones_like(var)  # Mean is identity (I)
        weights = mean + torch.sqrt(var) * torch.randn_like(var)  # Gaussian sampling with mean=1
        
        # Compute the weighted sum of the modality priors
        reconstructed_modality = torch.matmul(weights, modality_priors)
        return reconstructed_modality, weights, var

### 3. **Regularization Network**

class RegularizationNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(RegularizationNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, output_dim)
        self.fc_logvar = nn.Linear(hidden_dim, output_dim)
        
        self.apply(weights_init)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        mean = self.fc_mean(h)
        logvar = self.fc_logvar(h)
        std = torch.exp(0.5 * logvar)  # Convert log variance to standard deviation
        return mean, std

### 4. **SMIL Model (MAML Style Meta-Learning Framework)**

class SMILModel(nn.Module):  # Change this line to inherit from nn.Module
    def __init__(self, input_dim, hidden_dim, output_dim, num_priors, modality_priors, inner_lr, outer_lr, inner_steps):
        super(SMILModel, self).__init__()  # Add this line to initialize nn.Module
        # (rest of your __init__ code remains the same)
        self.main_network = MainNetwork(input_dim, hidden_dim, output_dim)
        self.reconstruction_network = ReconstructionNetwork(input_dim, hidden_dim, num_priors)
        self.regularization_network = RegularizationNetwork(input_dim, hidden_dim, input_dim)
        
        self.modality_priors = modality_priors  # Modality priors learned from complete samples
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.inner_steps = inner_steps  # Number of gradient steps in the inner loop

        self.optimizer = optim.Adam(
            list(self.main_network.parameters()) +
            list(self.reconstruction_network.parameters()) +
            list(self.regularization_network.parameters()), lr=self.outer_lr
        )

    def forward(self, x, modality_missing):
        # `x` is the input batch
        # `modality_missing` is a boolean vector indicating whether the modality is missing for each sample in the batch

        # Initialize lists to store results
        reconstructed_samples = []
        complete_samples = []
        weights_list = []
        var_list = []

        # Iterate over the batch
        for i in range(x.size(0)):
            if modality_missing[i]:
                # Process missing modality samples
                x_reconstructed, weights, var = self.reconstruction_network.reconstruct(x[i:i+1], self.modality_priors)

                # Obtain the mean and std for Gaussian sampling from the regularization network
                mean_reg, std_reg = self.regularization_network(x_reconstructed)
                regularizer = mean_reg + std_reg * torch.randn_like(std_reg)  # Gaussian sampling

                # Apply Softplus to the regularizer
                x_regularized = x_reconstructed * F.softplus(regularizer)

                # Append results to the lists
                reconstructed_samples.append(x_regularized)
                weights_list.append(weights)
                var_list.append(var)
            else:
                # Process complete samples
                complete_samples.append(x[i:i+1])

        # Combine complete samples and reconstructed samples
        if reconstructed_samples:
            reconstructed_batch = torch.cat(reconstructed_samples, dim=0)
            combined_batch = torch.cat(complete_samples + reconstructed_samples, dim=0)
        else:
            combined_batch = torch.cat(complete_samples, dim=0)

        # Pass the combined batch through the main network
        logits = self.main_network(combined_batch)

        # Initialize weights and var as None for cases with no missing modality
        weights = None
        var = None

        # If we have missing modalities, concatenate the weights and var for the entire batch
        if weights_list and var_list:
            weights = torch.cat(weights_list, dim=0)
            var = torch.cat(var_list, dim=0)

        return logits, weights, var


    def kl_divergence(self, mean, logvar):
        """
        Compute KL divergence between the approximated posterior and the prior.
        """
        kl = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        return kl

    def meta_train_step(self, data_loader, device):
        self.main_network.train()
        self.reconstruction_network.train()
        self.regularization_network.train()

        meta_loss = 0.0
        kl_loss = 0.0

        for x, y, modality_missing in data_loader:
            x, y = x.to(device), y.to(device)

            # Split x and y into support and query sets for the task with a 4:1 ratio
            support_size = int(0.8 * len(x))  # 80% for support
            query_size = len(x) - support_size  # 20% for query

            support_x, query_x = x[:support_size], x[support_size:]
            support_y, query_y = y[:support_size], y[support_size:]
            support_modality_missing, query_modality_missing = modality_missing[:support_size], modality_missing[support_size:]

            # Initialize a copy of network parameters for the inner loop adaptation
            theta_prime = {name: param.clone().detach().requires_grad_(True) for name, param in self.main_network.named_parameters()}

            # Inner loop: Fine-tune on the support set
            for _ in range(self.inner_steps):
                logits, weights, var = self.forward(support_x, support_modality_missing)
                loss = F.cross_entropy(logits, support_y)

                # Compute gradients w.r.t. the copied parameters
                grads = torch.autograd.grad(loss, theta_prime.values(), create_graph=True, allow_unused=True)
                theta_prime = {name: param - self.inner_lr * grad if grad is not None else param for (name, param), grad in zip(theta_prime.items(), grads)}

            # After inner loop, compute KL divergence on adapted parameters if weights are not None
            if weights is not None and var is not None:  # Ensure weights and var are used in the computation
                kl_loss += self.kl_divergence(weights, torch.log(var))

            # Outer loop: Compute meta-loss on the query set using adapted parameters
            logits_meta, _, _ = self.forward(query_x, query_modality_missing)
            meta_loss += F.cross_entropy(logits_meta, query_y)

        # Combine meta-loss with KL divergence loss
        total_loss = meta_loss / len(data_loader) + kl_loss / len(data_loader)
        print(total_loss)

        # Backward pass and optimization step for meta-parameters
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        return meta_loss, kl_loss


    def train(self, data_loader, device, epochs):
        for epoch in range(epochs):
            meta_loss, kl_loss = self.meta_train_step(data_loader, device)
            print(f"Epoch {epoch+1}/{epochs} completed. Meta Loss: {meta_loss.item()}, KL Loss: {kl_loss.item()}")

### 5. **Modality Priors Initialization**

def initialize_modality_priors(data, num_priors):
    """
    Initialize modality priors using K-means clustering on complete modality data.
    """
    kmeans = KMeans(n_clusters=num_priors)
    kmeans.fit(data)
    modality_priors = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
    return modality_priors

### 6. **Dataset and DataLoader**

class MultimodalDataset(Dataset):
    def __init__(self, data, labels, modality_missing_ratio):
        self.data = data
        self.labels = labels
        self.modality_missing_ratio = modality_missing_ratio
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        modality_missing = torch.rand(1).item() < self.modality_missing_ratio
        return x, y, modality_missing

### 7. **Training Loop**

def train_smil_model(data, labels, device, epochs=20, num_priors=10, inner_steps=5):
    # Define model parameters
    input_dim = data.shape[1]
    hidden_dim = 64
    output_dim = len(set(labels))
    
    # Initialize modality priors using K-means
    modality_priors = initialize_modality_priors(data, num_priors).to(device)
    
    # Create dataset and data loader
    dataset = MultimodalDataset(data, labels, modality_missing_ratio=0.9)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Initialize model
    model = SMILModel(input_dim, hidden_dim, output_dim, num_priors, modality_priors, inner_lr=1e-3, outer_lr=1e-2, inner_steps=inner_steps).to(device)

    # Train model
    model.train(data_loader, device, epochs)

    return model

### 8. **Example Usage**

# Simulated example
if __name__ == "__main__":
    # Simulated data for demonstration
    data = torch.randn(1000, 50)  # 1000 samples, 50-dimensional input
    labels = torch.randint(0, 2, (1000,))  # Binary classification

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Train the SMIL model
    trained_model = train_smil_model(data, labels, device)
