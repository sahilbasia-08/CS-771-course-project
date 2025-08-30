import torch
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

# ----------------------------
# Load CIFAR-10 Dataset
# ----------------------------
transform = transforms.Compose([transforms.ToTensor()])

trainset = torchvision.datasets.CIFAR10(root="./dataset/part_one_dataset/train_data", train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root="./dataset/part_one_dataset/eval_data", train=False, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=1000, shuffle=False, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False, num_workers=2)

def flatten_dataset(dataloader):
    features, labels = [], []
    for imgs, labs in tqdm(dataloader):
        flat = imgs.view(imgs.size(0), -1)  # [B, 3072]
        features.append(flat)
        labels.append(labs)
    return torch.cat(features, dim=0), torch.cat(labels, dim=0)

print("Flattening train set...")
X_train, y_train = flatten_dataset(trainloader)
print("Flattening test set...")
X_test, y_test = flatten_dataset(testloader)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

# ----------------------------
# PCA for dimensionality reduction
# ----------------------------
k = 50  # number of principal components
print(f"Running PCA (k={k})...")

mean = X_train.mean(dim=0, keepdim=True)
X_centered = X_train - mean
U, S, Vt = torch.linalg.svd(X_centered.cpu(), full_matrices=False)
V = Vt[:k].T.to(device)  # top-k components

X_train_pca = (X_train - mean.to(device)) @ V
X_test_pca = (X_test - mean.to(device)) @ V

# ----------------------------
# Mahalanobis Whitening in PCA space
# ----------------------------
print("Computing covariance in PCA space...")
mean_pca = X_train_pca.mean(dim=0, keepdim=True)
Xc_pca = X_train_pca - mean_pca
cov_pca = (Xc_pca.T @ Xc_pca) / (Xc_pca.size(0) - 1)

eigvals, eigvecs = torch.linalg.eigh(cov_pca.cpu())
eigvals = torch.clamp(eigvals, min=1e-5)
inv_sqrt = eigvecs @ torch.diag(eigvals.rsqrt()) @ eigvecs.T
inv_sqrt = inv_sqrt.to(device)

X_train_whiten = (X_train_pca - mean_pca.to(device)) @ inv_sqrt
X_test_whiten = (X_test_pca - mean_pca.to(device)) @ inv_sqrt

# ----------------------------
# LwP (Mahalanobis via whitened Euclidean)
# ----------------------------
def lwp_mahalanobis_gpu(X_train, y_train, X_test, batch_size=500):
    preds = []
    for i in tqdm(range(0, len(X_test), batch_size)):
        xb = X_test[i:i+batch_size]
        dists = torch.cdist(xb, X_train)  # Euclidean after whitening
        nn_idx = torch.argmin(dists, dim=1)
        preds.append(y_train[nn_idx].cpu())
    return torch.cat(preds)

print("Running LwP (Mahalanobis + PCA) on GPU...")
y_pred = lwp_mahalanobis_gpu(X_train_whiten, y_train, X_test_whiten, batch_size=200)

# ----------------------------
# Accuracy
# ----------------------------
acc = (y_pred == y_test.cpu()).float().mean().item() * 100
print(f"✅ LwP (Mahalanobis + PCA {k}) Accuracy: {acc:.2f}%")
