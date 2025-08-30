import torch
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
from sklearn.decomposition import PCA

# ----------------------------
# Load CIFAR-10 Dataset
# ----------------------------
transform = transforms.Compose([transforms.ToTensor()])

trainset = torchvision.datasets.CIFAR10(
    root="./dataset/part_one_dataset/train_data",
    train=True, download=True, transform=transform
)
testset = torchvision.datasets.CIFAR10(
    root="./dataset/part_one_dataset/eval_data",
    train=False, download=True, transform=transform
)

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

# ----------------------------
# Apply PCA (100 components)
# ----------------------------
print("Applying PCA with 100 components...")

# move to CPU for sklearn PCA
X_train_cpu = X_train.cpu().numpy()
X_test_cpu = X_test.cpu().numpy()

pca = PCA(n_components=100)
X_train_pca = pca.fit_transform(X_train_cpu)
X_test_pca = pca.transform(X_test_cpu)

# back to torch tensors
X_train = torch.tensor(X_train_pca, dtype=torch.float32)
X_test = torch.tensor(X_test_pca, dtype=torch.float32)

# Move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

# ----------------------------
# LwP (Euclidean with GPU)
# ----------------------------
def lwp_euclidean_gpu(X_train, y_train, X_test, batch_size=500):
    preds = []
    for i in tqdm(range(0, len(X_test), batch_size)):
        xb = X_test[i:i+batch_size]  # test batch
        dists = torch.cdist(xb, X_train)  # [batch, train_size]
        nn_idx = torch.argmin(dists, dim=1)  # nearest neighbor index
        preds.append(y_train[nn_idx].cpu())
    return torch.cat(preds)

print("Running LwP (Euclidean, PCA=100) on GPU...")
y_pred = lwp_euclidean_gpu(X_train, y_train, X_test, batch_size=200)

# ----------------------------
# Accuracy
# ----------------------------
acc = (y_pred == y_test.cpu()).float().mean().item() * 100
print(f"✅ LwP (Euclidean, PCA=100) Accuracy: {acc:.2f}%")
