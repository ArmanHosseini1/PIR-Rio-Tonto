import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import os
import rasterio as rio
from rasterio.plot import reshape_as_image, reshape_as_raster
from pathlib import Path
from osgeo import ogr, gdal
import matplotlib.pyplot as plt

def apply_mask(image):
    mask_norm = np.all((image >= 0) & (image <= 1), axis=2)
    mask_zero = np.all(image == 0, axis=2)
    image[(~mask_norm) | (mask_zero)] = np.nan
    image = reshape_as_raster(image)
    return image

print("test")
image_path = r'C:\Users\arman\Desktop\INSA Toulouse\5A\PIR\Données\SENTINEL2B_20240410-112055-466_L2A_T29SQB_C_V3-1_FRE_extrait_stack_gain_VNIR10.img'
raster_img = rio.open(image_path)
arr_img = reshape_as_image(raster_img.read())
image = apply_mask(arr_img) # mask to delete bad pixels

l, m, n = image.shape # check the size of the image 
print("Size of the image is "+str(l)+"x"+str(m)+"x"+str(n))

# 1️⃣ On met en forme : chaque pixel devient une ligne
X = image.reshape(l, -1).T

# 2️⃣ Centrage et réduction sur les bandes
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3️⃣ Conversion en tenseur PyTorch
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

# 3️⃣ PCA (par exemple 95 % de variance expliquée)
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)

# 4️⃣ On peut reformer l'image PCA (même taille spatiale)
nb_comp = X_pca.shape[1]
image_pca = X_pca.T.reshape(nb_comp, m, n)

print("Variance expliquée : ", np.sum(pca.explained_variance_ratio_))
print(nb_comp)

"""

plt.figure(figsize=(8,8))
plt.imshow(image_pca[[2, 1, 0], :, :])  # si ton image est (H, W, 3)
plt.show()

"""

# --- 1. Paramètres ---
n_bandes = nb_comp     # nombre de canaux spectraux
img_size = 64     # hauteur et largeur (carrée ici)
latent_dim = 64   # dimension du vecteur latent
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. Définition de l'encodeur ---
class Encoder(nn.Module):
    def __init__(self, in_channels=n_bandes, latent_dim=latent_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),  # 64 -> 32
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 32 -> 16
            nn.ReLU(True),
            nn.Conv2d(64, latent_dim, kernel_size=3, stride=2, padding=1),  # 16 -> 8
            nn.ReLU(True),
            nn.Flatten(),
        )

    def forward(self, x):
        x = self.encoder(x)
        return x


# --- 3. Définition du décodeur ---
class Decoder(nn.Module):
    def __init__(self, out_channels=n_bandes, latent_dim=latent_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()  # si les pixels sont normalisés entre 0 et 1
        )

    def forward(self, x):
        x = self.decoder(x)
        return x


# --- 4. Autoencodeur complet ---
class Autoencoder(nn.Module):
    def __init__(self, n_bandes, latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(n_bandes, latent_dim)
        self.decoder = Decoder(n_bandes, latent_dim)

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed


# --- 5. Initialisation du modèle ---
autoencoder = Autoencoder(n_bandes, latent_dim).to(device)

# --- 6. Optimiseur et fonction de perte ---
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=1e-3)

# --- 7. Exemple de données factices ---
# Remplace par ton vrai dataset (normalisé entre 0 et 1)

# --- 8. Boucle d'entraînement ---
n_epochs = 50
batch_size = 16

for epoch in range(n_epochs):
    autoencoder.train()
    epoch_loss = 0.0
    
    for i in range(0, X_train.size(0), batch_size):
        batch = X_train[i:i+batch_size]
        optimizer.zero_grad()
        outputs = autoencoder(batch)
        loss = criterion(outputs, batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    print(f"Epoch [{epoch+1}/{n_epochs}] - Loss: {epoch_loss / len(X_train):.6f}")

# --- 9. Test sur une image ---
autoencoder.eval()
with torch.no_grad():
    test_img = X_train[0:1]
    encoded = autoencoder.encoder(test_img)
    decoded = autoencoder.decoder(encoded)

print("Encodage latent :", encoded.shape)
print("Image reconstruite :", decoded.shape)