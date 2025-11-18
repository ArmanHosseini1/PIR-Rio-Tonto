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
from sklearn.cluster import KMeans
from matplotlib.colors import ListedColormap
from kneed import KneeLocator
from sklearn.metrics import silhouette_score

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

X = image.reshape(l, -1).T

scaling = 1
if (scaling==1):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
else:
    X_scaled = X

comparison = 0

if (comparison==1):

        # --- Liste des k à tester
    k_values = [10, 20, 30, 40, 50]

    # --- Définir une colormap fixe (ex : 10 couleurs maximum)
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    global_cmap = ListedColormap(colors)

    # --- Créer les segmentations pour chaque k
    labels_images = []

    for k in range(len(k_values)):
        kmeans = KMeans(n_clusters=k_values[k], init="k-means++",  # par défaut, mais utile à expliciter
        n_init=5,         # plus de répétitions -> plus stable
        max_iter=500,
        random_state=42)
        labels = kmeans.fit_predict(X_scaled)
        labels_images += [labels.reshape(m, n)]

        plt.imshow(labels_images[k], cmap='tab10')
        plt.title(f"Classification K-Means (k={k_values[k]})")
        plt.axis('off')
        plt.show()

else:

    inertias = []
    #sil_scores = []

    K_range = range(2, 40, 5)

    """
    sample_size = min(10000, X_scaled.shape[0])
    sample_idx = np.random.choice(X_scaled.shape[0], sample_size, replace=False)
    X_sample = X_scaled[sample_idx]
    """

    for k in K_range:
        kmeans = KMeans(n_clusters=k, init='k-means++', n_init=3, random_state=42)
        kmeans.fit(X_scaled)
        #labels = kmeans.fit_predict(X_scaled)
        #labels_sample = labels[sample_idx]
        #sil = silhouette_score(X_sample, labels_sample)
        #sil_scores.append(sil)
        inertias.append(kmeans.inertia_)
        print("Done")

    plt.plot(K_range, inertias, marker='o')
    #plt.plot(K_range, sil_scores, marker='o')
    plt.xlabel('Nombre de clusters k')
    plt.ylabel('Inertie (somme des distances)')
    #plt.ylabel('Score silhouette')
    plt.title('Méthode du coude')
    plt.show()

    knee = KneeLocator(K_range, inertias, curve='convex', direction='decreasing')
    optimal_k = knee.knee
    #optimal_k = K_range[np.argmax(sil_scores)]

    print(f"k optimal trouvé = {optimal_k}")

    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    labels = kmeans.fit_predict(X_scaled)

        # Reformater les labels dans la forme spatiale de l’image
    labels_image = labels.reshape(m, n)

    plt.imshow(labels_image, cmap='tab10')
    plt.title(f"Classification K-Means (k={optimal_k})")
    plt.axis('off')
    plt.show()