import pandas as pd
import cv2
import numpy as np
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from sklearn.preprocessing import StandardScaler



df = pd.read_csv("train.csv")
df = df.drop(columns=["Unnamed: 0"])

def extract_features(path):
    img = cv2.imread(path) #leer el path
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #convertir a escala de grises
    img = cv2.resize(img, (128,128)) #Cambiar el tamaño de la imagen
    
    features = {}
    
    # Sobel
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3) #Detecta bordes verticales
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3) #Detecta bordes horizontales
    sobel_mag = np.sqrt(sobelx**2 + sobely**2) #magnitud total del gradiente
    features["sobel_mean"] = np.mean(sobel_mag)
    features["sobel_std"] = np.std(sobel_mag)
    
    # Histograma
    #Se calcula la intesidad con 32 bins que van de 0 a 255 representa la proporcion de pixeles cuya intensidad cae en ese rango.
    #los primeros bins muestran intensidades mas oscuros y los ultimos bins los intensidades mas claras
    hist = cv2.calcHist([img],[0],None,[32],[0,256]).flatten()
    hist = hist / hist.sum()
    for i in range(len(hist)):
        features[f"histograma_{i}"] = hist[i]
    
    # Haralick
    #Gray-level-Co-occurrence Matrix (GLCM), captura como se co-ocurren intensidades especificas entre pares de pixxeles a cierta distancia y direccion 
    glcm = graycomatrix(img, distances=[1], angles=[0], levels=256, symmetric=True, normed=True,)
    features["glcm_contrast"] = graycoprops(glcm, 'contrast')[0,0]
    features["glcm_homogeneity"] = graycoprops(glcm, 'homogeneity')[0,0]
    features["glcm_energy"] = graycoprops(glcm, 'energy')[0,0]
    features["glcm_correlation"] = graycoprops(glcm, 'correlation')[0,0]
    
    # Estadísticas básicas
    features["mean_intensity"] = np.mean(img)
    features["std_intensity"] = np.std(img)
    features["min_intensity"] = np.min(img)
    features["max_intensity"] = np.max(img)
    
    return features

features_list = []
for fname in df["file_name"]:
    feats = extract_features(fname)
    features_list.append(feats)

features_df = pd.DataFrame(features_list)

final_df = pd.concat([df.reset_index(drop=True), features_df], axis=1)
print(final_df.head())
