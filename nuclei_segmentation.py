# Nuclei Segmentation using StarDist
# Adaptado do Google Colab para rodar localmente

import os
import numpy as np
import matplotlib.pyplot as plt

from tifffile import imread, imwrite
from csbdeep.utils import Path, normalize
from csbdeep.utils.tf import keras_import
keras = keras_import()

from stardist.plot import render_label
from stardist import export_imagej_rois, random_label_cmap
from stardist.models import StarDist2D

from skimage import color
from skimage import io

np.random.seed(0)
cmap = random_label_cmap()

def show_image(img, **kwargs):
    """Plot large image at different resolutions."""
    fig, ax = plt.subplots(2,4, figsize=(16,8))
    mid = [s//2 + 600 for s in img.shape[:2]]
    for a,t,u in zip(ax.ravel(),[1,2,4,8,16,32,64,128],[16,8,4,2,1,1,1,1]):
        sl = tuple(slice(c - s//t//2, c + s//t//2, u) for s,c in zip(img.shape[:2],mid))
        a.imshow(img[sl], **kwargs)
        a.axis('off')
    plt.tight_layout()
    plt.show()

# Caminho local para as imagens de teste
IMAGE_FOLDER = "tiff images"
IMAGE_NAME = "Campo 3.tif"  # Imagem escolhida para teste
IMAGE_PATH = os.path.join(IMAGE_FOLDER, IMAGE_NAME)

# Listar imagens disponíveis
print("Imagens disponíveis:")
for f in os.listdir(IMAGE_FOLDER):
    print(f"  - {f}")
print(f"\nImagem selecionada para teste: {IMAGE_NAME}")

# Carregar a imagem
img = imread(IMAGE_PATH)
imgGray = color.rgb2gray(io.imread(IMAGE_PATH))

print(f"Dimensões da imagem colorida: {img.shape}")
print(f"Dimensões da imagem em escala de cinza: {imgGray.shape}")

# Mostrar a imagem em escala de cinza
plt.figure(figsize=(10, 10))
plt.imshow(imgGray, cmap="gray")
plt.title("Imagem em escala de cinza")
plt.axis('off')
plt.show()

# Carregar modelos pré-treinados
print("\nCarregando modelos pré-treinados...")
model = StarDist2D.from_pretrained('2D_versatile_he')
modelFluo = StarDist2D.from_pretrained('2D_versatile_fluo')
print("Modelos carregados com sucesso!")

from csbdeep.data import Normalizer, normalize_mi_ma

class MyNormalizer(Normalizer):
    def __init__(self, mi, ma):
            self.mi, self.ma = mi, ma
    def before(self, x, axes):
        return normalize_mi_ma(x, self.mi, self.ma, dtype=np.float32)
    def after(*args, **kwargs):
        assert False
    @property
    def do_after(self):
        return False

mi, ma = 0, 255
normalizer = MyNormalizer(mi, ma)

# Calcular block_size, min_overlap e context baseado no tamanho da imagem
print("\nIniciando predição (isso pode demorar alguns minutos)...")
block_size = min(img.shape[0], img.shape[1], 4096)
min_overlap = int(block_size * 0.1)
context = int(block_size * 0.1)

# Realizar predição
labels, polys = model.predict_instances_big(
    img,
    axes="YXC",
    block_size=block_size,
    min_overlap=min_overlap,
    context=context,
    normalizer=normalizer,
    n_tiles=(4, 4, 1),
)
print(f"Predição concluída! Foram detectados {labels.max()} núcleos.")

# Visualizar resultados
plt.figure(figsize=(12, 6))
plt.subplot(1,2,1)
plt.imshow(imgGray, cmap="gray")
plt.axis("off")
plt.title("Imagem original")

plt.subplot(1,2,2)
plt.imshow(render_label(labels, img=imgGray))
plt.axis("off")
plt.title("Predição + overlay")
plt.tight_layout()
plt.show()

show_image(labels, cmap=cmap)

# Visualização com diferentes colormaps
fig, (a,b) = plt.subplots(1,2, figsize=(16,16))
a.imshow(labels[::8,::8], cmap='tab20b')
b.imshow(labels[::8,::8], cmap=cmap)
a.axis('off')
b.axis('off')
a.set_title('Colormap: tab20b')
b.set_title('Colormap: random')
plt.tight_layout()
plt.show()

# Análise dos resultados
from skimage.measure import regionprops_table
import pandas as pd

props = regionprops_table(labels, img, properties=['label', 'area', 'equivalent_diameter', 'mean_intensity', 'solidity'])

analysis_results = pd.DataFrame(props)
print("\n" + "="*50)
print("RESULTADOS DA ANÁLISE")
print("="*50)
print(f"\nTotal de núcleos detectados: {len(analysis_results)}")
print(f"\nÚltimas 5 entradas:")
print(analysis_results.tail())

print(f"\nEstatísticas da área dos núcleos:")
print(analysis_results['area'].describe())

# Histograma da distribuição de áreas
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.histplot(x="area", data=analysis_results, bins=50, log_scale=True)
plt.title("Distribuição das áreas dos núcleos")
plt.xlabel("Área (escala log)")
plt.ylabel("Contagem")
plt.tight_layout()
plt.show()

# Salvar resultados em CSV
output_csv = "nuclei_analysis_results.csv"
analysis_results.to_csv(output_csv, index=False)
print(f"\nResultados salvos em: {output_csv}")