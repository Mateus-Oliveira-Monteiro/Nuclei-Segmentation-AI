"""
API Flask para Segmentação de Núcleos usando StarDist
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend não-interativo para servidores
import matplotlib.pyplot as plt
from io import BytesIO
import base64

from tifffile import imread
from skimage import color
from skimage.measure import regionprops_table
from stardist.models import StarDist2D
from stardist.plot import render_label
from stardist import random_label_cmap
from csbdeep.data import Normalizer, normalize_mi_ma
import pandas as pd

# Configuração
app = Flask(__name__)
CORS(app)

IMAGE_FOLDER = "tiff images"
RESULTS_FOLDER = "static/results"

# Criar pasta de resultados se não existir
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Imagens disponíveis
AVAILABLE_IMAGES = {
    "Campo 1": "Campo 1.tif",
    "Campo 2": "Campo 2.tif",
    "Campo 3": "Campo 3.tif",
    "Campo 3R": "Campo 3R.tif",
}

# Carregar modelo uma vez na inicialização
print("=" * 50)
print("Carregando modelo StarDist...")
print("=" * 50)
model = StarDist2D.from_pretrained('2D_versatile_he')
print("Modelo carregado com sucesso!")
print("=" * 50)


# Normalizador customizado
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


def generate_result_image(img_gray, labels, output_path):
    """Gera e salva a imagem com overlay dos núcleos detectados."""
    cmap = random_label_cmap()
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Imagem original
    axes[0].imshow(img_gray, cmap='gray')
    axes[0].set_title('Imagem Original', fontsize=14)
    axes[0].axis('off')
    
    # Imagem com overlay
    axes[1].imshow(render_label(labels, img=img_gray))
    axes[1].set_title(f'Núcleos Detectados: {labels.max()}', fontsize=14)
    axes[1].axis('off')
    
    # Apenas os labels coloridos
    axes[2].imshow(labels, cmap=cmap)
    axes[2].set_title('Mapa de Segmentação', fontsize=14)
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def generate_histogram(areas, output_path):
    """Gera histograma da distribuição de áreas."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(areas, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
    ax.set_xlabel('Área (pixels)', fontsize=12)
    ax.set_ylabel('Contagem', fontsize=12)
    ax.set_title('Distribuição das Áreas dos Núcleos', fontsize=14)
    ax.set_xscale('log')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)


@app.route('/api/health', methods=['GET'])
def health():
    """Verifica se a API está funcionando."""
    return jsonify({
        'status': 'ok',
        'model_loaded': True,
        'available_images': list(AVAILABLE_IMAGES.keys())
    })


@app.route('/api/images', methods=['GET'])
def list_images():
    """Lista as imagens disponíveis para análise."""
    images = []
    for name, filename in AVAILABLE_IMAGES.items():
        filepath = os.path.join(IMAGE_FOLDER, filename)
        exists = os.path.exists(filepath)
        images.append({
            'id': name,
            'filename': filename,
            'available': exists
        })
    
    return jsonify({'images': images})


@app.route('/api/segment', methods=['POST'])
def segment():
    """
    Realiza a segmentação de núcleos na imagem selecionada.
    
    Body JSON:
        {
            "image_name": "Campo 1" | "Campo 2" | "Campo 3" | "Campo 3R"
        }
    
    Retorna:
        {
            "success": true,
            "image_name": "Campo 3",
            "nuclei_count": 226,
            "statistics": { ... },
            "result_image_url": "/static/results/campo_3_result.png",
            "histogram_url": "/static/results/campo_3_histogram.png",
            "nuclei_data": [ ... ]
        }
    """
    data = request.get_json()
    
    if not data or 'image_name' not in data:
        return jsonify({
            'error': 'Campo "image_name" é obrigatório',
            'available_images': list(AVAILABLE_IMAGES.keys())
        }), 400
    
    image_name = data['image_name']
    
    if image_name not in AVAILABLE_IMAGES:
        return jsonify({
            'error': f'Imagem "{image_name}" não encontrada',
            'available_images': list(AVAILABLE_IMAGES.keys())
        }), 404
    
    filename = AVAILABLE_IMAGES[image_name]
    filepath = os.path.join(IMAGE_FOLDER, filename)
    
    if not os.path.exists(filepath):
        return jsonify({
            'error': f'Arquivo {filename} não existe no servidor'
        }), 404
    
    try:
        print(f"\n{'='*50}")
        print(f"Processando: {image_name}")
        print(f"{'='*50}")
        
        # Carregar imagem
        print("Carregando imagem...")
        img = imread(filepath)
        img_gray = color.rgb2gray(img)
        
        print(f"Dimensões: {img.shape}")
        
        # Configurar predição
        normalizer = MyNormalizer(0, 255)
        block_size = min(img.shape[0], img.shape[1], 4096)
        min_overlap = int(block_size * 0.1)
        context = int(block_size * 0.1)
        
        # Realizar predição
        print("Realizando segmentação (isso pode demorar)...")
        labels, polys = model.predict_instances_big(
            img,
            axes="YXC",
            block_size=block_size,
            min_overlap=min_overlap,
            context=context,
            normalizer=normalizer,
            n_tiles=(4, 4, 1),
        )
        
        nuclei_count = int(labels.max())
        print(f"Núcleos detectados: {nuclei_count}")
        
        # Análise das propriedades
        print("Calculando propriedades...")
        props = regionprops_table(labels, img, properties=[
            'label', 'area', 'equivalent_diameter', 'solidity', 'centroid'
        ])
        df = pd.DataFrame(props)
        
        # Gerar imagens de resultado
        safe_name = image_name.lower().replace(" ", "_")
        result_image_path = os.path.join(RESULTS_FOLDER, f"{safe_name}_result.png")
        histogram_path = os.path.join(RESULTS_FOLDER, f"{safe_name}_histogram.png")
        
        print("Gerando visualizações...")
        generate_result_image(img_gray, labels, result_image_path)
        generate_histogram(df['area'].values, histogram_path)
        
        print("Concluído!")
        print(f"{'='*50}\n")
        
        # Preparar resposta
        response = {
            'success': True,
            'image_name': image_name,
            'nuclei_count': nuclei_count,
            'statistics': {
                'mean_area': round(float(df['area'].mean()), 2),
                'median_area': round(float(df['area'].median()), 2),
                'min_area': int(df['area'].min()),
                'max_area': int(df['area'].max()),
                'std_area': round(float(df['area'].std()), 2),
                'mean_solidity': round(float(df['solidity'].mean()), 4),
                'mean_diameter': round(float(df['equivalent_diameter'].mean()), 2),
            },
            'result_image_url': f"/{result_image_path}",
            'histogram_url': f"/{histogram_path}",
            # Enviar apenas os primeiros 100 núcleos para não sobrecarregar
            'nuclei_data': df.head(100).to_dict(orient='records')
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Erro: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/static/results/<path:filename>')
def serve_result(filename):
    """Serve as imagens de resultado geradas."""
    return send_from_directory(RESULTS_FOLDER, filename)


# Endpoint alternativo que retorna imagem como Base64 (opcional)
@app.route('/api/segment-base64', methods=['POST'])
def segment_base64():
    """
    Mesmo que /api/segment, mas retorna as imagens como Base64.
    Útil se não quiser servir arquivos estáticos.
    """
    data = request.get_json()
    
    if not data or 'image_name' not in data:
        return jsonify({
            'error': 'Campo "image_name" é obrigatório'
        }), 400
    
    image_name = data['image_name']
    
    if image_name not in AVAILABLE_IMAGES:
        return jsonify({'error': 'Imagem não encontrada'}), 404
    
    filename = AVAILABLE_IMAGES[image_name]
    filepath = os.path.join(IMAGE_FOLDER, filename)
    
    try:
        # Carregar e processar
        img = imread(filepath)
        img_gray = color.rgb2gray(img)
        
        normalizer = MyNormalizer(0, 255)
        block_size = min(img.shape[0], img.shape[1], 4096)
        
        labels, polys = model.predict_instances_big(
            img,
            axes="YXC",
            block_size=block_size,
            min_overlap=int(block_size * 0.1),
            context=int(block_size * 0.1),
            normalizer=normalizer,
            n_tiles=(4, 4, 1),
        )
        
        # Gerar imagem em memória
        cmap = random_label_cmap()
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        axes[0].imshow(img_gray, cmap='gray')
        axes[0].set_title('Original')
        axes[0].axis('off')
        
        axes[1].imshow(render_label(labels, img=img_gray))
        axes[1].set_title(f'Núcleos: {labels.max()}')
        axes[1].axis('off')
        
        plt.tight_layout()
        
        # Converter para Base64
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        
        # Análise
        props = regionprops_table(labels, img, properties=['label', 'area'])
        df = pd.DataFrame(props)
        
        return jsonify({
            'success': True,
            'nuclei_count': int(labels.max()),
            'result_image_base64': f'data:image/png;base64,{img_base64}',
            'statistics': {
                'mean_area': round(float(df['area'].mean()), 2),
                'min_area': int(df['area'].min()),
                'max_area': int(df['area'].max()),
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("API de Segmentação de Núcleos")
    print("=" * 50)
    print(f"Imagens disponíveis: {list(AVAILABLE_IMAGES.keys())}")
    print("Endpoints:")
    print("  GET  /api/health   - Status da API")
    print("  GET  /api/images   - Lista imagens disponíveis")
    print("  POST /api/segment  - Realiza segmentação")
    print("=" * 50 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
