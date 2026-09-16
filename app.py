from dotenv import load_dotenv
load_dotenv()

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import uuid
import tempfile
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend não-interativo para servidores
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import hmac
import hashlib
import time
from functools import wraps
from PIL import Image

from tifffile import imread
from skimage import color
from skimage.measure import regionprops_table
from stardist.models import StarDist2D
from stardist.plot import render_label
from stardist import random_label_cmap
from csbdeep.data import Normalizer, normalize_mi_ma
import pandas as pd

import azure_storage

# Configuração
app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50 MB

# Configurações de Segurança e Senha de Acesso
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123").strip()
SECRET_KEY = os.getenv("SECRET_KEY", "nuclei-secret-salt-key-2026").strip()

def generate_token():
    """Gera um token assinado com timestamp para a sessão do usuário."""
    timestamp = str(int(time.time()))
    sig = hmac.new(SECRET_KEY.encode('utf-8'), f"{ADMIN_PASSWORD}:{timestamp}".encode('utf-8'), hashlib.sha256).hexdigest()
    return f"{timestamp}:{sig}"

def verify_token(token, max_age_seconds=7 * 24 * 3600):
    """Valida a assinatura do token e verifica se não expirou (padrão: 7 dias)."""
    if not token or ':' not in token:
        return False
    try:
        timestamp_str, sig = token.split(':', 1)
        timestamp = int(timestamp_str)
        if time.time() - timestamp > max_age_seconds:
            return False
        expected_sig = hmac.new(SECRET_KEY.encode('utf-8'), f"{ADMIN_PASSWORD}:{timestamp_str}".encode('utf-8'), hashlib.sha256).hexdigest()
        return hmac.compare_digest(sig, expected_sig)
    except Exception:
        return False

def require_auth(f):
    """Decorator para proteger endpoints restritos."""
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get('Authorization', '')
        token = ''
        if auth_header.startswith('Bearer '):
            token = auth_header[7:].strip()
        if not verify_token(token):
            return jsonify({'error': 'Acesso não autorizado. Faça login para continuar.'}), 401
        return f(*args, **kwargs)
    return decorated

ALLOWED_EXTENSIONS = {'tif', 'tiff', 'png', 'jpg', 'jpeg'}

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({'error': 'Arquivo muito grande. O limite máximo permitido é 50 MB.'}), 413


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


PREDICTION_TILE_SIZE = 1024


def get_prediction_tiles(image_shape):
    """Calcula a quantidade de tiles 2D conforme o tamanho da imagem."""
    height, width = image_shape[:2]
    return (
        max(1, int(np.ceil(height / PREDICTION_TILE_SIZE))),
        max(1, int(np.ceil(width / PREDICTION_TILE_SIZE))),
        1,
    )


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_image_as_numpy(filepath_or_file, filename):
    """
    Carrega imagens nos formatos TIFF, PNG, JPG, JPEG e retorna como array NumPy (Y, X, C).
    """
    ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
    if ext in ['tif', 'tiff']:
        img = imread(filepath_or_file)
    else:
        pil_img = Image.open(filepath_or_file)
        if pil_img.mode != 'RGB':
            pil_img = pil_img.convert('RGB')
        img = np.array(pil_img)
    
    # Normaliza dimensões caso a imagem seja 2D (escala de cinza) ou 4D (RGBA)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    elif img.ndim == 3 and img.shape[-1] == 4:
        img = img[..., :3]
        
    return img


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
        'azure_storage_configured': azure_storage.is_azure_configured(),
        'available_images': list(AVAILABLE_IMAGES.keys())
    })


@app.route('/api/login', methods=['POST'])
def login():
    """Valida a senha de acesso e retorna o token de autenticação."""
    data = request.get_json() or {}
    password = data.get('password', '')
    if not password or not hmac.compare_digest(password, ADMIN_PASSWORD):
        return jsonify({'error': 'Senha incorreta. Tente novamente.'}), 401
    
    token = generate_token()
    return jsonify({'success': True, 'token': token})


@app.route('/api/verify-token', methods=['GET'])
def verify_session():
    """Verifica se o token da sessão atual é válido."""
    auth_header = request.headers.get('Authorization', '')
    token = auth_header[7:].strip() if auth_header.startswith('Bearer ') else ''
    if verify_token(token):
        return jsonify({'valid': True})
    return jsonify({'valid': False, 'error': 'Token inválido ou expirado'}), 401


@app.route('/api/images', methods=['GET'])
@require_auth
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
@require_auth
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
        
        # Configurar predição em tiles para controlar o uso de memória
        normalizer = MyNormalizer(0, 255)
        n_tiles = get_prediction_tiles(img.shape)
        print(f"Realizando segmentação com tiles: n_tiles={n_tiles}...")
        labels, _ = model.predict_instances(
            img,
            axes="YXC",
            normalizer=normalizer,
            n_tiles=n_tiles,
        )

        
        nuclei_count = int(labels.max())
        print(f"Núcleos detectados: {nuclei_count}")
        
        # Análise das propriedades
        print("Calculando propriedades...")
        props = regionprops_table(labels, img, properties=[
            'label', 'area', 'equivalent_diameter', 'solidity', 'centroid'
        ])
        df = pd.DataFrame(props)
        
        # Gerar imagens de resultado e CSV
        safe_name = image_name.lower().replace(" ", "_")
        result_filename = f"{safe_name}_result.png"
        histogram_filename = f"{safe_name}_histogram.png"
        csv_filename = f"{safe_name}_nuclei.csv"
        
        result_local_path = os.path.join(RESULTS_FOLDER, result_filename)
        histogram_local_path = os.path.join(RESULTS_FOLDER, histogram_filename)
        
        print("Gerando visualizações...")
        generate_result_image(img_gray, labels, result_local_path)
        generate_histogram(df['area'].values, histogram_local_path)
        
        # Upload para Azure Blob Storage (ou fallback local)
        result_image_url = azure_storage.upload_file(result_local_path, result_filename, content_type='image/png')
        histogram_url = azure_storage.upload_file(histogram_local_path, histogram_filename, content_type='image/png')
        
        csv_bytes = df.to_csv(index=False).encode('utf-8')
        csv_download_url = azure_storage.upload_bytes(csv_bytes, csv_filename, content_type='text/csv')
        
        print("Concluído!")
        print(f"{'='*50}\n")
        
        # Preparar resposta
        response = {
            'success': True,
            'image_name': image_name,
            'nuclei_count': nuclei_count,
            'statistics': {
                'mean_area': round(float(df['area'].mean()), 2) if len(df) > 0 else 0,
                'median_area': round(float(df['area'].median()), 2) if len(df) > 0 else 0,
                'min_area': int(df['area'].min()) if len(df) > 0 else 0,
                'max_area': int(df['area'].max()) if len(df) > 0 else 0,
                'std_area': round(float(df['area'].std()), 2) if len(df) > 0 else 0,
                'mean_solidity': round(float(df['solidity'].mean()), 4) if len(df) > 0 else 0,
                'mean_diameter': round(float(df['equivalent_diameter'].mean()), 2) if len(df) > 0 else 0,
            },
            'result_image_url': result_image_url,
            'histogram_url': histogram_url,
            'csv_download_url': csv_download_url,
            # Enviar apenas os primeiros 100 núcleos para não sobrecarregar
            'nuclei_data': df.head(100).to_dict(orient='records')
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Erro: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/upload-and-segment', methods=['POST'])
@require_auth
def upload_and_segment():

    """
    Recebe uma imagem enviada pelo usuário (TIFF, PNG, JPG, JPEG),
    executa a segmentação de núcleos com StarDist, gera métricas e gráficos,
    armazena os resultados no Azure Blob Storage (ou fallback local),
    e remove imediatamente a imagem original do servidor.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'Nenhum arquivo enviado no campo "file"'}), 400
        
    file = request.files['file']
    
    if not file or file.filename == '':
        return jsonify({'error': 'Nome de arquivo inválido ou vazio'}), 400
        
    if not allowed_file(file.filename):
        allowed_str = ', '.join(sorted(ALLOWED_EXTENSIONS))
        return jsonify({'error': f'Formato de arquivo não suportado. Formatos aceitos: {allowed_str}'}), 400
        
    temp_filepath = None
    try:
        # Gerar identificador único
        file_id = uuid.uuid4().hex[:10]
        _, ext = os.path.splitext(file.filename)
        
        # Salvar em arquivo temporário
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
            file.save(temp_file.name)
            temp_filepath = temp_file.name
            
        print(f"\n{'='*50}")
        print(f"Processando arquivo de upload: {file.filename} (ID: {file_id})")
        print(f"{'='*50}")
        
        # Carregar imagem como matriz NumPy
        print("Carregando imagem...")
        img = load_image_as_numpy(temp_filepath, file.filename)
        img_gray = color.rgb2gray(img)
        print(f"Dimensões: {img.shape}")
        
        # Configurar predição em tiles para controlar o uso de memória
        normalizer = MyNormalizer(0, 255)
        n_tiles = get_prediction_tiles(img.shape)

        # Predição com StarDist
        print(f"Realizando segmentação StarDist com tiles: n_tiles={n_tiles}...")
        labels, _ = model.predict_instances(
            img,
            axes="YXC",
            normalizer=normalizer,
            n_tiles=n_tiles,
        )

        
        nuclei_count = int(labels.max())
        print(f"Núcleos detectados: {nuclei_count}")
        
        # Cálculo de propriedades morfométricas
        print("Calculando propriedades morfométricas...")
        props = regionprops_table(labels, img, properties=[
            'label', 'area', 'equivalent_diameter', 'solidity', 'centroid'
        ])
        df = pd.DataFrame(props)
        
        # Nomes dos arquivos de saída
        result_filename = f"{file_id}_result.png"
        histogram_filename = f"{file_id}_histogram.png"
        csv_filename = f"{file_id}_nuclei_data.csv"
        
        result_local_path = os.path.join(RESULTS_FOLDER, result_filename)
        histogram_local_path = os.path.join(RESULTS_FOLDER, histogram_filename)
        
        # Gerar visualizações
        print("Gerando gráficos e overlay...")
        generate_result_image(img_gray, labels, result_local_path)
        generate_histogram(df['area'].values, histogram_local_path)
        
        # Enviar saídas para Azure Blob Storage ou disco local
        result_image_url = azure_storage.upload_file(result_local_path, result_filename, content_type='image/png')
        histogram_url = azure_storage.upload_file(histogram_local_path, histogram_filename, content_type='image/png')
        
        csv_bytes = df.to_csv(index=False).encode('utf-8')
        csv_download_url = azure_storage.upload_bytes(csv_bytes, csv_filename, content_type='text/csv')
        
        print("Processamento concluído com sucesso!")
        print(f"{'='*50}\n")
        
        # Resposta JSON
        response = {
            'success': True,
            'image_name': file.filename,
            'file_id': file_id,
            'nuclei_count': nuclei_count,
            'statistics': {
                'mean_area': round(float(df['area'].mean()), 2) if len(df) > 0 else 0,
                'median_area': round(float(df['area'].median()), 2) if len(df) > 0 else 0,
                'min_area': int(df['area'].min()) if len(df) > 0 else 0,
                'max_area': int(df['area'].max()) if len(df) > 0 else 0,
                'std_area': round(float(df['area'].std()), 2) if len(df) > 0 else 0,
                'mean_solidity': round(float(df['solidity'].mean()), 4) if len(df) > 0 else 0,
                'mean_diameter': round(float(df['equivalent_diameter'].mean()), 2) if len(df) > 0 else 0,
            },
            'result_image_url': result_image_url,
            'histogram_url': histogram_url,
            'csv_download_url': csv_download_url,
            'nuclei_data': df.head(100).to_dict(orient='records')
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Erro no processamento do upload: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
        
    finally:
        # Exclusão imediata da imagem original enviada
        if temp_filepath and os.path.exists(temp_filepath):
            try:
                os.remove(temp_filepath)
                print(f"Arquivo temporário de entrada excluído: {temp_filepath}")
            except Exception as cleanup_err:
                print(f"Aviso ao remover arquivo temporário: {cleanup_err}")


@app.route('/static/results/<path:filename>')
def serve_result(filename):
    """Serve as imagens de resultado geradas (com suporte a download forçado)."""
    as_attachment = request.args.get('download', 'false').lower() == 'true'
    return send_from_directory(RESULTS_FOLDER, filename, as_attachment=as_attachment)


@app.route('/api/download/<path:filename>', methods=['GET'])
def download_result_file(filename):
    """Rota utilitária para forçar o download de arquivos de resultado (CSV ou PNG)."""
    return send_from_directory(RESULTS_FOLDER, filename, as_attachment=True)



# Endpoint alternativo que retorna imagem como Base64 (opcional)
@app.route('/api/segment-base64', methods=['POST'])
@require_auth
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
        n_tiles = get_prediction_tiles(img.shape)

        labels, _ = model.predict_instances(
            img,
            axes="YXC",
            normalizer=normalizer,
            n_tiles=n_tiles,
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
