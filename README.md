# Nuclei Segmentation AI

## Visao geral
Este projeto oferece uma API Flask para segmentacao de nucleos em imagens TIFF usando o modelo StarDist2D pre-treinado. Alem da API, ha um script local para analise offline e geracao de estatisticas em CSV.

Principais capacidades:
- Segmentacao de nucleos com StarDist2D (modelo 2D_versatile_he).
- Geracao de overlay, mapa de rotulos e histograma de areas.
- Estatisticas basicas (area, diametro equivalente, solidez).
- Endpoint opcional que retorna imagem em Base64.

## Como funciona (fluxo)
1. A imagem TIFF e carregada do diretorio local.
2. A versao em escala de cinza e usada para visualizacao.
3. A normalizacao de intensidades e aplicada via `MyNormalizer`.
4. A segmentacao roda em blocos (`predict_instances_big`) para lidar com imagens grandes.
5. As propriedades dos nucleos sao calculadas com `regionprops_table`.
6. As visualizacoes sao salvas em `static/results`.
7. A API retorna JSON com contagem, estatisticas e uma amostra de nucleos.

## Estrutura do projeto
- app.py: API Flask com endpoints de segmentacao e visualizacao.
- nuclei_segmentation.py: script local para teste e analise offline.
- requirements.txt: dependencias do ambiente.
- procfile: comando para deploy (ajuste recomendado, ver Observacoes).
- static/results/: imagens de resultado geradas pela API.
- tiff images/: imagens TIFF de entrada.
- nuclei_analysis_results.csv: exemplo de saida do script offline.

## Requisitos
- Python compativel com TensorFlow 2.15.
- Dependencias listadas em requirements.txt.

## Instalacao
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Executar a API localmente
```bash
python app.py
```
A primeira execucao pode baixar os pesos do modelo StarDist.

Servidor padrão: `http://localhost:5000`

## Endpoints da API
### GET /api/health
Verifica status da API e lista imagens disponiveis.

Resposta (exemplo):
```json
{
  "status": "ok",
  "model_loaded": true,
  "available_images": ["Campo 1", "Campo 2", "Campo 3", "Campo 3R"]
}
```

### GET /api/images
Lista imagens registradas em `AVAILABLE_IMAGES` e indica se o arquivo existe.

Resposta (exemplo):
```json
{
  "images": [
    {"id": "Campo 1", "filename": "Campo 1.tif", "available": true}
  ]
}
```

### POST /api/segment
Executa segmentacao na imagem selecionada.

Request:
```json
{
  "image_name": "Campo 3"
}
```

Resposta (campos principais):
- `nuclei_count`: total de nucleos detectados.
- `statistics`: media, mediana, min, max, desvio, solidez media.
- `result_image_url`: caminho da imagem com overlay.
- `histogram_url`: caminho do histograma.
- `nuclei_data`: amostra dos primeiros 100 nucleos.

### POST /api/segment-base64
Igual a `/api/segment`, mas retorna a imagem com overlay em Base64.

### GET /static/results/<arquivo>
Serve as imagens geradas pela API.

## Exemplo rapido com curl
```bash
curl -X POST http://localhost:5000/api/segment \
  -H "Content-Type: application/json" \
  -d '{"image_name": "Campo 3"}'
```

## Script local de analise (offline)
O arquivo `nuclei_segmentation.py` executa uma analise local usando uma imagem especifica:
1. Ajuste `IMAGE_NAME` para a imagem desejada.
2. Rode o script:

```bash
python nuclei_segmentation.py
```

O script gera o arquivo `nuclei_analysis_results.csv` com as propriedades dos nucleos.

## Ajustes comuns
- Adicionar imagens: coloque arquivos TIFF em `tiff images/` e atualize `AVAILABLE_IMAGES`.
- Ajustar desempenho: altere `block_size`, `min_overlap` e `n_tiles`.
- Ajustar quantidade de dados retornados: modifique `df.head(100)`.

## Observacoes de deploy
- O `procfile` usa `python main.py`, mas o entrypoint atual e `app.py`.
  Se for usar esse arquivo, ajuste o comando para `python app.py` ou crie um `main.py` que exponha o app.
- Para executar com Gunicorn:
```bash
gunicorn -b 0.0.0.0:5000 app:app
```

## Limitacoes e desempenho
- Imagens muito grandes podem exigir mais memoria e tempo.
- Em CPU o processo pode ser lento; GPU acelera a segmentacao.
- A primeira execucao pode demorar por conta do download do modelo.
