"""
Módulo de gerenciamento de armazenamento para o Nuclei Segmentation AI.
Suporta Azure Blob Storage e fallback gracioso para armazenamento local.
"""

import os
import logging
import shutil

logger = logging.getLogger(__name__)

AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "").strip()
AZURE_CONTAINER_NAME = os.getenv("AZURE_CONTAINER_NAME", "nuclei-results").strip()
LOCAL_RESULTS_FOLDER = "static/results"

# Garantir que a pasta local exista sempre
os.makedirs(LOCAL_RESULTS_FOLDER, exist_ok=True)

_blob_service_client = None
_container_client = None

def get_azure_container():
    """Obtém ou cria o container no Azure Blob Storage se configurado."""
    global _blob_service_client, _container_client
    
    if not AZURE_STORAGE_CONNECTION_STRING:
        return None
        
    if _container_client is not None:
        return _container_client

    try:
        from azure.storage.blob import BlobServiceClient, PublicAccess
        _blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
        _container_client = _blob_service_client.get_container_client(AZURE_CONTAINER_NAME)
        
        if not _container_client.exists():
            # Cria o container com acesso de leitura público para os blobs
            _container_client = _blob_service_client.create_container(
                AZURE_CONTAINER_NAME,
                public_access=PublicAccess.Blob
            )
            logger.info(f"Container '{AZURE_CONTAINER_NAME}' criado com sucesso no Azure Blob Storage.")
        return _container_client
    except Exception as e:
        logger.warning(f"Não foi possível conectar ao Azure Blob Storage: {e}. Usando fallback local.")
        return None


def is_azure_configured():
    """Retorna True se o Azure Blob Storage estiver disponível e configurado."""
    return get_azure_container() is not None


def upload_file(local_path, blob_name, content_type=None):
    """
    Faz o upload de um arquivo para o Azure Blob Storage ou copia para static/results.
    Retorna a URL de acesso ao arquivo (seja URL do Azure ou caminho estático da API).
    """
    container = get_azure_container()
    
    if container is not None:
        try:
            from azure.storage.blob import ContentSettings
            blob_client = container.get_blob_client(blob_name)
            
            content_settings = ContentSettings(content_type=content_type) if content_type else None
            
            with open(local_path, "rb") as data:
                blob_client.upload_blob(data, overwrite=True, content_settings=content_settings)
            
            logger.info(f"Upload concluído no Azure: {blob_client.url}")
            return blob_client.url
        except Exception as e:
            logger.error(f"Falha no upload para o Azure: {e}. Salvando localmente como fallback.")
    
    # Fallback local
    local_dest = os.path.join(LOCAL_RESULTS_FOLDER, blob_name)
    if os.path.abspath(local_path) != os.path.abspath(local_dest):
        shutil.copy2(local_path, local_dest)
    
    return f"/static/results/{blob_name}"


def upload_bytes(data_bytes, blob_name, content_type=None):
    """
    Faz upload de bytes em memória para o Azure Blob Storage ou salva no disco local.
    Retorna a URL de acesso.
    """
    container = get_azure_container()
    
    if container is not None:
        try:
            from azure.storage.blob import ContentSettings
            blob_client = container.get_blob_client(blob_name)
            content_settings = ContentSettings(content_type=content_type) if content_type else None
            
            blob_client.upload_blob(data_bytes, overwrite=True, content_settings=content_settings)
            logger.info(f"Upload de bytes concluído no Azure: {blob_client.url}")
            return blob_client.url
        except Exception as e:
            logger.error(f"Falha no upload para o Azure: {e}. Salvando localmente como fallback.")
            
    # Fallback local
    local_dest = os.path.join(LOCAL_RESULTS_FOLDER, blob_name)
    with open(local_dest, "wb") as f:
        f.write(data_bytes)
        
    return f"/static/results/{blob_name}"
