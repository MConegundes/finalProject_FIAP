"""
Pipeline RAG (Retrieval-Augmented Generation) para enriquecer
"""

import logging
from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

logger = logging.getLogger(__name__)

CHROMA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "rag_database"
DOCS_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "documents"


def build_vector_store(
    docs_dir: str | Path | None = None,
    persist_dir: str | Path | None = None,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> Chroma:
    """Constrói ou carrega vector store a partir de documentos.

    Args:
        docs_dir: Diretório com documentos de contexto.
        persist_dir: Diretório para persistir o ChromaDB.
        chunk_size: Tamanho dos chunks de texto.
        chunk_overlap: Sobreposição entre chunks.
    """
    docs_dir = Path(docs_dir) if docs_dir else DOCS_DIR
    persist_dir = Path(persist_dir) if persist_dir else CHROMA_DIR

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
    )

    if persist_dir.exists() and any(persist_dir.iterdir()):
        return Chroma(
            persist_directory=str(persist_dir),
            embedding_function=embeddings,
        )

    if not docs_dir.exists():
        docs_dir.mkdir(parents=True, exist_ok=True)
        return Chroma(
            persist_directory=str(persist_dir),
            embedding_function=embeddings,
        )

    loader = DirectoryLoader(str(docs_dir), glob="**/*.txt", loader_cls=TextLoader)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(documents)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(persist_dir),
    )

    return vectorstore


def retrieve_context(query: str, k: int = 3) -> list[str]:
    """Busca contextos relevantes para uma query.

    Args:
        query: Pergunta do usuário.
        k: Número de documentos a retornar.

    Returns:
        Lista de textos relevantes.
    """
    vectorstore = build_vector_store()
    docs = vectorstore.similarity_search(query, k=k)
    return [doc.page_content for doc in docs]
