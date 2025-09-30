import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import numpy as np
import pandas as pd
from loguru import logger
from typing import List, Dict, Tuple, Optional

class TopicVectorStore:
    
    def __init__(
        self, 
        persist_dir: str = "./chroma_db",
        model_name: str = 'paraphrase-multilingual-mpnet-base-v2'
    ):
        logger.info("Initializing ChromaDB at {}", persist_dir)
        
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )
        
        logger.info("Configuring embedding function with model: {}", model_name)
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=model_name
        )
        
        self.collection = None

    def create_topic_collection(self, collection_name: str = "topics"):

        logger.info("Creating/getting collection: {}", collection_name)
        
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"}
        )
        
        logger.info("Collection ready: {} items", self.collection.count())
        return self.collection

    def add_topics(self, topics_df: pd.DataFrame, text_column: str = 'combined_text'):

        logger.info("Adding {} topics to vector store", len(topics_df))
        
        ## prep data
        ids = topics_df['id'].tolist()
        documents = topics_df[text_column].fillna('').tolist()
        
        ## prep metadata
        metadatas = []
        for _, row in topics_df.iterrows():
            metadata = {
                'title': str(row.get('title', ''))[:500],
                'description': str(row.get('description', ''))[:500],
                'language': str(row.get('language', '')),
                'category': str(row.get('category', '')),
                'level': str(row.get('level', ''))
            }
            metadatas.append(metadata)
        
        ## batch add
        batch_size = 5000
        for i in range(0, len(ids), batch_size):
            end_idx = min(i + batch_size, len(ids))
            
            self.collection.add(
                ids=ids[i:end_idx],
                documents=documents[i:end_idx],
                metadatas=metadatas[i:end_idx]
            )
            
            logger.info("Added batch {}-{}", i, end_idx)
        
        logger.info("Total topics in collection: {}", self.collection.count())

    def query_with_text(
        self, 
        content_text: str, 
        n_results: int = 50,
        where: Optional[Dict] = None
    ) -> Tuple[List[str], List[float], List[Dict]]:
        
        results = self.collection.query(
            query_texts=[content_text],
            n_results=n_results,
            where=where
        )
        
        topic_ids = results['ids'][0]
        distances = results['distances'][0]
        metadatas = results['metadatas'][0]
        
        return topic_ids, distances, metadatas

    def query_with_filters(
        self,
        content_text: str,
        n_results: int = 50,
        language: Optional[str] = None,
        category: Optional[str] = None
    ) -> Tuple[List[str], List[float]]:
        """Query with metadata filters."""
        
        where_clause = {}
        if language:
            where_clause['language'] = language
        if category:
            where_clause['category'] = category
        
        topic_ids, distances, _ = self.query_with_text(
            content_text, 
            n_results, 
            where=where_clause if where_clause else None
        )
        
        return topic_ids, distances