from typing import List, Optional
from pydantic import BaseModel
from pathlib import Path
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from loguru import logger
import numpy as np
import argparse
from src.metrics import precision_at_k, recall_at_k, f2_at_k
from src.data_loader import DataLoader
from src.preprocessor import TextProcessor
from tqdm import tqdm
import pandas as pd


class TopicPredictionRequest(BaseModel):
    content_title: Optional[str] = None
    content_description: Optional[str] = None
    content_kind: Optional[str] = None
    content_text: Optional[str] = None
    content_language: Optional[str] = None
    topic_title: Optional[str] = None
    topic_description: Optional[str] = None
    topic_category: Optional[str] = None


class TopicPredictor:
    def __init__(
        self, 
        chroma_dir: str = "./chroma_db",
        collection_name: str = "topics",
        model_name: str = "paraphrase-multilingual-mpnet-base-v2"
    ):
        logger.info("Loading TopicPredictor from {}", chroma_dir)
        
        self.client = chromadb.PersistentClient(
            path=chroma_dir,
            settings=Settings(anonymized_telemetry=False)
        )
        
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=model_name
        )
        
        self.collection = self.client.get_collection(
            name=collection_name,
            embedding_function=self.embedding_function
        )
        
        logger.info("Loaded collection with {} topics", self.collection.count())
        self.model = self.collection

    def predict(
        self, 
        request: TopicPredictionRequest, 
        n_results: int = 50
    ) -> List[str]:
        """Predict topics for content using semantic similarity."""
        
        text_parts = []
        if request.content_title:
            text_parts.append(request.content_title)
        if request.content_description:
            text_parts.append(request.content_description)
        if request.content_text:
            text_parts.append(request.content_text)
        
        content_text = ' '.join(text_parts).strip()
        
        if not content_text:
            logger.warning("No text provided in request")
            return []
        
        where_clause = None
        if request.content_language:
            where_clause = {'language': request.content_language}
        
        results = self.collection.query(
            query_texts=[content_text],
            n_results=n_results,
            where=where_clause
        )
        
        topic_ids = results['ids'][0] if results['ids'] else []
        
        return topic_ids


def test_predictor(
    data_dir: str = "./data",
    chroma_dir: str = "./chroma_db",
    max_level: int = 1,
    n_samples: int = 100
):
    """Test predictor on sample data with same filters as training."""
    
    logger.info("=== TESTING PREDICTOR ===")
    
    # Load and filter data same as training
    data_loader = DataLoader(data_dir)
    processor = TextProcessor()
    
    topics, content, correlations = data_loader.load_all()
    data_loader.filter_topics_by_level(max_level=max_level)
    
    content = data_loader.content
    content_to_topics = data_loader.content_to_topics
    
    content_texts = processor.consolidate_content_text(content)
    content['combined_text'] = content_texts
    
    # Sample content
    sample_content = content.sample(n=n_samples, random_state=42)
    
    # Initialize predictor
    predictor = TopicPredictor(chroma_dir=chroma_dir)
    
    # Collect metrics
    results = {
        'precision@5': [],
        'recall@5': [],
        'f2@5': [],
        'precision@10': [],
        'recall@10': [],
        'f2@10': []
    }
    
    logger.info("Testing on {} samples", len(sample_content))
    
    for idx, row in tqdm(sample_content.iterrows(), total=len(sample_content)):
        content_id = row['id']
        
        actual_topics = content_to_topics.get(content_id, [])
        if len(actual_topics) == 0:
            continue
        
        # Convert NaN to None for Pydantic
        def safe_get(value):
            if pd.isna(value):
                return None
            return value
        
        request = TopicPredictionRequest(
            content_title=safe_get(row.get('title')),
            content_description=safe_get(row.get('description')),
            content_text=safe_get(row.get('text')),
            content_language=safe_get(row.get('language'))
        )
        
        predicted_topics = predictor.predict(request, n_results=10)
        
        # Calculate metrics
        results['precision@5'].append(precision_at_k(predicted_topics, actual_topics, 5))
        results['recall@5'].append(recall_at_k(predicted_topics, actual_topics, 5))
        results['f2@5'].append(f2_at_k(predicted_topics, actual_topics, 5))
        
        results['precision@10'].append(precision_at_k(predicted_topics, actual_topics, 10))
        results['recall@10'].append(recall_at_k(predicted_topics, actual_topics, 10))
        results['f2@10'].append(f2_at_k(predicted_topics, actual_topics, 10))
    
    # Print results
    logger.info("\n=== EVALUATION RESULTS ===")
    for metric, values in results.items():
        avg = np.mean(values)
        logger.info("{}: {:.4f}", metric, avg)
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test TopicPredictor with evaluation metrics")
    
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        help="Path to data directory"
    )
    
    parser.add_argument(
        "--chroma_dir",
        type=str,
        default="./chroma_db",
        help="Path to ChromaDB directory"
    )
    
    parser.add_argument(
        "--max_level",
        type=int,
        default=1,
        help="Maximum topic level to include"
    )
    
    parser.add_argument(
        "--n_samples",
        type=int,
        default=500,
        help="Number of samples to evaluate"
    )
    
    args = parser.parse_args()
    
    test_predictor(
        data_dir=args.data_dir,
        chroma_dir=args.chroma_dir,
        max_level=args.max_level,
        n_samples=args.n_samples
    )