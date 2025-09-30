from src.data_loader import DataLoader
from src.preprocessor import TextProcessor
from src.vec_store import TopicVectorStore
from loguru import logger

def train_and_build_vector_store(
    data_dir: str = "./data",
    chroma_dir: str = "./chroma_db",
    max_level: int = 1
):
    """Load data, filter, and build vector store."""
    
    logger.info("=== TRAINING PIPELINE ===")
    
    data_loader = DataLoader(data_dir)
    processor = TextProcessor()
    vec_store = TopicVectorStore(persist_dir=chroma_dir)
    
    logger.info("Loading datasets...")
    topics, content, correlations = data_loader.load_all()
    
    logger.info("Filtering topics by level <= {}", max_level)
    data_loader.filter_topics_by_level(max_level=max_level)
    
    topics = data_loader.topics
    content = data_loader.content
    
    logger.info("Final sizes: {} topics, {} content", len(topics), len(content))
    
    logger.info("Processing text fields...")
    topic_texts = processor.consolidate_topic_text(topics)
    topics['combined_text'] = topic_texts
    
    logger.info("Creating vector store collection...")
    vec_store.create_topic_collection()
    
    logger.info("Adding topics to vector store...")
    vec_store.add_topics(topics, text_column='combined_text')
    
    logger.info("=== TRAINING COMPLETE ===")
    logger.info("Vector store saved to: {}", chroma_dir)
    
    return vec_store


if __name__ == "__main__":
    train_and_build_vector_store(
        data_dir="./data",
        chroma_dir="./chroma_db",
        max_level=1
    )