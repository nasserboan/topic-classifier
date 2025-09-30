import pandas as pd
from loguru import logger

class TextProcessor:
    """
    Consolidates and cleans text fields for embedding.
    """

    def __init__(self):
        self.content_texts = None
        self.topic_texts = None
    
    def consolidate_content_text(self, content_df: pd.DataFrame) -> pd.Series:
        """
        Combine title + description + text into single string per content.
        """
        logger.info("Consolidating content text fields")
        
        title = content_df['title'].fillna('')
        description = content_df['description'].fillna('')
        text = content_df['text'].fillna('')
        
        combined = title + ' ' + description + ' ' + text
        
        combined = combined.str.strip()
        
        logger.info("Processed {} content texts", len(combined))
        return combined
    
    def consolidate_topic_text(self, topics_df: pd.DataFrame) -> pd.Series:
        """
        Combine title + description for topics.
        """
        logger.info("Consolidating topic text fields")
        
        title = topics_df['title'].fillna('')
        description = topics_df['description'].fillna('')
        
        combined = title + ' ' + description
        combined = combined.str.strip()
        
        logger.info("Processed {} topic texts", len(combined))
        return combined
    
    def check_empty_texts(self, text_series: pd.Series, label: str):
        """
        Log statistics about empty text entries.
        """
        empty_count = (text_series == '').sum()
        logger.warning("{}: {} empty texts out of {}", 
                      label, empty_count, len(text_series))