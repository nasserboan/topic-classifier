import pandas as pd
from loguru import logger
from pathlib import Path
from typing import Dict, List, Tuple


class DataLoader:

    def __init__(self, data_dir: str = "./data"):
        self.data_dir = Path(data_dir)
        self.topics = None
        self.content = None
        self.correlations = None
        self.content_to_topics = None

    def load_all(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load all the data from the data directory.
        """

        self.topics = pd.read_csv(self.data_dir / "topics.csv")
        logger.info(f"Loaded {len(self.topics)} topics")

        self.content = pd.read_csv(self.data_dir / "content.csv")
        logger.info(f"Loaded {len(self.content)} content")

        self.correlations = pd.read_csv(self.data_dir / "correlations.csv")
        logger.info(f"Loaded {len(self.correlations)} correlations")

        return self.topics, self.content, self.correlations

    def build_content_topic_mapping(self) -> Dict[str, List[str]]:
        """
        Convert correlactions to content_id -> [topic_ids] dict
        """

        logger.info(f"Building content to topic mapping")

        content_to_topics = {}

        for _, row in self.correlations.iterrows():
            topic_id = row['topic_id']
            content_ids = row['content_ids'].split()
            
            for content_id in content_ids:
                if content_id not in content_to_topics:
                    content_to_topics[content_id] = []
                content_to_topics[content_id].append(topic_id)

        self.content_to_topics = content_to_topics
        logger.info("Mapped {} content items to topics", len(content_to_topics))

        return content_to_topics

    def filter_topics_by_level(self, max_level: int = 0) -> pd.DataFrame:
        """Keep only topics at or below specified level and filter related content."""
        logger.info("Filtering topics: max_level={}", max_level)
        
        original_topic_count = len(self.topics)
        original_content_count = len(self.content)
        
        ## filter topics
        self.topics = self.topics[self.topics['level'] <= max_level]
        logger.info("Topics reduced: {} → {}", original_topic_count, len(self.topics))
        
        ## get remaining topic IDs
        valid_topic_ids = set(self.topics['id'])
        
        ## filter correlations to only include valid topics
        original_corr = len(self.correlations)
        self.correlations = self.correlations[
            self.correlations['topic_id'].isin(valid_topic_ids)
        ]
        logger.info("Correlations reduced: {} → {}", original_corr, len(self.correlations))
        
        ## rebuild content-to-topics mapping with filtered topics
        self.build_content_topic_mapping()
        
        ## filter content to only items with valid topics
        valid_content_ids = set(self.content_to_topics.keys())
        self.content = self.content[self.content['id'].isin(valid_content_ids)]
        
        logger.info("Content reduced: {} → {}", original_content_count, len(self.content))
        
        return self.topics