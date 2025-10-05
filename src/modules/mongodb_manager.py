import pandas as pd
import numpy as np
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import ConnectionFailure, OperationFailure
import json
from datetime import datetime, timedelta
from bson import ObjectId, json_util
import logging
from typing import Dict, List, Any, Optional, Union

class MongoDBManager:
    """MongoDB Atlas database manager for the analytics platform"""
    
    def __init__(self, connection_string: str, database_name: str):
        self.connection_string = connection_string
        self.database_name = database_name
        self.client = None
        self.db = None
        self.logger = logging.getLogger(__name__)
        self._connect()
    
    def _connect(self):
        """Establish connection to MongoDB Atlas"""
        try:
            self.client = MongoClient(self.connection_string, serverSelectionTimeoutMS=5000)
            # Test connection
            self.client.admin.command('ping')
            self.db = self.client[self.database_name]
            self.logger.info(f"Successfully connected to MongoDB Atlas: {self.database_name}")
        except ConnectionFailure as e:
            self.logger.error(f"Failed to connect to MongoDB Atlas: {e}")
            raise
    
    def close_connection(self):
        """Close database connection"""
        if self.client:
            self.client.close()
            self.logger.info("MongoDB connection closed")
    
    def insert_dataframe(self, collection_name: str, df: pd.DataFrame, 
                        metadata: Dict = None) -> List[ObjectId]:
        """Insert pandas DataFrame into MongoDB collection"""
        try:
            collection = self.db[collection_name]
            
            # Convert DataFrame to dictionary
            records = df.to_dict('records')
            
            # Add metadata if provided
            if metadata:
                for record in records:
                    record['_metadata'] = metadata
            
            # Add timestamp
            for record in records:
                record['created_at'] = datetime.utcnow()
            
            result = collection.insert_many(records)
            self.logger.info(f"Inserted {len(result.inserted_ids)} records into {collection_name}")
            return result.inserted_ids
        
        except Exception as e:
            self.logger.error(f"Error inserting DataFrame: {e}")
            raise
    
    def find_dataframe(self, collection_name: str, query: Dict = None, 
                      projection: Dict = None, limit: int = 0) -> pd.DataFrame:
        """Retrieve data from MongoDB and return as pandas DataFrame"""
        try:
            collection = self.db[collection_name]
            
            if query is None:
                query = {}
            if projection is None:
                projection = {}
            
            cursor = collection.find(query, projection)
            
            if limit > 0:
                cursor = cursor.limit(limit)
            
            # Convert to list and then to DataFrame
            records = list(cursor)
            
            if not records:
                return pd.DataFrame()
            
            # Remove MongoDB _id field and metadata for cleaner DataFrame
            for record in records:
                if '_id' in record:
                    del record['_id']
                if '_metadata' in record:
                    del record['_metadata']
                if 'created_at' in record:
                    del record['created_at']
            
            return pd.DataFrame(records)
        
        except Exception as e:
            self.logger.error(f"Error retrieving DataFrame: {e}")
            raise
    
    def save_model_metadata(self, model_name: str, model_type: str, 
                           parameters: Dict, performance: Dict, 
                           feature_importance: List = None) -> ObjectId:
        """Save machine learning model metadata to MongoDB"""
        try:
            collection = self.db['model_metadata']
            
            model_doc = {
                'model_name': model_name,
                'model_type': model_type,
                'parameters': parameters,
                'performance': performance,
                'feature_importance': feature_importance or [],
                'created_at': datetime.utcnow(),
                'version': '1.0'
            }
            
            result = collection.insert_one(model_doc)
            self.logger.info(f"Saved model metadata: {model_name} with ID: {result.inserted_id}")
            return result.inserted_id
        
        except Exception as e:
            self.logger.error(f"Error saving model metadata: {e}")
            raise
    
    def save_analysis_results(self, analysis_type: str, dataset_id: str,
                             results: Dict, visualizations: List = None) -> ObjectId:
        """Save analysis results to MongoDB"""
        try:
            collection = self.db['analysis_results']
            
            result_doc = {
                'analysis_type': analysis_type,
                'dataset_id': dataset_id,
                'results': results,
                'visualizations': visualizations or [],
                'created_at': datetime.utcnow()
            }
            
            result = collection.insert_one(result_doc)
            self.logger.info(f"Saved {analysis_type} analysis results")
            return result.inserted_id
        
        except Exception as e:
            self.logger.error(f"Error saving analysis results: {e}")
            raise
    
    def get_model_history(self, model_name: str = None, limit: int = 10) -> List[Dict]:
        """Retrieve model training history"""
        try:
            collection = self.db['model_metadata']
            
            query = {}
            if model_name:
                query['model_name'] = model_name
            
            cursor = collection.find(query).sort('created_at', DESCENDING).limit(limit)
            return list(cursor)
        
        except Exception as e:
            self.logger.error(f"Error retrieving model history: {e}")
            raise
    
    def get_collection_stats(self, collection_name: str) -> Dict:
        """Get statistics for a collection"""
        try:
            collection = self.db[collection_name]
            
            stats = {
                'total_documents': collection.count_documents({}),
                'storage_size': collection.estimated_document_count(),
                'indexes': list(collection.index_information().keys()),
                'sample_document': collection.find_one()
            }
            
            return stats
        
        except Exception as e:
            self.logger.error(f"Error getting collection stats: {e}")
            raise
    
    def create_index(self, collection_name: str, fields: List, unique: bool = False):
        """Create index on specified fields"""
        try:
            collection = self.db[collection_name]
            index_fields = [(field, ASCENDING) for field in fields]
            collection.create_index(index_fields, unique=unique)
            self.logger.info(f"Created index on {fields} in {collection_name}")
        
        except Exception as e:
            self.logger.error(f"Error creating index: {e}")
            raise
    
    def backup_data(self, collection_name: str, backup_collection: str):
        """Create backup of a collection"""
        try:
            source_collection = self.db[collection_name]
            backup_coll = self.db[backup_collection]
            
            # Add timestamp to backup
            backup_data = list(source_collection.find())
            for doc in backup_data:
                doc['backup_date'] = datetime.utcnow()
            
            if backup_data:
                backup_coll.insert_many(backup_data)
                self.logger.info(f"Backed up {len(backup_data)} documents from {collection_name}")
        
        except Exception as e:
            self.logger.error(f"Error during backup: {e}")
            raise

# Singleton instance
_mongodb_manager = None

def get_mongodb_manager():
    """Get MongoDB manager instance (singleton)"""
    global _mongodb_manager
    from src.config.config import config
    
    if _mongodb_manager is None and config.USE_MONGODB:
        _mongodb_manager = MongoDBManager(
            connection_string=config.MONGODB_ATLAS_URI,
            database_name=config.MONGODB_DATABASE
        )
    
    return _mongodb_manager