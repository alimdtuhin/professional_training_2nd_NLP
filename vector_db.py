import chromadb
from chromadb.config import Settings
import uuid
from typing import List, Dict, Any
import os
import numpy as np

class ChromaDBManager:
    def __init__(self, collection_name="conversations_knowledge"):
        # Create persistence directory
        persist_directory = "./chroma_db"
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize Chroma client
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection_name = collection_name
        
        # Try to use SentenceTransformer if available
        self.embedding_model = None
        self.use_fallback = False
        
        try:
            from sentence_transformers import SentenceTransformer
            local_model_path = "models/sentence-transformers/all-MiniLM-L6-v2"
            if os.path.exists(local_model_path):
                print(f"Loading local embedding model from: {local_model_path}")
                self.embedding_model = SentenceTransformer(local_model_path)
            else:
                print("Warning: Local embedding model not found")
                print("Attempting to download embedding model...")
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.use_fallback = False
            print("Embedding model loaded for ChromaDB")
        except Exception as e:
            print(f"Could not load embedding model for ChromaDB: {e}")
            print("Using fallback embeddings for ChromaDB...")
            self.embedding_model = None
            self.use_fallback = True
        
        # Create or get collection
        try:
            self.collection = self.client.get_collection(collection_name)
            print(f"Loaded existing collection: {collection_name}")
        except:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine", "description": "Conversations and knowledge database"}
            )
            print(f"Created new collection: {collection_name}")
    
    def generate_embeddings(self, texts: List[str]):
        """Generate embeddings for texts"""
        if self.use_fallback or self.embedding_model is None:
            # Generate simple fallback embeddings
            embeddings = []
            for text in texts:
                # Simple hash-based embedding (for demonstration)
                seed = hash(text) % 10000
                np.random.seed(seed)
                embedding = np.random.randn(384)  # Same dimension as the model
                embedding = embedding / np.linalg.norm(embedding)  # Normalize
                embeddings.append(embedding.tolist())
            return embeddings
        else:
            # Generate embeddings using the model
            # Simple text cleaning
            cleaned_texts = []
            for text in texts:
                if isinstance(text, str):
                    text = text.strip()
                    cleaned_texts.append(text)
                else:
                    cleaned_texts.append(str(text))
            
            try:
                embeddings = self.embedding_model.encode(cleaned_texts)
                return embeddings.tolist()
            except Exception as e:
                print(f"Error generating embeddings: {e}")
                # Fallback
                embeddings = []
                for text in cleaned_texts:
                    seed = hash(text) % 10000
                    np.random.seed(seed)
                    embedding = np.random.randn(384)
                    embedding = embedding / np.linalg.norm(embedding)
                    embeddings.append(embedding.tolist())
                return embeddings
    
    def add_documents(self, texts: List[str], metadata: List[Dict] = None):
        """Add documents to ChromaDB with optional metadata"""
        if metadata is None:
            # Create default metadata if none provided
            metadata = [{} for _ in texts]
        
        # Ensure metadata is list of dicts and not empty
        cleaned_metadata = []
        for i, meta in enumerate(metadata):
            if not isinstance(meta, dict):
                # Create default metadata with index
                cleaned_metadata.append({
                    "source": "user_input",
                    "added_at": "unknown",
                    "index": str(i)
                })
            else:
                # Ensure metadata dict is not empty
                if not meta:
                    meta = {
                        "source": "user_input",
                        "added_at": "unknown",
                        "index": str(i)
                    }
                cleaned_metadata.append(meta)
        
        # Generate embeddings
        embeddings = self.generate_embeddings(texts)
        
        # Generate IDs
        ids = [str(uuid.uuid4()) for _ in texts]
        
        # Add to collection
        try:
            self.collection.add(
                embeddings=embeddings,
                documents=texts,
                metadatas=cleaned_metadata,
                ids=ids
            )
            print(f"Added {len(texts)} documents to ChromaDB")
            return ids
        except Exception as e:
            print(f"Error adding documents: {e}")
            # Try adding one by one
            success_count = 0
            for i, (text, embedding, meta, id_) in enumerate(zip(texts, embeddings, cleaned_metadata, ids)):
                try:
                    self.collection.add(
                        embeddings=[embedding],
                        documents=[text],
                        metadatas=[meta],
                        ids=[id_]
                    )
                    success_count += 1
                except Exception as e2:
                    print(f"Error adding document {i}: {e2}")
            
            print(f"Successfully added {success_count}/{len(texts)} documents")
            return ids[:success_count]
    
    def search_similar(self, query: str, n_results: int = 10, where: Dict = None):
        """Search for similar documents with optional filtering"""
        try:
            # Generate embedding for query
            query_embedding = self.generate_embeddings([query])
            
            # Prepare query parameters
            query_params = {
                "query_embeddings": query_embedding,
                "n_results": min(n_results, 20)
            }
            
            # Add filter if provided
            if where:
                query_params["where"] = where
            
            # Search in collection
            results = self.collection.query(**query_params)
            
            # Debug output
            print(f"Search query: '{query}'")
            print(f"Results found: {len(results['documents'][0]) if results['documents'] else 0}")
            
            return results
            
        except Exception as e:
            print(f"Error in search_similar: {e}")
            # Return empty results structure
            return {
                'documents': [[]],
                'distances': [[]],
                'metadatas': [[]],
                'ids': [[]]
            }
    
    def get_all_documents(self):
        """Get all documents from collection"""
        try:
            # Get all documents from collection
            results = self.collection.get()
            print(f"Retrieved {len(results.get('documents', []))} documents from collection")
            return results
        except Exception as e:
            print(f"Error getting all documents: {e}")
            return {'documents': [], 'metadatas': []}
    
    def search_by_metadata(self, metadata_filter: Dict):
        """Search documents by metadata"""
        try:
            results = self.collection.get(where=metadata_filter)
            return results
        except Exception as e:
            print(f"Error searching by metadata: {e}")
            return {'documents': [], 'metadatas': []}
    
    def delete_collection(self):
        """Delete the collection"""
        try:
            self.client.delete_collection(self.collection_name)
            print(f"Deleted collection: {self.collection_name}")
        except Exception as e:
            print(f"Error deleting collection: {e}")
    
    def reset_collection(self):
        """Reset the collection"""
        try:
            self.delete_collection()
        except:
            pass
        
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine", "description": "Conversations and knowledge database"}
        )
        print(f"Reset collection: {self.collection_name}")

if __name__ == "__main__":
    # Test ChromaDB with conversations
    print("Testing ChromaDB with conversation storage...")
    db = ChromaDBManager()
    
    # Test adding conversation
    test_conversation = "USER: What is artificial intelligence?\nAI: Artificial intelligence is the simulation of human intelligence in machines."
    test_metadata = {
        "type": "conversation",
        "timestamp": "2024-01-01T12:00:00",
        "user": "test_user"
    }
    
    print(f"\nAdding test conversation...")
    ids = db.add_documents([test_conversation], [test_metadata])
    print(f"Added conversation with ID: {ids[0]}")
    
    # Test search
    print(f"\nSearching for 'artificial intelligence'...")
    results = db.search_similar("artificial intelligence", n_results=5)
    
    if results and 'documents' in results and results['documents']:
        docs = results['documents'][0]
        metadatas = results['metadatas'][0] if results['metadatas'] else []
        
        print(f"Found {len(docs)} results:")
        for i, (doc, meta) in enumerate(zip(docs, metadatas)):
            print(f"\nResult {i+1}:")
            print(f"Type: {meta.get('type', 'unknown')}")
            print(f"Content: {doc[:100]}...")
    else:
        print("No results found")