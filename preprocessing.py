import re
import nltk
from nltk.corpus import stopwords
import numpy as np
from sentence_transformers import SentenceTransformer
import os

class TextPreprocessor:
    def __init__(self):
        # Download NLTK data only once
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            print("Downloading NLTK stopwords...")
            nltk.download('stopwords')
        
        self.stop_words = set(stopwords.words('english'))
        self.embedding_model = None
        self.use_fallback = False
        
        # Check if local model exists
        local_model_path = "models/sentence-transformers/all-MiniLM-L6-v2"
        if os.path.exists(local_model_path):
            print(f"Loading local embedding model from: {local_model_path}")
            try:
                self.embedding_model = SentenceTransformer(local_model_path)
                print("Local embedding model loaded successfully")
            except Exception as e:
                print(f"Error loading local embedding model: {e}")
                self.use_fallback = True
        else:
            print("Local embedding model not found")
            print("Attempting to download from Hugging Face...")
            try:
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                print("Embedding model loaded from Hugging Face")
            except Exception as e:
                print(f"Could not load embedding model: {e}")
                print("Using fallback embeddings...")
                self.use_fallback = True
    
    def clean_text(self, text):
        """Clean and preprocess text"""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and numbers (keep basic punctuation for context)
        text = re.sub(r'[^\w\s\.\,\!\?]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize_text(self, text):
        """Simple tokenization without NLTK punkt dependency"""
        # Simple whitespace tokenization with punctuation handling
        text = re.sub(r'([^\w\s])', r' \1 ', text)
        tokens = text.split()
        return tokens
    
    def remove_stopwords(self, tokens):
        """Remove stopwords from tokens"""
        return [token for token in tokens if token.lower() not in self.stop_words]
    
    def preprocess(self, text):
        """Complete preprocessing pipeline"""
        cleaned = self.clean_text(text)
        tokens = self.tokenize_text(cleaned)
        filtered_tokens = self.remove_stopwords(tokens)
        return ' '.join(filtered_tokens)
    
    def generate_embeddings(self, texts):
        """Generate embeddings for texts"""
        if isinstance(texts, str):
            texts = [texts]
        
        if self.use_fallback:
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
            # Clean texts first (but don't over-clean for embeddings)
            cleaned_texts = [self.clean_text(text) for text in texts]
            embeddings = self.embedding_model.encode(cleaned_texts)
            return embeddings.tolist()
    
    def batch_preprocess(self, texts):
        """Preprocess multiple texts"""
        return [self.preprocess(text) for text in texts]

if __name__ == "__main__":
    # Test the preprocessor
    preprocessor = TextPreprocessor()
    sample_text = "Hello! This is a test sentence with numbers 123 and special #characters."
    print("Original:", sample_text)
    print("Cleaned:", preprocessor.preprocess(sample_text))