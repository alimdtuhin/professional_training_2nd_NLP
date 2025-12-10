from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from typing import List, Dict, Any
import os

class TextGenerator:
    def __init__(self, model_name="gpt2"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None
        self.generator = None
        self.use_fallback = False
        
        print(f"Device set to use {self.device}")
        
        # Check if local model exists
        local_model_path = f"models/{model_name}"
        if os.path.exists(local_model_path):
            print(f"Loading local model from: {local_model_path}")
            self.load_local_model(local_model_path)
        else:
            print(f"Local model not found at {local_model_path}")
            print("Attempting to download from Hugging Face...")
            try:
                self.load_model()
            except Exception as e:
                print(f"Could not load transformer model: {e}")
                print("Using fallback text generator...")
                self.use_fallback = True
    
    def load_local_model(self, model_path: str):
        """Load model from local directory"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            # Fix for older models
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with appropriate settings
            if self.device == "cuda":
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16
                ).to(self.device)
            else:
                self.model = AutoModelForCausalLM.from_pretrained(model_path)
            
            # Create text generation pipeline
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1
            )
            
            print(f"Local model loaded successfully on {self.device}")
        except Exception as e:
            print(f"Error loading local model: {e}")
            self.use_fallback = True
    
    def load_model(self):
        """Load the transformer model from Hugging Face"""
        print(f"Loading model: {self.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Fix for older models
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        if self.device == "cuda":
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16
            ).to(self.device)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        
        # Create text generation pipeline
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if self.device == "cuda" else -1
        )
        
        print(f"Model loaded on {self.device}")
    
    def generate_text(self, prompt: str, max_length: int = 100, 
                     temperature: float = 0.7, top_p: float = 0.9,
                     num_return_sequences: int = 1) -> List[str]:
        """Generate text based on prompt"""
        
        if self.use_fallback:
            # Fallback response for conversations
            fallback_responses = [
                f"{prompt}\n\nThis is a demonstration response. The AI would typically provide more detailed information here.",
                f"Based on your question '{prompt}', here's what I can tell you...",
                f"I understand you're asking about {prompt}. In a real implementation, I would provide a comprehensive answer."
            ]
            import random
            return [random.choice(fallback_responses)]
        
        try:
            # Generate text with updated parameters
            generated = self.generator(
                prompt,
                max_new_tokens=max_length,  # Use max_new_tokens instead of max_length
                temperature=temperature,
                top_p=top_p,
                num_return_sequences=num_return_sequences,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,
                truncation=True  # Add truncation parameter
            )
            
            # Extract generated text
            results = [g['generated_text'] for g in generated]
            
            # Clean up the response (remove the prompt if it's repeated)
            cleaned_results = []
            for result in results:
                if result.startswith(prompt):
                    result = result[len(prompt):].strip()
                cleaned_results.append(result)
            
            return cleaned_results
            
        except Exception as e:
            print(f"Error in generate_text: {e}")
            return [f"I encountered an error while generating a response. Please try again."]
    
    def generate_with_context(self, prompt: str, context: List[str], 
                            max_length: int = 150) -> str:
        """Generate text with context from similar documents"""
        
        if self.use_fallback:
            # Fallback response with context
            context_text = "\n".join(context[:2]) if context else "No context available"
            return f"Based on the context provided, I would respond to '{prompt}' with relevant information from previous discussions."
        
        # Combine context with prompt
        if context:
            context_text = "\n".join([f"Context: {c}" for c in context[:3]])
            enhanced_prompt = f"{context_text}\n\nQuestion: {prompt}\n\nAnswer:"
        else:
            enhanced_prompt = f"Question: {prompt}\n\nAnswer:"
        
        # Generate text
        results = self.generate_text(
            enhanced_prompt,
            max_length=max_length,
            temperature=0.8
        )
        
        return results[0] if results else "I couldn't generate a response at this time."

if __name__ == "__main__":
    # Test the text generator
    print("Testing text generator...")
    generator = TextGenerator()
    
    # Test generation
    prompt = "What is artificial intelligence?"
    results = generator.generate_text(prompt, max_length=100)
    
    print("Prompt:", prompt)
    print("Generated:", results[0])