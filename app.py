import os
os.environ['NO_PROXY'] = '*'
os.environ['all_proxy'] = ''
os.environ['ALL_PROXY'] = ''
os.environ['http_proxy'] = ''
os.environ['https_proxy'] = ''
os.environ['socks_proxy'] = ''
# Set environment variables for offline mode
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

# Download NLTK data at the beginning
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK data...")
    nltk.download('punkt')
    nltk.download('stopwords')

import gradio as gr
from model_handler import TextGenerator
from vector_db import ChromaDBManager
from preprocessing import TextPreprocessor
import time
import json
from datetime import datetime
from typing import List, Dict

class LLMApplication:
    def __init__(self):
        # Initialize components
        print("Initializing text generator...")
        self.text_generator = TextGenerator()
        print("Initializing text preprocessor...")
        self.preprocessor = TextPreprocessor()
        
        # Conversation history
        self.conversation_history = []
        self.current_conversation = []
        
        # Try to initialize ChromaDB with retry logic
        max_retries = 3
        self.vector_db = None
        
        for attempt in range(max_retries):
            try:
                print(f"Attempting to initialize ChromaDB (attempt {attempt + 1})...")
                self.vector_db = ChromaDBManager()
                print(f"ChromaDB initialized successfully")
                break
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)  # Wait before retry
                else:
                    print("Could not initialize ChromaDB. Running in fallback mode.")
        
        # Sample data for initialization
        self.initialize_sample_data()
    
    def initialize_sample_data(self):
        """Initialize with sample data if database is empty"""
        if self.vector_db is None:
            print("Running without ChromaDB - using in-memory fallback")
            self.sample_data = [
                "Artificial intelligence is transforming industries worldwide.",
                "Machine learning algorithms learn from data patterns.",
                "Deep learning models require large datasets for training.",
                "Natural language processing enables chatbots and translators.",
                "Transformers revolutionized NLP with attention mechanisms.",
            ]
            return
        
        try:
            docs = self.vector_db.get_all_documents()
            if len(docs) == 0 or (isinstance(docs, dict) and len(docs.get('documents', [])) == 0):
                print("Database is empty, adding sample data...")
                sample_texts = [
                    "Artificial intelligence is transforming industries worldwide.",
                    "Machine learning algorithms learn from data patterns.",
                    "Deep learning models require large datasets for training.",
                    "Natural language processing enables chatbots and translators.",
                    "Transformers revolutionized NLP with attention mechanisms.",
                    "Hugging Face provides accessible pre-trained models.",
                    "Vector databases store embeddings for semantic search.",
                    "Docker containers package applications with dependencies.",
                    "Python is the primary language for machine learning.",
                    "Gradio provides easy interfaces for ML models."
                ]
                
                # Add metadata for each sample text
                sample_metadata = []
                for i, text in enumerate(sample_texts):
                    sample_metadata.append({
                        "type": "knowledge",
                        "source": "sample_data",
                        "index": str(i),
                        "added_at": "initialization"
                    })
                
                self.vector_db.add_documents(sample_texts, sample_metadata)
                print(f"Initialized with {len(sample_texts)} sample documents")
            else:
                if isinstance(docs, dict):
                    doc_count = len(docs.get('documents', []))
                else:
                    doc_count = len(docs)
                print(f"Database already has {doc_count} documents")
        except Exception as e:
            print(f"Error initializing data: {e}")
            self.sample_data = [
                "Artificial intelligence is transforming industries worldwide.",
                "Machine learning algorithms learn from data patterns.",
                "Deep learning models require large datasets for training.",
            ]
    
    def save_conversation(self, user_input: str, ai_response: str, context_used: bool = False):
        """Save a conversation turn to the database"""
        if self.vector_db is None:
            return False
        
        try:
            # Format conversation for storage
            conversation_text = f"USER: {user_input}\nAI: {ai_response}"
            
            # Create metadata for better search
            metadata = {
                "type": "conversation",
                "timestamp": datetime.now().isoformat(),
                "context_used": str(context_used),
                "user_query": user_input[:100]  # Store first 100 chars for reference
            }
            
            # Save to database
            self.vector_db.add_documents([conversation_text], [metadata])
            
            # Also save to conversation history
            self.current_conversation.append({
                "user": user_input,
                "ai": ai_response,
                "timestamp": datetime.now().isoformat(),
                "context_used": context_used
            })
            
            # Keep last 10 conversations in memory
            if len(self.current_conversation) > 10:
                self.conversation_history.append(self.current_conversation.copy())
                self.current_conversation = []
            
            print(f"Saved conversation to database: {len(user_input)} chars")
            return True
            
        except Exception as e:
            print(f"Error saving conversation: {e}")
            return False
    
    def generate_response(self, prompt, use_context, max_length, temperature, save_conversation=True):
        """Generate response with optional context and save conversation"""
        try:
            context = []
            if use_context and self.vector_db:
                # Search for similar documents
                similar = self.vector_db.search_similar(prompt, n_results=3)
                
                # Extract context from results
                if similar and 'documents' in similar and similar['documents']:
                    context = similar['documents']
                    # Flatten the list of lists
                    if isinstance(context, list) and len(context) > 0:
                        context = context[0] if isinstance(context[0], list) else context
                
                print(f"Found {len(context)} context documents")
                
                # Generate with context
                response = self.text_generator.generate_with_context(
                    prompt, 
                    context, 
                    max_length=int(max_length)
                )
            else:
                # Generate without context
                response = self.text_generator.generate_text(
                    prompt,
                    max_length=int(max_length),
                    temperature=temperature
                )[0]
            
            # Save conversation if enabled
            if save_conversation:
                self.save_conversation(prompt, response, use_context)
            
            return response
        
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            print(error_msg)
            return error_msg
    
    def add_to_knowledge_base(self, text):
        """Add text to vector database"""
        if self.vector_db is None:
            return "❌ ChromaDB not available. Running in fallback mode."
        
        try:
            # Add metadata to identify as general knowledge
            metadata = {
                "type": "knowledge",
                "timestamp": datetime.now().isoformat(),
                "source": "user_input"
            }
            
            self.vector_db.add_documents([text], [metadata])
            return f"✅ Successfully added to knowledge base!\nText: {text[:100]}..." if len(text) > 100 else f"✅ Successfully added to knowledge base!\nText: {text}"
        except Exception as e:
            return f"❌ Error adding text: {str(e)}"
    
    def search_knowledge_base(self, query, search_type="all"):
        """Search in knowledge base with optional filtering by type"""
        if self.vector_db is None:
            return "ChromaDB not available. Running in fallback mode.\n\nSample data:\n" + "\n".join([f"{i+1}. {doc}" for i, doc in enumerate(self.sample_data[:3])])
        
        try:
            # Get results from database
            results = self.vector_db.search_similar(query, n_results=10)
            
            if not results:
                return "No results returned from search."
            
            if isinstance(results, dict):
                documents = results.get('documents', [])
                distances = results.get('distances', [])
                metadatas = results.get('metadatas', [])
                
                if not documents:
                    return "No similar documents found."
                
                # Ensure documents is a list of lists
                if documents and isinstance(documents[0], list):
                    documents = documents[0]
                
                # Ensure distances is a list of lists
                if distances and isinstance(distances[0], list):
                    distances = distances[0]
                
                # Ensure metadatas is a list of lists
                if metadatas and isinstance(metadatas[0], list):
                    metadatas = metadatas[0]
                
                formatted_results = "**Search Results:**\n\n"
                conversation_count = 0
                knowledge_count = 0
                
                for i in range(min(len(documents), len(distances))):
                    if i >= len(documents):
                        break
                    
                    doc = documents[i] if i < len(documents) else "No document"
                    distance = distances[i] if i < len(distances) else 1.0
                    metadata = metadatas[i] if i < len(metadatas) else {}
                    
                    # Filter by type if specified
                    doc_type = metadata.get('type', 'knowledge')
                    if search_type != "all" and doc_type != search_type:
                        continue
                    
                    # Count by type
                    if doc_type == 'conversation':
                        conversation_count += 1
                    else:
                        knowledge_count += 1
                    
                    # Format the document
                    similarity = max(0, 1 - distance)
                    
                    if doc_type == 'conversation':
                        # Format conversation nicely
                        lines = doc.split('\n')
                        formatted_doc = ""
                        for line in lines:
                            if line.startswith('USER:'):
                                formatted_doc += f"👤 **User:** {line[6:]}\n"
                            elif line.startswith('AI:'):
                                formatted_doc += f"🤖 **AI:** {line[4:]}\n"
                            else:
                                formatted_doc += f"{line}\n"
                        
                        # Add timestamp if available
                        if 'timestamp' in metadata:
                            try:
                                dt = datetime.fromisoformat(metadata['timestamp'].replace('Z', '+00:00'))
                                formatted_doc += f"📅 **Time:** {dt.strftime('%Y-%m-%d %H:%M:%S')}\n"
                            except:
                                pass
                        
                        formatted_results += f"**Conversation #{conversation_count}** (Similarity: {similarity:.3f})\n"
                        formatted_results += formatted_doc + "\n" + "-"*50 + "\n\n"
                    else:
                        # Format knowledge/document
                        doc_preview = doc[:150] + "..." if len(doc) > 150 else doc
                        formatted_results += f"**Document #{knowledge_count}** (Similarity: {similarity:.3f})\n"
                        formatted_results += f"{doc_preview}\n\n" + "-"*50 + "\n\n"
                
                # Add summary
                if conversation_count == 0 and knowledge_count == 0:
                    return "No results found for the selected type."
                
                summary = f"\n**Summary:** Found {conversation_count} conversations and {knowledge_count} knowledge documents."
                formatted_results += summary
                
                return formatted_results
            else:
                return f"Unexpected result format: {type(results)}"
        
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            return f"Error searching: {str(e)}\n\nDetails:\n{error_details}"
    
    def get_conversation_history(self):
        """Get recent conversation history"""
        all_conversations = []
        
        # Add current conversation
        if self.current_conversation:
            all_conversations.extend(self.current_conversation)
        
        # Add from history
        for conv in self.conversation_history[-5:]:  # Last 5 conversation sessions
            all_conversations.extend(conv)
        
        return all_conversations[-20:]  # Return last 20 messages
    
    def clear_current_conversation(self):
        """Clear current conversation"""
        if self.current_conversation:
            self.conversation_history.append(self.current_conversation.copy())
            self.current_conversation = []
            return f"✅ Cleared current conversation. Saved {len(self.conversation_history[-1])} messages to history."
        return "ℹ️ No active conversation to clear."

def create_interface():
    """Create Gradio interface"""
    app = LLMApplication()
    
    # Define custom CSS
    custom_css = """
    .conversation-box { border: 1px solid #ccc; padding: 10px; border-radius: 5px; margin: 5px 0; }
    .user-msg { background-color: #e3f2fd; padding: 8px; border-radius: 5px; margin: 5px 0; }
    .ai-msg { background-color: #f3e5f5; padding: 8px; border-radius: 5px; margin: 5px 0; }
    """
    
    with gr.Blocks(title="LLM Text Generator with Conversation Memory") as demo:
        # Add custom CSS via HTML
        gr.HTML(f"<style>{custom_css}</style>")
        
        gr.Markdown("# 🤖 LLM Chat with Conversation Memory")
        gr.Markdown("Chat with AI and automatically save conversations to searchable database")
        
        with gr.Tab("💬 Chat"):
            with gr.Row():
                with gr.Column(scale=2):
                    chatbot = gr.Chatbot(label="Conversation", height=400)
                    with gr.Row():
                        msg = gr.Textbox(
                            label="Type your message",
                            placeholder="Type your message here...",
                            lines=2,
                            scale=4
                        )
                        send_btn = gr.Button("Send", variant="primary", scale=1)
                    
                    with gr.Row():
                        use_context = gr.Checkbox(
                            label="Use context from knowledge base",
                            value=True
                        )
                        save_conversation = gr.Checkbox(
                            label="💾 Save conversation to database",
                            value=True
                        )
                    
                    # Removed "View Conversation History" button
                    with gr.Row():
                        clear_chat = gr.Button("🗑️ Clear Chat", variant="secondary")
                        # Removed: view_history = gr.Button("📜 View Conversation History", variant="secondary")
                
                with gr.Column(scale=1):
                    gr.Markdown("### Generation Settings")
                    max_length = gr.Slider(
                        minimum=50,
                        maximum=500,
                        value=200,
                        step=50,
                        label="Max response length"
                    )
                    temperature = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=0.7,
                        step=0.1,
                        label="Temperature"
                    )
                    
                    gr.Markdown("### Current Stats")
                    stats_output = gr.Markdown("")
                    
                    def update_stats():
                        history = app.get_conversation_history()
                        total_messages = len(history)
                        return f"**Current Session:** {len(app.current_conversation)} messages\n**Total Recent:** {total_messages} messages"
                    
                    demo.load(update_stats, inputs=[], outputs=stats_output)
        
        with gr.Tab("🔍 Search Conversations"):
            with gr.Row():
                with gr.Column():
                    search_query = gr.Textbox(
                        label="Search in conversations",
                        placeholder="Enter keywords to search in past conversations...",
                        lines=2
                    )
                    with gr.Row():
                        search_type = gr.Radio(
                            choices=["all", "conversation", "knowledge"],
                            value="conversation",
                            label="Search type"
                        )
                        search_btn = gr.Button("🔍 Search", variant="primary")
                    
                    search_results = gr.Markdown(label="Search Results", show_label=False)
                
                with gr.Column():
                    gr.Markdown("### Recent Conversations")
                    recent_conversations = gr.Markdown("")
                    
                    def load_recent_conversations():
                        history = app.get_conversation_history()
                        if not history:
                            return "No recent conversations."
                        
                        formatted = "### Last 10 Messages:\n\n"
                        for i, msg in enumerate(history[-10:]):
                            role = "👤 User" if 'user' in msg else "🤖 AI"
                            text = msg.get('user') or msg.get('ai', '')
                            time_str = ""
                            if 'timestamp' in msg:
                                try:
                                    dt = datetime.fromisoformat(msg['timestamp'].replace('Z', '+00:00'))
                                    time_str = dt.strftime("%H:%M")
                                except:
                                    pass
                            
                            formatted += f"{i+1}. **{role}** ({time_str}): {text[:80]}...\n"
                        
                        return formatted
                    
                    demo.load(load_recent_conversations, inputs=[], outputs=recent_conversations)
        
        # Removed the "Knowledge Base" tab completely
        
        with gr.Tab("📊 Database Status"):
            with gr.Row():
                with gr.Column():
                    status_btn = gr.Button("🔄 Refresh Database Status", variant="secondary")
                    status_output = gr.Markdown(label="Database Information")
                    
                    def check_database_status():
                        if app.vector_db is None:
                            return "❌ ChromaDB is not initialized.\nRunning in fallback mode."
                        
                        try:
                            docs = app.vector_db.get_all_documents()
                            if isinstance(docs, dict):
                                all_docs = docs.get('documents', [])
                                all_metas = docs.get('metadatas', [])
                            else:
                                all_docs = docs
                                all_metas = []
                            
                            doc_count = len(all_docs) if isinstance(all_docs, list) else 0
                            
                            # Count by type
                            conversation_count = 0
                            knowledge_count = 0
                            
                            if all_metas:
                                for meta in all_metas:
                                    if isinstance(meta, dict) and meta.get('type') == 'conversation':
                                        conversation_count += 1
                                    else:
                                        knowledge_count += 1
                            
                            status = f"✅ **Database Status:** Operational\n\n"
                            status += f"**Total Documents:** {doc_count}\n"
                            status += f"**Conversations:** {conversation_count}\n"
                            status += f"**Knowledge Items:** {knowledge_count}\n\n"
                            
                            # Show recent additions
                            status += "**Recent Activity:**\n"
                            history = app.get_conversation_history()
                            if history:
                                status += f"- Current session: {len(app.current_conversation)} messages\n"
                                status += f"- Total recent: {len(history)} messages\n"
                            else:
                                status += "- No recent activity\n"
                            
                            return status
                        except Exception as e:
                            return f"❌ Error checking database: {str(e)}"
                    
                    status_btn.click(check_database_status, inputs=[], outputs=status_output)
        
        # Event handlers
        def respond(message, chat_history, use_context, max_length, temperature, save_conversation_flag):
            if not message.strip():
                return "", chat_history, update_stats()
            
            # Add user message to chat history (new format)
            chat_history.append({"role": "user", "content": message})
            
            # Generate AI response
            response = app.generate_response(
                message, 
                use_context, 
                max_length, 
                temperature,
                save_conversation_flag
            )
            
            # Add AI response to chat history (new format)
            chat_history.append({"role": "assistant", "content": response})
            
            # Update stats after response
            stats = update_stats()
            
            return "", chat_history, stats
        
        # Update the response function to also return stats
        msg.submit(
            respond,
            [msg, chatbot, use_context, max_length, temperature, save_conversation],
            [msg, chatbot, stats_output]
        )
        
        send_btn.click(
            respond,
            [msg, chatbot, use_context, max_length, temperature, save_conversation],
            [msg, chatbot, stats_output]
        )
        
        clear_chat.click(
            lambda: ([], app.clear_current_conversation(), update_stats()),
            inputs=[],
            outputs=[chatbot, stats_output]
        )
        
        search_btn.click(
            lambda query, search_type: app.search_knowledge_base(query, search_type),
            inputs=[search_query, search_type],
            outputs=search_results
        )
    
    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        # server_name="0.0.0.0",
        # server_port=7860,
        # share=False
    )