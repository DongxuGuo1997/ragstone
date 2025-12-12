"""
Main entry point for the LangChain RAG Pipeline Streamlit application.

This module provides a web interface for the RAG pipeline with support for:
- Multiple LLM providers (OpenAI, Ollama)
- Various document sources (local files, URLs, Wikipedia)
- Interactive chat interface with memory
- Advanced RAG techniques
"""

import argparse
import asyncio
import logging
import os
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple

import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

# Add the src directory to the path for imports
current_dir = Path(__file__).parent
src_dir = current_dir.parent.parent.parent
sys.path.insert(0, str(src_dir))

from langchain_rag.rag.pipeline import Pipeline, OpenAIPipeline, OllamaPipeline
from langchain_rag import PACKAGE_INFO, get_package_info
from langchain_rag.config.settings import get_config, Config

# Local exception definitions to avoid import issues
class PipelineError(Exception):
    """Base pipeline error."""
    pass

class LLMInitializationError(Exception):
    """LLM initialization error."""
    pass

class ConfigurationError(Exception):
    """Configuration error."""
    pass

class VectorStoreError(Exception):
    """Vector store error."""
    pass

class LoaderError(Exception):
    """Loader error."""
    pass

def safe_execute(func, *args, **kwargs):
    """Execute function safely."""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        return None

def format_exception_context(exception, context=None):
    """Format exception with context."""
    return str(exception)

# Initialize configuration first
config: Config = get_config()
logger = logging.getLogger(__name__)


class StreamlitApp:
    """Main Streamlit application class for the RAG pipeline."""
    
    def __init__(self):
        """Initialize the Streamlit application."""
        self.config = get_config()
        self._setup_page_config()
        self._initialize_session_state()
    
    def _setup_page_config(self) -> None:
        """Configure Streamlit page settings."""
        st.set_page_config(
            page_title=self.config.ui.title,
            page_icon=self.config.ui.page_icon,
            layout=self.config.ui.layout,
            initial_sidebar_state=self.config.ui.initial_sidebar_state
        )
    
    def _initialize_session_state(self) -> None:
        """Initialize Streamlit session state variables."""
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "pipeline" not in st.session_state:
            st.session_state.pipeline = None
        if "uploaded_files" not in st.session_state:
            st.session_state.uploaded_files = None
    
    def run(self) -> None:
        """Run the main application."""
        try:
            self._render_header()
            self._render_sidebar()
            self._render_main_content()
        except Exception as e:
            logger.error(f"Application error: {e}", exc_info=True)
            st.error(f"An unexpected error occurred: {str(e)}")
    
    def _render_header(self) -> None:
        """Render the application header."""
        st.title(self.config.ui.title)
        
        # Package info in expander
        with st.expander("📋 System Information", expanded=False):
            package_info = get_package_info()
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Version:** {package_info['version']}")
                st.write(f"**Environment:** {self.config.environment}")
                st.write(f"**Debug Mode:** {'✅' if self.config.debug else '❌'}")
            
            with col2:
                st.write(f"**Vector Store:** {self.config.database.default_type}")
                st.write(f"**Batch Size:** {self.config.database.batch_size}")
                st.write(f"**Similarity K:** {self.config.database.similarity_k}")
    
    def _render_sidebar(self) -> None:
        """Render the sidebar with configuration options."""
        with st.sidebar:
            st.header("⚙️ Configuration")
            
            # Model selection
            mode = self._select_mode()
            if not mode:
                return
                
            model = self._select_model(mode)
            if not model:
                return
            
            # Data sources
            st.subheader("📂 Data Sources")
            uploaded_files, page_urls, wiki_query = self._get_data_inputs()
            
            # Advanced settings
            with st.expander("🔧 Advanced Settings"):
                use_ensemble = st.checkbox(
                    "Use Ensemble Retriever (BM25 + Vector)", 
                    value=True,
                    help="Combines BM25 and vector similarity for better retrieval"
                )
                
                chain_type = st.selectbox(
                    "RAG Chain Type",
                    ["simple", "multi_query", "fusion"],
                    help="Choose the RAG technique to use"
                )
                
                temperature = st.slider(
                    "LLM Temperature", 
                    0.0, 2.0, 
                    self.config.llm.default_temperature,
                    0.1,
                    help="Controls randomness in responses (0.0 = deterministic, 2.0 = very creative)"
                )
            
            # Build pipeline button
            if st.button("🚀 Build Pipeline", type="primary"):
                self._build_pipeline(
                    mode=mode,
                    model=model,
                    uploaded_files=uploaded_files,
                    page_urls=page_urls,
                    wiki_query=wiki_query,
                    use_ensemble=use_ensemble,
                    chain_type=chain_type,
                    temperature=temperature
                )
    
    def _render_main_content(self) -> None:
        """Render the main content area."""
        if st.session_state.pipeline is None:
            self._render_welcome_message()
        else:
            self._render_chat_interface()
    
    def _render_welcome_message(self) -> None:
        """Render welcome message when no pipeline is loaded."""
        st.markdown("""
        ## 🚀 Welcome to the LangChain RAG Pipeline!
        
        To get started:
        1. **Select a mode** (Online/Offline) in the sidebar
        2. **Choose a model** based on your preference
        3. **Add data sources** (upload files, provide URLs, or use Wikipedia)
        4. **Configure advanced settings** if needed
        5. **Click "Build Pipeline"** to initialize the system
        
        ### 📚 Supported Data Sources:
        - **Local Files**: PDF, TXT, CSV, DOCX, MD
        - **Web Pages**: Any public URL
        - **Wikipedia**: Search and load articles
        
        ### 🤖 Available Models:
        - **OpenAI**: GPT-3.5 Turbo, GPT-4, GPT-4o Mini
        - **Ollama**: Llama3, Phi4, DeepSeek-R1
        """)
    
    def _render_chat_interface(self) -> None:
        """Render the chat interface."""
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask your question here..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("🤔 Thinking..."):
                    start_time = time.time()
                    try:
                        response = st.session_state.pipeline.ask_question(prompt)
                        response_time = time.time() - start_time
                        
                        if response:
                            st.markdown(response)
                            if response_time < 0.5:
                                st.info(f"⚡ **Ultra-fast response!** Answered in {response_time:.2f}s")
                            elif response_time < 2.0:
                                st.info(f"🚀 **Fast response!** Answered in {response_time:.2f}s")
                            else:
                                st.info(f"🤔 **Response** in {response_time:.2f}s")
                            
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": response
                            })
                        else:
                            error_msg = "Sorry, I couldn't generate a response. Please try again."
                            st.error(error_msg)
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": error_msg
                            })
                    except Exception as e:
                        error_msg = f"Error generating response: {str(e)}"
                        logger.error(error_msg, exc_info=True)
                        st.error(error_msg)
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": error_msg
                        })
        
        # Clear chat button
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()
    
    def _select_mode(self) -> Optional[str]:
        """Allow user to select pipeline mode."""
        mode = st.selectbox(
            "🔌 Pipeline Mode",
            ["", "online", "offline"],
            index=0,
            help="Online: Uses cloud APIs (OpenAI). Offline: Uses local models (Ollama)"
        )
        
        if mode == "":
            st.info("👆 Please select a pipeline mode to continue")
            return None
            
        return mode
    
    def _select_model(self, mode: str) -> Optional[str]:
        """Allow user to select model based on mode."""
        if mode == "online":
            models = self.config.llm.openai_models
            if not self.config.api.openai_api_key:
                st.error("🔑 OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
                return None
        else:
            models = self.config.llm.ollama_models
        
        model = st.selectbox(
            f"🤖 {'OpenAI' if mode == 'online' else 'Ollama'} Model",
            [""] + models,
            index=0,
            help=f"Select the {mode} model to use"
        )
        
        if model == "":
            st.info("👆 Please select a model to continue")
            return None
            
        return model
    
    def _get_data_inputs(self) -> Tuple[Optional[List[UploadedFile]], Optional[List[str]], Optional[str]]:
        """Get data source inputs from user."""
        # File upload
        uploaded_files = st.file_uploader(
            "📁 Upload Files",
            type=["txt", "pdf", "csv", "docx", "md"],
            accept_multiple_files=True,
            help="Upload documents to include in the knowledge base"
        )
        
        # Store in session state for caching
        if uploaded_files:
            st.session_state.uploaded_files = uploaded_files
        
        # URL input
        url_input = st.text_area(
            "🌐 Web URLs",
            placeholder="Enter URLs, one per line...",
            help="Provide web page URLs to scrape and include"
        )
        page_urls = [url.strip() for url in url_input.split('\n') if url.strip()] if url_input else None
        
        # Wikipedia query
        wiki_query = st.text_input(
            "📖 Wikipedia Query",
            placeholder="Enter search term for Wikipedia articles...",
            help="Search Wikipedia and include relevant articles"
        )
        
        return uploaded_files, page_urls, wiki_query if wiki_query else None
    
    def _build_pipeline(
        self, 
        mode: str, 
        model: str, 
        uploaded_files: Optional[List[UploadedFile]], 
        page_urls: Optional[List[str]], 
        wiki_query: Optional[str],
        use_ensemble: bool,
        chain_type: str,
        temperature: float
    ) -> None:
        """Build the RAG pipeline with the given configuration."""
        try:
            with st.spinner("🔧 Building pipeline..."):
                # Validate inputs
                if not any([uploaded_files, page_urls, wiki_query]):
                    st.warning("⚠️ Please provide at least one data source")
                    return
                
                # Build pipeline
                if mode == "online":
                    pipeline = self._build_openai_pipeline(
                        model, uploaded_files, page_urls, wiki_query, temperature
                    )
                else:
                    pipeline = self._build_ollama_pipeline(
                        model, uploaded_files, page_urls, wiki_query, temperature
                    )
                
                if pipeline:
                    # Set retriever
                    if mode == "online":
                        pipeline.set_retriever_openai(use_ensemble=use_ensemble)
                    else:
                        pipeline.set_retriever_ollama(use_ensemble=use_ensemble)
                    
                    # Create RAG chain
                    pipeline.create_rag_chain(chain_type=chain_type)
                    
                    # Store in session state
                    st.session_state.pipeline = pipeline
                    st.session_state.messages = [{
                        "role": "assistant", 
                        "content": f"🎉 Pipeline built successfully! I'm ready to answer questions about your documents. Using {model} with {chain_type} RAG chain."
                    }]
                    
                    st.success("✅ Pipeline built successfully!")
                    st.rerun()
                
        except Exception as e:
            error_msg = f"Failed to build pipeline: {str(e)}"
            logger.error(error_msg, exc_info=True)
            st.error(error_msg)
    
    def _build_openai_pipeline(
        self, 
        model: str, 
        uploaded_files: Optional[List[UploadedFile]], 
        page_urls: Optional[List[str]], 
        wiki_query: Optional[str],
        temperature: float
    ) -> Optional[OpenAIPipeline]:
        """Build OpenAI pipeline."""
        try:
            pipeline = OpenAIPipeline(model=model)
            pipeline.LLM.set_llm(model, temperature=temperature)
            
            # Load and split documents
            texts = pipeline.load_and_split(
                data_dir=self.config.loader.default_data_dir,
                uploaded_files=uploaded_files,
                page_urls=page_urls,
                wiki_query=wiki_query
            )
            
            if not texts:
                st.warning("⚠️ No documents were loaded. Please check your data sources.")
                return None
                
            logger.info(f"OpenAI pipeline built with {len(texts)} document chunks")
            return pipeline
            
        except Exception as e:
            raise LLMInitializationError(
                f"Failed to build OpenAI pipeline: {str(e)}",
                model_name=model,
                provider="openai"
            )
    
    def _build_ollama_pipeline(
        self, 
        model: str, 
        uploaded_files: Optional[List[UploadedFile]], 
        page_urls: Optional[List[str]], 
        wiki_query: Optional[str],
        temperature: float
    ) -> Optional[OllamaPipeline]:
        """Build Ollama pipeline."""
        try:
            pipeline = OllamaPipeline(model=model)
            pipeline.LLM.set_llm(model, temperature=temperature)
            
            # Load and split documents
            texts = pipeline.load_and_split(
                data_dir=self.config.loader.default_data_dir,
                uploaded_files=uploaded_files,
                page_urls=page_urls,
                wiki_query=wiki_query
            )
            
            if not texts:
                st.warning("⚠️ No documents were loaded. Please check your data sources.")
                return None
                
            logger.info(f"Ollama pipeline built with {len(texts)} document chunks")
            return pipeline
            
        except Exception as e:
            raise LLMInitializationError(
                f"Failed to build Ollama pipeline: {str(e)}",
                model_name=model,
                provider="ollama"
            )


def configure_logging():
    """Configure application-wide logging with argument parsing."""
    parser = argparse.ArgumentParser(add_help=False)  # Don't interfere with Streamlit
    parser.add_argument(
        "--log", 
        default="INFO", 
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set logging level"
    )
    
    try:
        args, _ = parser.parse_known_args()
        level_name = args.log.upper()
    except:
        level_name = "INFO"  # Fallback
    
    # Update config with command line level
    config.logging.level = level_name
    config.setup_logging()


def main():
    """Main entry point for the application."""
    try:
        # Configure logging first
        configure_logging()
        
        # Log startup information
        package_info = get_package_info()
        logger.info(f"Starting {package_info['name']} v{package_info['version']}")
        logger.info(f"Environment: {config.environment}, Debug: {config.debug}")
        
        # Create and run the Streamlit app
        app = StreamlitApp()
        app.run()
        
    except ConfigurationError as e:
        st.error(f"Configuration Error: {e.message}")
        logger.error(f"Configuration error: {e}")
        if config.debug:
            st.exception(e)
    except Exception as e:
        error_msg = f"Critical application error: {str(e)}"
        st.error(error_msg)
        logger.critical(error_msg, exc_info=True)
        if config.debug:
            st.exception(e)


if __name__ == "__main__":
    # Disable tokenizers parallelism for stability
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Run the application
    main()

# Helper function for package info
def get_package_info() -> Dict[str, Any]:
    """Get package information."""
    return PACKAGE_INFO
