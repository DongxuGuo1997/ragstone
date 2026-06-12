"""
Main entry point for the Know-RAG Streamlit application.

This module provides a web interface for the RAG pipeline with support for:
- Multiple LLM providers (OpenAI, Ollama)
- Various document sources (local files, URLs, Wikipedia)
- Interactive chat interface with memory
- Advanced RAG techniques
"""

import argparse
import logging
import os
import sys
import uuid
from pathlib import Path
from typing import List, Optional, Tuple

import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

# Allow `streamlit run src/know_rag/ui/streamlit_app.py` without install
current_dir = Path(__file__).parent
src_dir = current_dir.parent.parent.parent
sys.path.insert(0, str(src_dir))

from know_rag import get_package_info  # noqa: E402
from know_rag.config.settings import Config, get_config  # noqa: E402
from know_rag.rag.pipeline import OllamaPipeline, OpenAIPipeline  # noqa: E402
from know_rag.utils import (  # noqa: E402
    ConfigurationError,
    LLMInitializationError,
)

# Initialize configuration and logger first
config: Config = get_config()
logger = logging.getLogger(__name__)

# Streamlit's file watcher walks every imported module looking for source
# files. With the `rerank` extra installed, that walk touches transformers'
# lazy module registry and logs a harmless traceback for each vision model
# that would need torchvision. Silence just that watcher logger.
logging.getLogger("streamlit.watcher.local_sources_watcher").setLevel(logging.ERROR)


def check_openai_connectivity() -> bool:
    """Check if OpenAI API is accessible."""
    try:
        import requests

        if not config.api.openai_api_key:
            return False

        headers = {
            "Authorization": f"Bearer {config.api.openai_api_key}",
            "Content-Type": "application/json",
        }

        response = requests.get(
            "https://api.openai.com/v1/models", headers=headers, timeout=3
        )

        return response.status_code == 200

    except Exception:
        return False


def get_available_ollama_models() -> List[str]:
    """Get list of available Ollama models."""
    try:
        import requests

        base_url = get_config().api.ollama_base_url
        response = requests.get(f"{base_url}/api/tags", timeout=3)
        if response.status_code == 200:
            models_data = response.json().get("models", [])
            return [model["name"] for model in models_data]
        else:
            return []

    except Exception:
        return []


class StreamlitApp:
    """Main Streamlit application class for the RAG pipeline."""

    def __init__(self):
        """Initialize the Streamlit application."""
        self.config = config
        self._initialize_session_state()
        self._check_services_once()

    def _initialize_session_state(self) -> None:
        """Initialize Streamlit session state variables."""
        # Initialize session state variables only if not already set
        defaults = {
            "messages": [],
            "pipeline": None,
            "uploaded_files": None,
            "pipeline_built": False,
            "current_model": None,
            "current_mode": None,
            "openai_available": None,
            "ollama_models": None,
            "status_checked": False,
            "chat_session_id": str(uuid.uuid4()),
        }

        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value

    def _check_services_once(self) -> None:
        """Check OpenAI and Ollama services once at startup."""
        if not st.session_state.status_checked:
            # Check OpenAI
            st.session_state.openai_available = check_openai_connectivity()

            # Check Ollama
            st.session_state.ollama_models = get_available_ollama_models()

            # Mark as checked
            st.session_state.status_checked = True

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

            # Show current pipeline status
            if st.session_state.pipeline_built:
                st.info(
                    f"🤖 **Active Pipeline:** {st.session_state.current_mode} mode with {st.session_state.current_model}"
                )

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
                st.info(
                    "💡 Built-in 3-tier caching system is always enabled for optimal performance"
                )

                use_ensemble = st.checkbox(
                    "Use Ensemble Retriever (BM25 + Vector)",
                    value=True,
                    help="Combines BM25 and vector similarity for better retrieval",
                )

                use_reranker = st.checkbox(
                    "Use Reranker (cross-encoder)",
                    value=False,
                    help=(
                        "Rescore a wide candidate pool with a local cross-encoder "
                        'for better precision. Requires: pip install -e ".[rerank]"'
                    ),
                )

                chain_type = st.selectbox(
                    "RAG Chain Type",
                    ["simple", "multi_query", "fusion"],
                    help="Choose the RAG technique to use",
                )

                temperature = st.slider(
                    "LLM Temperature",
                    0.0,
                    2.0,
                    self.config.llm.default_temperature,
                    0.1,
                    help="Controls randomness in responses (0.0 = deterministic, 2.0 = very creative)",
                )

                # Simplified Cache Setting (Optional)
                cache_enabled = st.checkbox(
                    "Enable Response Cache",
                    value=self.config.cache.enable_response_cache,
                    help="Cache answers to repeated identical questions (optional)",
                )

                # Update config with new value
                self.config.cache.enable_response_cache = cache_enabled

            # Build pipeline button (with state tracking to avoid unnecessary rebuilds)
            if st.button("🚀 Build Pipeline", type="primary"):
                # Only rebuild if settings OR data sources changed. The
                # fingerprint must include the data sources, otherwise a
                # newly uploaded file is silently ignored.
                files_sig = ",".join(
                    sorted(f"{f.name}:{f.size}" for f in (uploaded_files or []))
                )
                current_settings = (
                    f"{mode}|{model}|{use_ensemble}|{use_reranker}"
                    f"|{chain_type}|{temperature}"
                    f"|{files_sig}|{','.join(page_urls or [])}|{wiki_query or ''}"
                )
                if (
                    not st.session_state.pipeline_built
                    or st.session_state.get("last_settings") != current_settings
                ):
                    logger.info(f"Building pipeline with settings: {current_settings}")
                    self._build_pipeline(
                        mode=mode,
                        model=model,
                        uploaded_files=uploaded_files,
                        page_urls=page_urls,
                        wiki_query=wiki_query,
                        use_ensemble=use_ensemble,
                        use_reranker=use_reranker,
                        chain_type=chain_type,
                        temperature=temperature,
                        settings_fingerprint=current_settings,
                    )
                else:
                    st.info("✅ Pipeline already built with these settings!")

    def _render_main_content(self) -> None:
        """Render the main content area."""
        if st.session_state.pipeline is None:
            self._render_welcome_message()
        else:
            self._render_chat_interface()

    def _render_welcome_message(self) -> None:
        """Render welcome message when no pipeline is loaded."""
        st.markdown("""
        ## 🚀 Welcome to the Know-RAG!

        ### 🔌 **Choose Your Mode:**

        **🌐 Online Mode** (Cloud-based)
        - Uses OpenAI models (GPT-3.5, GPT-4)
        - Requires internet connection and API key
        - Fast and powerful responses

        **🦙 Offline Mode** (Local, Private)
        - Uses Ollama models (Llama3, Phi4, DeepSeek-R1)
        - Works without internet connection
        - Complete privacy - data stays on your machine
        - **⚡ Recommended if you have network issues**

        ### 🚀 **Quick Start:**
        1. **Select a mode** in the sidebar (Online/Offline)
        2. **Choose a model** from available options
        3. **Add data sources** (upload files, URLs, or Wikipedia)
        4. **Click "Build Pipeline"** to start

        ### 📚 **Supported Data Sources:**
        - **Local Files**: PDF, TXT, CSV, DOCX, MD
        - **Web Pages**: Any public URL
        - **Wikipedia**: Search and load articles

        ### 💡 **Troubleshooting:**
        - **OpenAI not working?** → Try Offline mode with Ollama
        - **Ollama not available?** → Run `ollama serve` and `ollama pull llama3`
        - **Pipeline stuck?** → Check your internet connection or switch modes
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
                import time

                start_time = time.time()

                try:
                    response = st.write_stream(
                        st.session_state.pipeline.ask_question_stream(
                            prompt, session_id=st.session_state.chat_session_id
                        )
                    )
                    response_time = time.time() - start_time

                    if response:
                        # Show the retrieved sources behind the answer
                        sources = st.session_state.pipeline.get_sources(prompt)
                        if sources:
                            with st.expander(f"📄 Sources ({len(sources)})"):
                                for src in sources:
                                    st.markdown(f"**{src['source']}**")
                                    st.caption(src["snippet"])

                        # Add response time indicator
                        if response_time < 0.5:
                            st.info(
                                f"⚡ **Ultra-fast response!** Answered in {response_time:.2f}s"
                            )
                        elif response_time < 2.0:
                            st.info(
                                f"🚀 **Fast response!** Answered in {response_time:.2f}s"
                            )
                        else:
                            st.info(f"🤔 **Response** in {response_time:.2f}s")

                        st.session_state.messages.append(
                            {"role": "assistant", "content": response}
                        )
                    else:
                        error_msg = (
                            "Sorry, I couldn't generate a response. Please try again."
                        )
                        st.error(error_msg)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": error_msg}
                        )
                except Exception as e:
                    error_msg = f"Error generating response: {str(e)}"
                    logger.error(error_msg, exc_info=True)
                    st.error(error_msg)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": error_msg}
                    )

        # Action buttons at the bottom
        col1, col2 = st.columns(2)

        with col1:
            if st.button("🗑️ Clear Chat"):
                st.session_state.messages = []
                st.rerun()

        with col2:
            if st.button("🧹 Clear Cache"):
                if hasattr(st.session_state.pipeline, "clear_cache"):
                    st.session_state.pipeline.clear_cache()
                    st.success("🗑️ Cache cleared successfully!")
                    st.rerun()
                else:
                    st.warning("Cache not available for this pipeline")

    def _select_mode(self) -> Optional[str]:
        """Allow user to select pipeline mode with smart defaults."""

        # Use cached status from session state
        openai_available = st.session_state.openai_available

        # Set default mode based on availability
        if openai_available:
            mode_options = ["", "online", "offline"]
            help_text = (
                "Online: Uses cloud APIs (OpenAI). Offline: Uses local models (Ollama)"
            )
        else:
            mode_options = ["", "offline", "online"]
            help_text = (
                "⚠️ OpenAI connectivity issues detected. Offline mode recommended."
            )
            # Only show warning once per session
            if not st.session_state.get("openai_warning_shown", False):
                st.warning(
                    "🌐 **OpenAI API connectivity issues detected**\n\n"
                    "✅ **Recommendation:** Use **Offline mode** with Ollama models\n"
                    "- Works without internet connectivity\n"
                    "- Fast local processing\n"
                    "- Privacy-focused (no data leaves your machine)"
                )
                st.session_state.openai_warning_shown = True

        mode = st.selectbox("🔌 Pipeline Mode", mode_options, index=0, help=help_text)

        if mode == "":
            if openai_available:
                st.info("👆 Please select a pipeline mode to continue")
            else:
                st.info(
                    "👆 **Recommended:** Select **offline** mode for local processing"
                )
            return None

        return mode

    def _select_model(self, mode: str) -> Optional[str]:
        """Allow user to select model based on mode with availability checks."""
        if mode == "online":
            models = self.config.llm.openai_models
            if not self.config.api.openai_api_key:
                st.error(
                    "🔑 OpenAI API key not found. Please set OPENAI_API_KEY environment variable."
                )
                return None
        else:
            # Use cached Ollama models from session state
            available_models = st.session_state.ollama_models
            if available_models:
                models = available_models
                # Only show success message once per session
                if not st.session_state.get("ollama_success_shown", False):
                    st.success(
                        f"🦙 **Ollama Status:** {len(models)} models available locally"
                    )
                    st.session_state.ollama_success_shown = True
            else:
                models = self.config.llm.ollama_models
                # Only show warning once per session
                if not st.session_state.get("ollama_warning_shown", False):
                    st.warning(
                        "🦙 **Ollama Status:** Service not running or no models found\n\n"
                        "💡 **Quick Setup:**\n"
                        "1. Start Ollama: `ollama serve`\n"
                        "2. Pull a model: `ollama pull llama3`"
                    )
                    st.session_state.ollama_warning_shown = True

        model = st.selectbox(
            f"🤖 {'OpenAI' if mode == 'online' else 'Ollama'} Model",
            [""] + models,
            index=0,
            help=f"Select the {mode} model to use",
        )

        if model == "":
            st.info("👆 Please select a model to continue")
            return None

        return model

    def _get_data_inputs(
        self,
    ) -> Tuple[Optional[List[UploadedFile]], Optional[List[str]], Optional[str]]:
        """Get data source inputs from user."""
        # File upload
        uploaded_files = st.file_uploader(
            "📁 Upload Files",
            type=["txt", "pdf", "csv", "docx", "md"],
            accept_multiple_files=True,
            help="Upload documents to include in the knowledge base",
        )

        # Store in session state for caching
        if uploaded_files:
            st.session_state.uploaded_files = uploaded_files

        # URL input
        url_input = st.text_area(
            "🌐 Web URLs",
            placeholder="Enter URLs, one per line or separated by commas",
            help="Provide web page URLs to scrape and include",
        )
        page_urls = (
            [
                url.strip()
                for url in url_input.replace(",", "\n").split("\n")
                if url.strip()
            ]
            if url_input
            else None
        )

        # Wikipedia query
        wiki_query = st.text_input(
            "📖 Wikipedia Query",
            placeholder="Enter search term for Wikipedia articles...",
            help="Search Wikipedia and include relevant articles",
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
        use_reranker: bool,
        chain_type: str,
        temperature: float,
        settings_fingerprint: str = "",
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
                    try:
                        if mode == "online":
                            pipeline.set_retriever_openai(
                                use_ensemble=use_ensemble, use_reranker=use_reranker
                            )
                        else:
                            pipeline.set_retriever_ollama(
                                use_ensemble=use_ensemble, use_reranker=use_reranker
                            )
                    except ImportError as e:
                        # The rerank extra is not installed — explain and stop
                        st.error(f"❌ {e}")
                        return

                    # Create RAG chain
                    pipeline.create_rag_chain(chain_type=chain_type)

                    # Store in session state. Record the settings fingerprint
                    # only now, on success — a failed build must not block
                    # retries with the same settings.
                    st.session_state.pipeline = pipeline
                    st.session_state.pipeline_built = True
                    st.session_state.last_settings = settings_fingerprint
                    st.session_state.current_model = model
                    st.session_state.current_mode = mode
                    st.session_state.messages = [
                        {
                            "role": "assistant",
                            "content": f"🎉 Pipeline built successfully! I'm ready to answer questions about your documents. Using {model} with {chain_type} RAG chain.",
                        }
                    ]

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
        temperature: float,
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
                wiki_query=wiki_query,
            )

            if not texts:
                st.warning(
                    "⚠️ No documents were loaded. Please check your data sources."
                )
                return None

            logger.info(f"OpenAI pipeline built with {len(texts)} document chunks")
            return pipeline

        except Exception as e:
            raise LLMInitializationError(
                f"Failed to build OpenAI pipeline: {str(e)}",
                model_name=model,
                provider="openai",
            )

    def _build_ollama_pipeline(
        self,
        model: str,
        uploaded_files: Optional[List[UploadedFile]],
        page_urls: Optional[List[str]],
        wiki_query: Optional[str],
        temperature: float,
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
                wiki_query=wiki_query,
            )

            if not texts:
                st.warning(
                    "⚠️ No documents were loaded. Please check your data sources."
                )
                return None

            logger.info(f"Ollama pipeline built with {len(texts)} document chunks")
            return pipeline

        except Exception as e:
            raise LLMInitializationError(
                f"Failed to build Ollama pipeline: {str(e)}",
                model_name=model,
                provider="ollama",
            )


def configure_logging():
    """Configure application-wide logging with argument parsing."""
    parser = argparse.ArgumentParser(add_help=False)  # Don't interfere with Streamlit
    parser.add_argument(
        "--log",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set logging level",
    )

    try:
        args, _ = parser.parse_known_args()
        level_name = args.log.upper()
    except Exception:
        level_name = "INFO"  # Fallback

    # Update config with command line level
    config.logging.level = level_name
    config.setup_logging()


def main():
    """Main entry point for the application."""

    # CRITICAL: Set page config as the absolute first Streamlit command
    st.set_page_config(
        page_title="Know-RAG",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    try:
        # Configure logging only once
        configure_logging()

        # Log startup information once
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
