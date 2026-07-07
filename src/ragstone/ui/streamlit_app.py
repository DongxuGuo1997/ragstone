"""
Main entry point for the Ragstone Streamlit application.

This module provides a web interface for the RAG pipeline with support for:
- Multiple LLM providers (OpenAI, Ollama)
- Various document sources (local files, URLs, Wikipedia)
- Interactive chat interface with memory
- Advanced RAG techniques
"""

import argparse
import logging
import os
import queue
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

# Allow `streamlit run src/ragstone/ui/streamlit_app.py` without install
current_dir = Path(__file__).parent
src_dir = current_dir.parent.parent.parent
sys.path.insert(0, str(src_dir))

from ragstone import get_package_info  # noqa: E402
from ragstone.config.settings import Config, get_config  # noqa: E402
from ragstone.rag.pipeline import OllamaPipeline, OpenAIPipeline  # noqa: E402
from ragstone.utils import (  # noqa: E402
    ConfigurationError,
    LLMInitializationError,
)
from ragstone.utils.observability import (  # noqa: E402
    estimate_cost_usd,
    track_request,
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
        defaults: Dict[str, Any] = {
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
                    ["simple", "multi_query", "fusion", "agent", "corrective"],
                    help=(
                        "Choose the RAG technique to use. 'agent' lets the "
                        "LLM drive retrieval via a search tool (slower, "
                        "more LLM calls)."
                    ),
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

                # Apply only on change. Note: this writes the process-global
                # config — fine for a single-user app; a multi-user
                # deployment should move the flag into session state.
                if cache_enabled != self.config.cache.enable_response_cache:
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
            return
        # A radio, not st.tabs: st.chat_input only stays pinned to the
        # bottom of the page when rendered at the app root — inside a tab
        # container it renders inline, which puts the input box above
        # freshly streamed answers.
        view = st.radio(
            "View",
            ["💬 Chat", "⚔️ Compare"],
            horizontal=True,
            label_visibility="collapsed",
        )
        if view == "💬 Chat":
            self._render_chat_interface()
        else:
            self._render_compare_interface()

    def _render_welcome_message(self) -> None:
        """Render welcome message when no pipeline is loaded."""
        st.markdown("""
        ## 🚀 Welcome to the Ragstone!

        ### 🔌 **Choose Your Mode:**

        **🌐 Online Mode** (Cloud-based)
        - Uses OpenAI models (GPT-4o Mini, GPT-4o, GPT-4.1)
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

    def _collect_trace(self, prompt: str) -> Dict[str, Any]:
        """Assemble the glass-box trace for the answer just generated.

        Reads state the pipeline already recorded (request metrics, the
        rephrased question, the retrieval record) — no extra LLM or
        embedding calls are made here.
        """
        pipeline = st.session_state.pipeline
        trace: Dict[str, Any] = {
            "sources": pipeline.get_sources(prompt),
            "interpretation": None,
            "chain_type": None,
            "latency_ms": None,
            "tokens": None,
            "cost_usd": None,
            "cache_hit": False,
        }

        interpretation = pipeline.get_last_interpretation(
            st.session_state.chat_session_id
        )
        # Only show the interpretation when the rephrase step actually
        # changed the question — echoing it back verbatim teaches nothing.
        if interpretation and interpretation.strip() != prompt.strip():
            trace["interpretation"] = interpretation

        metrics = pipeline.last_metrics
        if metrics is not None:
            trace["chain_type"] = metrics.chain_type
            trace["latency_ms"] = metrics.latency_ms
            trace["first_token_ms"] = metrics.first_token_ms
            trace["tokens"] = metrics.tokens
            trace["cache_hit"] = metrics.cache_hit
            if metrics.stage_ms:
                trace["stage_ms"] = dict(metrics.stage_ms)
                trace["generation_ms"] = metrics.generation_ms
            model = pipeline.LLM.get_model_name() if pipeline.LLM else None
            trace["cost_usd"] = estimate_cost_usd(
                metrics.input_tokens, metrics.output_tokens, model
            )
        return trace

    @staticmethod
    def _render_trace(trace: Dict[str, Any]) -> None:
        """Render the 'how this answer was made' panel under an answer."""
        with st.expander("🔍 How this answer was made"):
            if trace.get("interpretation"):
                st.markdown(
                    f"**Interpreted your question as:** _{trace['interpretation']}_"
                )

            badge_parts = []
            if trace.get("first_token_ms") is not None:
                badge_parts.append(
                    f"⚡ first token {trace['first_token_ms'] / 1000:.2f}s"
                )
            if trace.get("latency_ms") is not None:
                badge_parts.append(f"⏱ total {trace['latency_ms'] / 1000:.2f}s")
            if trace.get("cache_hit"):
                badge_parts.append("♻️ served from cache")
            elif trace.get("tokens"):
                badge_parts.append(f"🔤 {trace['tokens']:,} tokens")
                if trace.get("cost_usd") is not None:
                    badge_parts.append(f"💰 ~${trace['cost_usd']:.4f}")
            if trace.get("chain_type"):
                badge_parts.append(f"⛓ chain: {trace['chain_type']}")
            if badge_parts:
                st.caption(" · ".join(badge_parts))

            # Stage decomposition: WHERE the time went, when recorded.
            if trace.get("stage_ms"):
                stage_parts = [
                    f"{name} {ms / 1000:.2f}s"
                    for name, ms in sorted(trace["stage_ms"].items())
                ]
                if trace.get("generation_ms"):
                    stage_parts.append(
                        f"generation {trace['generation_ms'] / 1000:.2f}s"
                    )
                st.caption("🧭 " + " → ".join(stage_parts))

            sources = trace.get("sources") or []
            if sources:
                st.markdown(f"**📄 Sources ({len(sources)})**")
                for src in sources:
                    st.markdown(f"**{src['source']}**")
                    st.caption(src["snippet"])

    @staticmethod
    def _event_line(event: Dict[str, Any]) -> Optional[str]:
        """Human-readable line for a progress event, or None to skip it."""
        kind = event.get("event")
        if kind == "search":
            return f'🔍 Searching: "{event.get("query", "")}"'
        if kind == "retrieve":
            return f'📥 Retrieving: "{event.get("query", "")}"'
        if kind == "grade":
            verdict = "relevant ✓" if event.get("relevant") else "irrelevant ✗"
            return f"⚖️ Graded results: {verdict}"
        if kind == "rewrite":
            return f'✏️ Rewriting query: "{event.get("query", "")}"'
        return None

    def _stream_answer(self, prompt: str) -> str:
        """Stream the answer, rendering progress events live.

        Text chunks accumulate into the answer; dict events (agent
        searches, corrective-loop grades/rewrites) render as status lines
        above it, so the user watches the system think before the answer
        streams in.
        """
        status = None
        placeholder = st.empty()
        parts: List[str] = []
        last_render = 0.0

        for chunk in st.session_state.pipeline.ask_question_stream(
            prompt, session_id=st.session_state.chat_session_id
        ):
            if isinstance(chunk, dict):
                line = self._event_line(chunk)
                if line:
                    if status is None:
                        status = st.status("🤖 Working…", expanded=True)
                    status.write(line)
            elif chunk:
                parts.append(chunk)
                # Throttle redraws: a markdown round-trip per token adds
                # real latency to the measured window; ~20 fps is plenty.
                now = time.perf_counter()
                if now - last_render > 0.05:
                    placeholder.markdown("".join(parts) + "▌")
                    last_render = now

        if status is not None:
            status.update(label="🤖 Working", state="complete", expanded=False)
        answer = "".join(parts)
        placeholder.markdown(answer)
        return answer

    def _get_compare_chain(self, chain_type: str):
        """Chain variant for the compare tab, cached per pipeline build.

        Variants share the pipeline's retriever and vector store (no
        re-embedding); the cache is invalidated when the pipeline object
        changes (i.e. after a rebuild in the sidebar).
        """
        pipeline = st.session_state.pipeline
        cache = st.session_state.get("compare_chains")
        if not cache or cache.get("pipeline") is not pipeline:
            cache = {"pipeline": pipeline, "chains": {}}
            st.session_state.compare_chains = cache
        if chain_type not in cache["chains"]:
            cache["chains"][chain_type] = pipeline.make_chain_variant(chain_type)
        return cache["chains"][chain_type]

    @staticmethod
    def _stream_chain_to_queue(chain, chain_type, question, session_id, out_queue):
        """Worker: stream one chain's answer into a queue.

        Runs in a background thread, so it must never touch st.*; the main
        thread drains the queue and renders. track_request gives the
        column its own latency/token metrics (the usage callback is
        contextvar-based, so per-thread isolation is exact).
        """
        metrics = None
        try:
            with track_request(session_id, chain_type=chain_type) as metrics:
                stream_start = time.perf_counter()
                for chunk in chain.stream_question(question, session_id):
                    if isinstance(chunk, str) and metrics.first_token_ms is None:
                        metrics.first_token_ms = int(
                            (time.perf_counter() - stream_start) * 1000
                        )
                    out_queue.put(("chunk", chunk))
        except Exception as exc:  # surfaced in the column, not swallowed
            out_queue.put(("error", str(exc)))
        finally:
            out_queue.put(("done", metrics))

    def _render_compare_interface(self) -> None:
        """Side-by-side comparison: one question, two chain types."""
        st.caption(
            "One question, two techniques, the same corpus. Judge the "
            "answers yourself — latency and cost are measured for you."
        )
        options = ["simple", "multi_query", "fusion", "agent", "corrective"]
        select_left, select_right = st.columns(2)
        with select_left:
            left = st.selectbox("Left chain", options, index=0, key="cmp_left")
        with select_right:
            right = st.selectbox("Right chain", options, index=3, key="cmp_right")

        question = st.text_input(
            "Question for both chains",
            key="cmp_question",
            placeholder="e.g. What does fault code E-42 mean?",
        )
        if not st.button("⚔️ Run comparison", disabled=not question.strip()):
            return

        try:
            chains = [self._get_compare_chain(ct) for ct in (left, right)]
        except Exception as e:
            st.error(f"Could not build the chains: {e}")
            return

        run_id = uuid.uuid4().hex[:6]
        labels = (left, right)
        queues: List[queue.Queue] = [queue.Queue(), queue.Queue()]
        for chain, label, out_queue in zip(chains, labels, queues):
            threading.Thread(
                target=self._stream_chain_to_queue,
                args=(chain, label, question, f"cmp-{run_id}-{label}", out_queue),
                daemon=True,
            ).start()

        columns = st.columns(2)
        bodies, footers = [], []
        for column, label in zip(columns, labels):
            with column:
                st.subheader(f"⛓ {label}")
                bodies.append(st.empty())
                footers.append(st.empty())

        events: List[List[str]] = [[], []]
        parts: List[List[str]] = [[], []]
        metrics_by_column: List[Any] = [None, None]
        finished = [False, False]

        def render_column(i: int, streaming: bool) -> None:
            blocks = []
            if events[i]:
                blocks.append("\n".join(events[i]))
            text = "".join(parts[i])
            blocks.append(text + ("▌" if streaming else ""))
            bodies[i].markdown("\n\n".join(blocks))

        while not all(finished):
            progressed = False
            for i in (0, 1):
                if finished[i]:
                    continue
                while True:
                    try:
                        kind, payload = queues[i].get_nowait()
                    except queue.Empty:
                        break
                    progressed = True
                    if kind == "chunk":
                        if isinstance(payload, dict):
                            line = self._event_line(payload)
                            if line:
                                events[i].append(f"_{line}_")
                        else:
                            parts[i].append(payload)
                    elif kind == "error":
                        events[i].append(f"❌ {payload}")
                    elif kind == "done":
                        finished[i] = True
                        metrics_by_column[i] = payload
                if progressed:
                    render_column(i, streaming=not finished[i])
            if not progressed:
                time.sleep(0.05)

        model = (
            st.session_state.pipeline.LLM.get_model_name()
            if st.session_state.pipeline.LLM
            else None
        )
        for i, label in enumerate(labels):
            m = metrics_by_column[i]
            if m is None:
                continue
            badge = []
            if m.first_token_ms is not None:
                badge.append(f"⚡ first token {m.first_token_ms / 1000:.2f}s")
            badge.append(f"⏱ total {m.latency_ms / 1000:.2f}s")
            badge.append(f"🔤 {m.tokens:,} tokens")
            cost = estimate_cost_usd(m.input_tokens, m.output_tokens, model)
            if cost is not None:
                badge.append(f"💰 ~${cost:.4f}")
            footers[i].caption(" · ".join(badge))

        m0, m1 = metrics_by_column
        if m0 and m1 and not (m0.error or m1.error) and m0.latency_ms and m1.latency_ms:
            faster_idx = 0 if m0.latency_ms <= m1.latency_ms else 1
            ratio = max(m0.latency_ms, m1.latency_ms) / max(
                1, min(m0.latency_ms, m1.latency_ms)
            )
            st.success(
                f"⚡ **{labels[faster_idx]}** answered {ratio:.1f}× faster "
                f"({labels[0]}: {m0.tokens:,} vs {labels[1]}: {m1.tokens:,} tokens). "
                "Same corpus, same question — the quality difference is yours to judge."
            )

    def _render_chat_interface(self) -> None:
        """Render the chat interface."""
        # Display chat messages (with their glass-box traces, if recorded)
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message.get("trace"):
                    self._render_trace(message["trace"])

        # Chat input
        if prompt := st.chat_input("Ask your question here..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate response
            with st.chat_message("assistant"):
                try:
                    response = self._stream_answer(prompt)

                    if response:
                        trace = self._collect_trace(prompt)
                        self._render_trace(trace)
                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": response,
                                "trace": trace,
                            }
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
                pipeline: Union[OpenAIPipeline, OllamaPipeline, None]
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
                        if isinstance(pipeline, OpenAIPipeline):
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
            assert pipeline.LLM is not None  # set by the constructor
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
            assert pipeline.LLM is not None  # set by the constructor
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
        page_title="Ragstone",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    try:
        # Configure logging once per session — main() reruns on every
        # interaction, and setup_logging() tears down and re-adds the root
        # handlers (reopening the log file) each time it runs.
        if not st.session_state.get("logging_configured"):
            configure_logging()
            st.session_state.logging_configured = True

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
