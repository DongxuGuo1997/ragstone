#!/usr/bin/env python3
"""Terminal chat for Ragstone — the CLI counterpart of the Streamlit app.

Answers stream token by token with live progress events (agent searches,
corrective grading), followed by the same glass-box trace the web UI
shows: first-token and total latency, tokens, estimated cost, cache hits,
and the rephrase → retrieval → generation stage breakdown. Slash commands
cover the rest of the feature set:

    /sources            citations behind the last answer
    /trace              re-print the last answer's trace
    /chain <type>       switch chain type (simple, multi_query, fusion,
                        agent, corrective) — no re-embedding
    /compare <type> <q> run one question through the current chain AND a
                        variant, then compare latency/tokens/cost
    /cache on|off       toggle the response cache
    /new                start a fresh conversation (new session)
    /help, /exit

No dependencies beyond the pipeline: color is plain ANSI and disables
itself when stdout is not a TTY or NO_COLOR is set.

Usage: ragstone-chat [--provider openai|ollama] [--model NAME]
                     [--data-dir DIR] [--chain TYPE]
"""

import argparse
import os
import sys
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from ragstone.config.settings import get_config
from ragstone.rag.citations import find_supporting_spans, highlight_spans
from ragstone.rag.pipeline import build_pipeline
from ragstone.utils.exceptions import DocumentLoadingError, PipelineError
from ragstone.utils.observability import estimate_cost_usd, track_request

try:  # arrow-key history and line editing; absent on some platforms
    import readline  # noqa: F401
except ImportError:  # pragma: no cover
    pass

CHAIN_TYPES = ("simple", "multi_query", "fusion", "agent", "corrective", "auto")

HELP_TEXT = """commands:
  /sources             show the citations behind the last answer
  /trace               re-print the last answer's trace
  /chain <type>        switch chain: simple, multi_query, fusion, agent, corrective
  /compare <type> <q>  answer <q> with the current chain AND <type>, compare cost
  /cache on|off        toggle the response cache
  /new                 start a fresh conversation
  /help                this text
  /exit                leave (also: quit, q, Ctrl-C)"""


class Style:
    """ANSI styling that degrades to plain text off-TTY or under NO_COLOR."""

    def __init__(self, enabled: Optional[bool] = None):
        if enabled is None:
            enabled = sys.stdout.isatty() and not os.getenv("NO_COLOR")
        self.enabled = enabled

    def _wrap(self, code: str, text: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.enabled else text

    def dim(self, text: str) -> str:
        return self._wrap("2", text)

    def bold(self, text: str) -> str:
        return self._wrap("1", text)

    def cyan(self, text: str) -> str:
        return self._wrap("36", text)

    def yellow(self, text: str) -> str:
        return self._wrap("33", text)


def format_event_line(event: Dict[str, Any]) -> Optional[str]:
    """Human line for a progress event dict, or None for unknown events.

    Mirrors the Streamlit app's event rendering, in plain text.
    """
    kind = event.get("event")
    if kind == "search":
        return f"searching: {event.get('query', '')}"
    if kind == "retrieve":
        return f"retrieving: {event.get('query', '')}"
    if kind == "grade":
        verdict = "relevant" if event.get("relevant") else "not relevant, retrying"
        return f"grading passages: {verdict}"
    if kind == "rewrite":
        return f"rewriting query: {event.get('query', '')}"
    if kind == "route":
        return f"routed to the {event.get('strategy', '?')} path"
    return None


def format_trace(metrics: Any, model: Optional[str] = None) -> str:
    """One-line glass-box trace from a RequestMetrics, matching the web UI.

    Example: first token 320 ms · total 1.42 s · 1183 tokens · ~$0.0004
             · chain=simple · rephrase 210 ms > retrieval 89 ms > generation 1121 ms
    """
    parts: List[str] = []
    if metrics.first_token_ms is not None:
        parts.append(f"first token {metrics.first_token_ms} ms")
    parts.append(f"total {metrics.latency_ms / 1000:.2f} s")
    if metrics.cache_hit:
        parts.append("cache hit")
    elif metrics.tokens:
        parts.append(f"{metrics.tokens} tokens")
        cost = estimate_cost_usd(metrics.input_tokens, metrics.output_tokens, model)
        if cost is not None:
            parts.append(f"~${cost:.4f}")
    parts.append(f"chain={metrics.chain_type}")

    if metrics.stage_ms and not metrics.cache_hit:
        ordered = [s for s in ("rephrase", "retrieval") if s in metrics.stage_ms]
        stages = [f"{s} {metrics.stage_ms[s]} ms" for s in ordered]
        stages.append(f"generation {metrics.generation_ms} ms")
        parts.append(" > ".join(stages))
    return " · ".join(parts)


def parse_command(line: str) -> Tuple[str, str]:
    """Split "/chain corrective" into ("chain", "corrective").

    Plain questions return ("", line) so the caller can treat command
    dispatch and asking uniformly.
    """
    if not line.startswith("/"):
        return "", line
    head, _, rest = line[1:].partition(" ")
    return head.lower(), rest.strip()


class ChatInterface:
    """Interactive REPL over a configured pipeline.

    Building the pipeline (load, embed, chain) happens in the constructor
    unless a ready `pipeline` is injected — which is how the unit tests
    drive the command surface without a network.
    """

    def __init__(
        self,
        pipeline_type: str = "ollama",
        model: str = "llama3",
        data_dir: str = "data",
        chain_type: str = "simple",
        pipeline: Optional[Any] = None,
        style: Optional[Style] = None,
    ):
        self.config = get_config()
        self.style = style or Style()
        self.chain_type = chain_type
        self.model = model
        self.session_id = f"chat_{uuid.uuid4().hex[:8]}"
        self._last_question: Optional[str] = None
        self._last_answer: Optional[str] = None
        self._last_trace: Optional[str] = None

        if pipeline is not None:
            self.pipeline = pipeline
            return

        self.pipeline = build_pipeline(pipeline_type, model)
        print(f"Initialized {pipeline_type} pipeline with {model}")

        print("Loading documents...")
        texts = self.pipeline.load_and_split(data_dir=data_dir)
        if not texts:
            # Fail here instead of entering a chat loop that can never answer.
            raise DocumentLoadingError(
                f"No documents loaded. Add files to the '{data_dir}/' "
                "directory before starting the chat."
            )
        print(f"Loaded {len(texts)} document chunks")

        print("Setting up retriever...")
        self.pipeline.setup_retriever(use_ensemble=True)

        print(f"Creating RAG chain ({chain_type})...")
        self.pipeline.create_rag_chain(chain_type=chain_type)

        print("Ready. Type /help for commands, /exit to leave.\n")

    # -- answering ----------------------------------------------------------

    def ask(self, question: str, use_cache: bool = True) -> None:
        """Stream one answer with events, then print the glass-box trace."""
        style = self.style
        self._last_question = question
        parts: List[str] = []
        printed_any = False
        print(style.cyan("Assistant: "), end="", flush=True)
        try:
            for chunk in self.pipeline.ask_question_stream(
                question, session_id=self.session_id, use_cache=use_cache
            ):
                if isinstance(chunk, dict):
                    line = format_event_line(chunk)
                    if line:
                        # Events land on their own dim line, then the
                        # answer continues on a fresh one.
                        prefix = "\n" if printed_any else ""
                        print(prefix + style.dim(f"  [{line}]"), flush=True)
                        printed_any = False
                else:
                    print(chunk, end="", flush=True)
                    parts.append(chunk)
                    printed_any = True
        except PipelineError as exc:
            print(f"\n{style.yellow(f'Error: {exc}')}")
            return
        print()
        self._last_answer = "".join(parts)

        interpretation = self.pipeline.get_last_interpretation(self.session_id)
        if interpretation and interpretation != question:
            print(style.dim(f"  interpreted as: {interpretation}"))

        metrics = self.pipeline.last_metrics
        if metrics is not None:
            self._last_trace = format_trace(metrics, self.model)
            print(style.dim(f"  {self._last_trace}"))

        names = []
        for source in self.pipeline.get_sources(question):
            if source["source"] not in names:
                names.append(source["source"])
        if names:
            print(style.dim(f"  sources: {', '.join(names)}  (/sources for text)"))

    # -- commands -----------------------------------------------------------

    def cmd_sources(self) -> None:
        if not self._last_question:
            print("Ask a question first.")
            return
        sources = self.pipeline.get_sources(self._last_question)
        if not sources:
            print("No sources recorded for the last answer.")
            return
        for i, source in enumerate(sources, 1):
            print(self.style.bold(f"[{i}] {source['source']}"))
            snippet = source["snippet"]
            spans = (
                find_supporting_spans(self._last_answer, snippet)
                if self._last_answer
                else []
            )
            # Evidence highlighting: the exact source words the answer
            # reuses (yellow on a TTY, plain text otherwise).
            rendered = (
                highlight_spans(snippet, spans, "\033[33m", "\033[0m")
                if (spans and self.style.enabled)
                else snippet
            )
            print(f"    {rendered}\n")

    def cmd_trace(self) -> None:
        print(self._last_trace or "No answer yet.")

    def cmd_chain(self, arg: str) -> None:
        if arg not in CHAIN_TYPES:
            print(f"Unknown chain type {arg!r}. Options: {', '.join(CHAIN_TYPES)}")
            return
        self.pipeline.create_rag_chain(chain_type=arg)
        self.chain_type = arg
        print(f"Switched to the {arg} chain (same corpus, no re-embedding).")

    def cmd_cache(self, arg: str) -> None:
        if arg not in ("on", "off"):
            print("Usage: /cache on|off")
            return
        self.config.cache.enable_response_cache = arg == "on"
        print(f"Response cache {arg}.")

    def cmd_new(self) -> None:
        self.session_id = f"chat_{uuid.uuid4().hex[:8]}"
        self._last_question = None
        self._last_answer = None
        self._last_trace = None
        print("Started a fresh conversation.")

    def cmd_compare(self, arg: str) -> None:
        """Run one question through the current chain and a variant."""
        variant_type, _, question = arg.partition(" ")
        question = question.strip()
        if variant_type not in CHAIN_TYPES or not question:
            print("Usage: /compare <chain_type> <question>")
            return
        style = self.style

        # Current chain first — through the normal path (cache bypassed so
        # both sides do real work), which also fills last_metrics.
        print(style.bold(f"\n--- {self.chain_type} (current) ---"))
        self.ask(question, use_cache=False)
        current_metrics = self.pipeline.last_metrics

        print(style.bold(f"\n--- {variant_type} ---"))
        try:
            variant = self.pipeline.make_chain_variant(variant_type)
        except PipelineError as exc:
            print(style.yellow(f"Could not build {variant_type}: {exc}"))
            return
        variant_session = f"compare_{uuid.uuid4().hex[:8]}"
        printed_any = False
        print(style.cyan("Assistant: "), end="", flush=True)
        with track_request(variant_session, chain_type=variant_type) as metrics:
            try:
                started = time.perf_counter()
                for chunk in variant.stream_question(question, variant_session):
                    if isinstance(chunk, dict):
                        line = format_event_line(chunk)
                        if line:
                            prefix = "\n" if printed_any else ""
                            print(prefix + style.dim(f"  [{line}]"), flush=True)
                            printed_any = False
                    else:
                        if metrics.first_token_ms is None:
                            metrics.first_token_ms = int(
                                (time.perf_counter() - started) * 1000
                            )
                        print(chunk, end="", flush=True)
                        printed_any = True
            except PipelineError as exc:
                print(f"\n{style.yellow(f'Error: {exc}')}")
                return
        print()
        print(style.dim(f"  {format_trace(metrics, self.model)}"))

        if current_metrics is not None and metrics.latency_ms:
            ratio = current_metrics.latency_ms / metrics.latency_ms
            faster, slower = (
                (self.chain_type, variant_type)
                if ratio < 1
                else (variant_type, self.chain_type)
            )
            factor = max(ratio, 1 / ratio) if ratio else 1.0
            print(
                style.bold(
                    f"\nverdict: {faster} was {factor:.1f}x faster than {slower} "
                    "(same corpus, same question)"
                )
            )

    def handle(self, line: str) -> bool:
        """Dispatch one input line; returns False when the loop should end."""
        command, arg = parse_command(line)
        if not command:
            if arg:  # a plain question
                self.ask(arg)
            return True
        if command in ("exit", "quit", "q"):
            return False
        handlers = {
            "help": lambda: print(HELP_TEXT),
            "sources": self.cmd_sources,
            "trace": self.cmd_trace,
            "new": self.cmd_new,
        }
        if command in handlers:
            handlers[command]()
        elif command == "chain":
            self.cmd_chain(arg)
        elif command == "cache":
            self.cmd_cache(arg)
        elif command == "compare":
            self.cmd_compare(arg)
        else:
            print(f"Unknown command /{command} — try /help")
        return True

    def chat(self) -> None:
        """The interactive loop."""
        while True:
            try:
                line = input(self.style.bold("\nYou: ")).strip()
            except (KeyboardInterrupt, EOFError):
                print("\nGoodbye!")
                break
            if not line:
                continue
            if line.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break
            try:
                if not self.handle(line):
                    print("Goodbye!")
                    break
            except Exception as exc:  # keep the loop alive on any failure
                print(self.style.yellow(f"Error: {exc}"))


def get_available_ollama_models() -> List[str]:
    """Names of models the local Ollama server reports, or [] offline."""
    try:
        import requests

        base_url = get_config().api.ollama_base_url
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        if response.status_code == 200:
            models_data = response.json().get("models", [])
            return [model["name"] for model in models_data]
    except Exception:
        pass
    return []


def select_model_interactive() -> Tuple[str, str]:
    """Prompt for provider and model; returns (pipeline_type, model)."""
    print("\nSelect Pipeline Type:")
    print("1. Ollama (Local models)")
    print("2. OpenAI (Cloud models)")

    while True:
        choice = input("\nEnter choice (1-2) [1]: ").strip() or "1"
        if choice in ["1", "2"]:
            break
        print("Invalid choice. Please enter 1 or 2.")

    if choice == "1":
        pipeline_type = "ollama"
        print("\nOllama Models:")

        available_models = get_available_ollama_models()

        if available_models:
            print("Available models:")
            for i, model in enumerate(available_models, 1):
                print(f"{i}. {model}")

            while True:
                model_choice = input(
                    f"\nSelect model (1-{len(available_models)}) or enter custom name: "
                ).strip()

                if model_choice.isdigit() and 1 <= int(model_choice) <= len(
                    available_models
                ):
                    model = available_models[int(model_choice) - 1]
                    break
                elif model_choice:
                    model = model_choice
                    break
                else:
                    print("Please enter a valid choice or model name.")
        else:
            print("Ollama not running or no models found")
            print("Common models: llama3, phi4, deepseek-r1, mixtral")
            model = input("Enter model name [llama3]: ").strip() or "llama3"

    else:
        pipeline_type = "openai"
        print("\nOpenAI Models:")
        openai_models = get_config().llm.openai_models

        for i, model in enumerate(openai_models, 1):
            print(f"{i}. {model}")

        while True:
            model_choice = input(
                f"\nSelect model (1-{len(openai_models)}) or enter custom name: "
            ).strip()

            if model_choice.isdigit() and 1 <= int(model_choice) <= len(openai_models):
                model = openai_models[int(model_choice) - 1]
                break
            elif model_choice:
                model = model_choice
                break
            else:
                print("Please enter a valid choice or model name.")

    return pipeline_type, model


def main():
    """Entry point for the ragstone-chat console script."""
    parser = argparse.ArgumentParser(description="Ragstone terminal chat")
    parser.add_argument("--provider", choices=["openai", "ollama"], default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--chain", choices=CHAIN_TYPES, default="simple")
    args = parser.parse_args()

    print("Ragstone Chat")
    print("=" * 50)

    if args.provider and args.model:
        pipeline_type, model = args.provider, args.model
    else:
        pipeline_type, model = select_model_interactive()

    try:
        chat_interface = ChatInterface(
            pipeline_type=pipeline_type,
            model=model,
            data_dir=args.data_dir,
            chain_type=args.chain,
        )
        chat_interface.chat()
    except Exception as e:
        print(f"Failed to initialize chat interface: {e}")
        print("Make sure:")
        print("- Ollama is running (for Ollama models)")
        print("- OpenAI API key is set (for OpenAI models)")
        print(f"- Documents exist in the '{args.data_dir}/' directory")


if __name__ == "__main__":
    main()
