#!/usr/bin/env python3
"""
Simple Chat Interface for LangChain RAG Pipeline
Usage: python -m langchain_rag.ui.chat_interface (or the rag-chat console script)
"""

from langchain_rag.config.settings import get_config
from langchain_rag.rag.pipeline import OllamaPipeline, OpenAIPipeline
from langchain_rag.utils.exceptions import DocumentLoadingError


class ChatInterface:
    def __init__(self, pipeline_type="ollama", model="llama3"):
        """Initialize the chat interface with a RAG pipeline."""
        self.config = get_config()

        # Create pipeline
        if pipeline_type.lower() == "ollama":
            self.pipeline = OllamaPipeline(model=model)
            print(f"🦙 Initialized Ollama pipeline with {model}")
        else:
            self.pipeline = OpenAIPipeline(model=model)
            print(f"🤖 Initialized OpenAI pipeline with {model}")

        # Load documents
        print("📚 Loading documents...")
        self.texts = self.pipeline.load_and_split(data_dir="data")
        if self.texts:
            print(f"✅ Loaded {len(self.texts)} document chunks")
        else:
            # Fail here instead of entering a chat loop that can never answer.
            raise DocumentLoadingError(
                "No documents loaded. Add files to the 'data/' directory "
                "before starting the chat."
            )

        # Setup retriever
        print("🔍 Setting up retriever...")
        if pipeline_type.lower() == "ollama":
            self.pipeline.set_retriever_ollama(use_ensemble=True)
        else:
            self.pipeline.set_retriever_openai(use_ensemble=True)

        # Create RAG chain
        print("⛓️  Creating RAG chain...")
        self.pipeline.create_rag_chain(chain_type="simple")

        print("🚀 Chat interface ready! Type 'quit' to exit.\n")

    def chat(self):
        """Start the interactive chat loop."""
        session_id = "chat_session"

        while True:
            try:
                # Get user input
                question = input("\n💬 You: ").strip()

                if question.lower() in ["quit", "exit", "q"]:
                    print("👋 Goodbye!")
                    break

                if not question:
                    continue

                # Get response from RAG pipeline
                print("🤔 Thinking...")
                response = self.pipeline.ask_question(question, session_id=session_id)

                if response:
                    print(f"🤖 Assistant: {response}")
                else:
                    print("❌ Sorry, I couldn't generate a response.")

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")


def get_available_ollama_models():
    """Get list of available Ollama models."""
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


def select_model_interactive():
    """Interactive model selection."""
    print("\n🔌 Select Pipeline Type:")
    print("1. Ollama (Local models)")
    print("2. OpenAI (Cloud models)")

    while True:
        choice = input("\nEnter choice (1-2) [1]: ").strip() or "1"
        if choice in ["1", "2"]:
            break
        print("❌ Invalid choice. Please enter 1 or 2.")

    if choice == "1":
        pipeline_type = "ollama"
        print("\n🦙 Ollama Models:")

        # Get available models
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
                    print("❌ Please enter a valid choice or model name.")
        else:
            print("⚠️  Ollama not running or no models found")
            print("Common models: llama3, phi4, deepseek-r1, mixtral")
            model = input("Enter model name [llama3]: ").strip() or "llama3"

    else:
        pipeline_type = "openai"
        print("\n🤖 OpenAI Models:")
        openai_models = ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo", "gpt-4o"]

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
                print("❌ Please enter a valid choice or model name.")

    return pipeline_type, model


def main():
    """Main function to run the chat interface."""
    print("🌟 LangChain RAG Pipeline Chat Interface")
    print("=" * 50)

    # Interactive model selection
    pipeline_type, model = select_model_interactive()

    try:
        # Initialize chat interface
        chat_interface = ChatInterface(pipeline_type=pipeline_type, model=model)

        # Start chatting
        chat_interface.chat()

    except Exception as e:
        print(f"❌ Failed to initialize chat interface: {e}")
        print("Make sure:")
        print("- Ollama is running (for Ollama models)")
        print("- OpenAI API key is set (for OpenAI models)")
        print("- Documents exist in 'data/' directory")


if __name__ == "__main__":
    main()
