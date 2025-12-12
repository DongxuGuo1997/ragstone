"""
This module provides classes for loading documents from local and remote sources.
It defines an abstract base class `Loader` and concrete implementations
`LocalLoader` for file system operations and `RemoteLoader` for web-based content.
"""
import os
import logging
from pathlib import Path
from abc import ABC, abstractmethod
from typing import List, Optional, Union, Any, TYPE_CHECKING, Dict
import tempfile # For handling uploaded files

# Lazy imports - only import when needed
if TYPE_CHECKING:
    from pypdf import PdfReader
    from langchain.docstore.document import Document
    from streamlit.runtime.uploaded_file_manager import UploadedFile

# Optional Streamlit import with fallback
try:
    from streamlit.runtime.uploaded_file_manager import UploadedFile
    STREAMLIT_AVAILABLE = True
except ImportError:
    UploadedFile = None
    STREAMLIT_AVAILABLE = False

logger = logging.getLogger(__name__)

# Import cache for heavy document loaders
_loader_cache = {}
_document_cache = {}  # Cache for loaded documents

def _get_cached_loader(loader_type: str):
    """Get cached loader import or import and cache it."""
    if loader_type not in _loader_cache:
        try:
            if loader_type == "text":
                from langchain_community.document_loaders import TextLoader
                _loader_cache[loader_type] = TextLoader
            elif loader_type == "csv":
                from langchain_community.document_loaders import CSVLoader
                _loader_cache[loader_type] = CSVLoader
            elif loader_type == "pdf":
                from langchain_community.document_loaders import PyPDFLoader
                _loader_cache[loader_type] = PyPDFLoader
            elif loader_type == "word":
                from langchain_community.document_loaders import UnstructuredWordDocumentLoader
                _loader_cache[loader_type] = UnstructuredWordDocumentLoader
            elif loader_type == "markdown":
                from langchain_community.document_loaders import UnstructuredMarkdownLoader
                _loader_cache[loader_type] = UnstructuredMarkdownLoader
            elif loader_type == "web":
                from langchain_community.document_loaders import WebBaseLoader
                _loader_cache[loader_type] = WebBaseLoader
            elif loader_type == "wiki":
                from langchain_community.document_loaders import WikipediaLoader
                _loader_cache[loader_type] = WikipediaLoader
            elif loader_type == "html":
                from langchain_community.document_loaders import AsyncHtmlLoader
                _loader_cache[loader_type] = AsyncHtmlLoader
            elif loader_type == "html_transformer":
                from langchain_community.document_transformers import Html2TextTransformer
                _loader_cache[loader_type] = Html2TextTransformer
            else:
                raise ValueError(f"Unknown loader type: {loader_type}")
        except ImportError as e:
            logger.error(f"Failed to import {loader_type} loader: {e}")
            raise
    
    return _loader_cache[loader_type]

def _lazy_import_document():
    """Lazy import for LangChain Document class."""
    if "document" not in _loader_cache:
        from langchain.docstore.document import Document
        _loader_cache["document"] = Document
    return _loader_cache["document"]


class Loader(ABC):
    """Abstract base class for document loaders."""

    def __init__(self, name: str):
        self._name = name
        self._documents: List = []
        logger.info(f"{self._name} loader initialized.")

    @abstractmethod
    def load(self, *args, **kwargs) -> None:
        """
        Load documents from a source.
        Implementations should populate `self._documents`.
        """
        pass

    def get_documents(self) -> List:
        """
        Returns the list of loaded documents.

        Returns:
            List: A list of Langchain Document objects.
        """
        return self._documents

    def clear_documents(self) -> None:
        """Clears the list of loaded documents."""
        self._documents = []
        logger.info(f"Documents cleared for {self._name} loader.")


class LocalLoader(Loader):
    """Loads documents from the local file system and uploaded files with lazy loading."""

    def __init__(self, name: str = "Local"):
        super().__init__(name)

    def _load_files_from_dir(
        self,
        data_dir: str,
        glob_pattern: str,
        loader_type: str,
        loader_kwargs: Optional[dict] = None,
    ) -> List:
        """Helper to load files of a specific type from a directory with lazy loading."""
        docs: List = []
        if not Path(data_dir).is_dir():
            logger.warning(f"Data directory '{data_dir}' not found or is not a directory. Skipping {glob_pattern} files.")
            return docs
        
        if loader_kwargs is None:
            loader_kwargs = {}

        # Lazy import the loader class
        loader_cls = _get_cached_loader(loader_type)

        paths = Path(data_dir).glob(glob_pattern)
        for path in paths:
            if path.is_file():
                try:
                    logger.info(f"Loading {path}")
                    loader_instance = loader_cls(str(path), **loader_kwargs)
                    docs.extend(loader_instance.load())
                except Exception as e:
                    logger.error(f"Failed to load {path}: {e}", exc_info=True)
        return docs

    def _load_text_files(self, data_dir: str) -> List:
        return self._load_files_from_dir(data_dir, "**/*.txt", "text")

    def _load_csv_files(self, data_dir: str) -> List:
        """Load CSV files with lazy loading."""
        docs: List = []
        if not Path(data_dir).is_dir():
            logger.warning(f"Data directory '{data_dir}' not found. Skipping CSV files.")
            return docs
        
        # Lazy import CSVLoader
        CSVLoader = _get_cached_loader("csv")
        
        paths = Path(data_dir).glob("**/*.csv") # Search in subdirectories too
        for path in paths:
            if path.is_file():
                try:
                    logger.info(f"Loading {path}")
                    loader = CSVLoader(file_path=str(path))
                    docs.extend(loader.load())
                except Exception as e:
                    logger.error(f"Failed to load {path}: {e}", exc_info=True)
        return docs

    def _load_md_files(self, data_dir: str) -> List:
        return self._load_files_from_dir(data_dir, "**/*.md", "markdown")

    def _load_pdf_files(self, data_dir: str) -> List:
        return self._load_files_from_dir(data_dir, "**/*.pdf", "pdf")

    def _load_word_files(self, data_dir: str) -> List:
        return self._load_files_from_dir(data_dir, "**/*.docx", "word")

    def _process_uploaded_files(self, uploaded_files: Optional[List[Any]]) -> List:
        """
        Process Streamlit uploaded files and return Document objects.
        
        This method is specifically designed for Streamlit file upload objects.
        During testing/CLI usage, it safely returns empty list without errors.
        """
        if not uploaded_files:
            logger.debug("No uploaded files to process")
            return []

        docs = []
        
        # Check if we have actual Streamlit uploaded file objects
        # Streamlit uploaded files have .name and .getvalue() methods
        valid_uploaded_files = []
        for uploaded_file in uploaded_files:
            if hasattr(uploaded_file, 'name') and hasattr(uploaded_file, 'getvalue'):
                valid_uploaded_files.append(uploaded_file)
            else:
                # This is likely a test case or CLI usage - just skip gracefully
                logger.debug(f"Skipping non-Streamlit uploaded file object: {type(uploaded_file)}")
        
        if not valid_uploaded_files:
            logger.debug("No valid Streamlit uploaded files found - likely testing/CLI usage")
            return []

        # Lazy import Document class
        Document = _lazy_import_document()
        logger.info(f"Processing {len(valid_uploaded_files)} Streamlit uploaded files")

        for uploaded_file in valid_uploaded_files:
            try:
                fname = uploaded_file.name
                title = os.path.splitext(fname)[0]
                logger.info(f"Processing uploaded file: {fname}")

                # Use a temporary file to work with Langchain loaders that expect file paths
                with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{fname}") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                
                file_loader = None
                
                try:
                    if fname.lower().endswith(".pdf"):
                        PyPDFLoader = _get_cached_loader("pdf")
                        file_loader = PyPDFLoader(tmp_file_path)
                    elif fname.lower().endswith(".txt"):
                        TextLoader = _get_cached_loader("text")
                        file_loader = TextLoader(tmp_file_path)
                    elif fname.lower().endswith(".md"):
                        UnstructuredMarkdownLoader = _get_cached_loader("markdown")
                        file_loader = UnstructuredMarkdownLoader(tmp_file_path)
                    elif fname.lower().endswith(".csv"):
                        CSVLoader = _get_cached_loader("csv")
                        file_loader = CSVLoader(tmp_file_path)
                    else:
                        # Fallback: try to read as text
                        logger.warning(f"Unsupported file type: {fname}. Attempting to read as text.")
                        try:
                            content = uploaded_file.getvalue().decode("utf-8")
                            docs.append(Document(page_content=content, metadata={"source": fname, "title": title}))
                        except Exception as decode_error:
                            logger.warning(f"Could not read {fname} as text: {decode_error}")
                        continue

                    if file_loader:
                        loaded_docs = file_loader.load()
                        # Add metadata
                        for doc in loaded_docs:
                            doc.metadata["source"] = fname
                            doc.metadata["title"] = title
                        docs.extend(loaded_docs)
                        logger.debug(f"Loaded {len(loaded_docs)} documents from {fname}")
                
                finally:
                    # Always clean up temp file
                    if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
                        os.remove(tmp_file_path)

            except Exception as e:
                logger.warning(f"Failed to process uploaded file {fname}: {e}")
                logger.debug(f"Uploaded file processing error details:", exc_info=True)
                
                # Ensure cleanup on error
                if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
                    os.remove(tmp_file_path)
                
                # Continue processing other files
                continue
        
        if docs:
            logger.info(f"Successfully processed {len(docs)} documents from {len(valid_uploaded_files)} uploaded files")
        else:
            logger.debug("No documents loaded from uploaded files")
        
        return docs

    def load(self, data_dir: str = "data", uploaded_files: Optional[List[Any]] = None) -> None:
        """
        Loads documents from a specified directory and from uploaded files.

        Args:
            data_dir (str): The directory to load local files from. Defaults to "data".
            uploaded_files (Optional[List]): A list of files uploaded by the user.
        """
        loaded_docs: List = []
        logger.info(f"Starting local loading from data_dir: '{data_dir}' and uploaded files.")

        if data_dir: # Only load from dir if data_dir is provided
            logger.info(f"Loading local files from directory: {data_dir}")
            try:
                # Load different file types with lazy loading
                loaded_docs.extend(self._load_text_files(data_dir))
                loaded_docs.extend(self._load_csv_files(data_dir))
                loaded_docs.extend(self._load_md_files(data_dir))
                loaded_docs.extend(self._load_pdf_files(data_dir))
                loaded_docs.extend(self._load_word_files(data_dir))
                logger.info(f"Local loading completed. Loaded {len(loaded_docs)} documents from {data_dir}")
            except Exception as e:
                logger.error(f"Error during local file loading from {data_dir}: {e}", exc_info=True)

        # Process uploaded files
        if uploaded_files:
            logger.info(f"Processing {len(uploaded_files)} uploaded files")
            try:
                uploaded_docs = self._process_uploaded_files(uploaded_files)
                loaded_docs.extend(uploaded_docs)
                logger.info(f"Uploaded file processing completed. Loaded {len(uploaded_docs)} documents")
            except Exception as e:
                logger.error(f"Error during uploaded file processing: {e}", exc_info=True)

        # Store all loaded documents
        self._documents = loaded_docs
        total_docs = len(loaded_docs)
        logger.info(f"Local loading complete. Total documents loaded: {total_docs}")


class OptimizedLocalLoader(LocalLoader):
    """
    🚀 OPTIMIZED document loader with parallel processing, smart caching, and efficiency improvements.
    
    Key optimizations:
    - Single directory traversal instead of 5 separate ones
    - Parallel file processing using ThreadPoolExecutor
    - Smart caching with file modification time tracking
    - Progress tracking and statistics
    - Batch processing for similar file types
    - Better error handling and recovery
    """

    def __init__(self, name: str = "OptimizedLocal", max_workers: int = 4, enable_cache: bool = True):
        super().__init__(name)
        self.max_workers = max_workers
        self.enable_cache = enable_cache
        self._file_stats = {}
        self._cache_hits = 0
        self._cache_misses = 0
        
        # Supported file extensions mapping
        self._file_type_mapping = {
            '.txt': 'text',
            '.csv': 'csv', 
            '.md': 'markdown',
            '.pdf': 'pdf',
            '.docx': 'word'
        }
        
        logger.info(f"OptimizedLocalLoader initialized: max_workers={max_workers}, cache_enabled={enable_cache}")

    def _get_file_hash(self, file_path: Path) -> str:
        """Generate a cache key based on file path and modification time."""
        try:
            stat = file_path.stat()
            return f"{file_path}:{stat.st_mtime}:{stat.st_size}"
        except OSError:
            return f"{file_path}:unknown"

    def _is_file_cached(self, file_path: Path) -> bool:
        """Check if file is in cache and hasn't been modified."""
        if not self.enable_cache:
            return False
            
        file_hash = self._get_file_hash(file_path)
        return file_hash in _document_cache

    def _get_cached_documents(self, file_path: Path) -> Optional[List]:
        """Retrieve documents from cache if available."""
        if not self.enable_cache:
            return None
            
        file_hash = self._get_file_hash(file_path)
        if file_hash in _document_cache:
            self._cache_hits += 1
            logger.debug(f"Cache HIT: {file_path.name}")
            return _document_cache[file_hash]
        
        self._cache_misses += 1
        return None

    def _cache_documents(self, file_path: Path, documents: List) -> None:
        """Cache documents for future use."""
        if not self.enable_cache:
            return
            
        file_hash = self._get_file_hash(file_path)
        _document_cache[file_hash] = documents
        logger.debug(f"Cached {len(documents)} documents for {file_path.name}")

    def _scan_directory(self, data_dir: str) -> Dict[str, List[Path]]:
        """
        Single directory traversal to categorize all files by type.
        Much more efficient than multiple glob operations.
        """
        files_by_type = {file_type: [] for file_type in self._file_type_mapping.values()}
        files_by_type['unknown'] = []
        
        if not Path(data_dir).is_dir():
            logger.warning(f"Data directory '{data_dir}' not found or is not a directory.")
            return files_by_type
        
        logger.info(f"🔍 Scanning directory: {data_dir}")
        total_files = 0
        
        # Single recursive directory walk
        for file_path in Path(data_dir).rglob('*'):
            if file_path.is_file():
                total_files += 1
                file_ext = file_path.suffix.lower()
                
                if file_ext in self._file_type_mapping:
                    file_type = self._file_type_mapping[file_ext]
                    files_by_type[file_type].append(file_path)
                else:
                    files_by_type['unknown'].append(file_path)
        
        # Log scan results
        logger.info(f"📊 Directory scan complete: {total_files} total files found")
        for file_type, paths in files_by_type.items():
            if paths:
                logger.info(f"   {file_type}: {len(paths)} files")
        
        return files_by_type

    def _load_single_file(self, file_path: Path, file_type: str) -> List:
        """Load a single file with caching and error handling."""
        try:
            # Check cache first
            cached_docs = self._get_cached_documents(file_path)
            if cached_docs is not None:
                return cached_docs
            
            # Load the file
            logger.debug(f"Loading {file_path}")
            
            # Get the appropriate loader
            loader_cls = _get_cached_loader(file_type)
            loader_kwargs = {}
            
            # Special handling for CSV files
            if file_type == 'csv':
                loader_kwargs = {}  # Add CSV-specific kwargs if needed
            
            loader_instance = loader_cls(str(file_path), **loader_kwargs)
            docs = loader_instance.load()
            
            # Enhance metadata
            for doc in docs:
                if hasattr(doc, 'metadata'):
                    doc.metadata.update({
                        'file_name': file_path.name,
                        'file_type': file_type,
                        'file_size': file_path.stat().st_size,
                        'last_modified': file_path.stat().st_mtime
                    })
            
            # Cache the documents
            self._cache_documents(file_path, docs)
            
            return docs
            
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")
            return []

    def _load_files_parallel(self, files_by_type: Dict[str, List[Path]]) -> List:
        """Load files in parallel using ThreadPoolExecutor."""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import time
        
        all_documents = []
        total_files = sum(len(paths) for paths in files_by_type.values() if paths)
        
        if total_files == 0:
            logger.info("No supported files found to load")
            return all_documents
        
        logger.info(f"🚀 Starting parallel loading of {total_files} files with {self.max_workers} workers")
        start_time = time.time()
        
        # Prepare tasks for all files
        tasks = []
        for file_type, file_paths in files_by_type.items():
            if file_type != 'unknown' and file_paths:  # Skip unknown file types
                for file_path in file_paths:
                    tasks.append((file_path, file_type))
        
        loaded_count = 0
        failed_count = 0
        
        # Execute tasks in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(self._load_single_file, file_path, file_type): (file_path, file_type)
                for file_path, file_type in tasks
            }
            
            # Process completed tasks
            for future in as_completed(future_to_file):
                file_path, file_type = future_to_file[future]
                try:
                    docs = future.result()
                    if docs:
                        all_documents.extend(docs)
                        loaded_count += 1
                        logger.debug(f"✅ Loaded {len(docs)} docs from {file_path.name}")
                    else:
                        failed_count += 1
                        
                    # Progress update every 10 files
                    if (loaded_count + failed_count) % 10 == 0:
                        progress = (loaded_count + failed_count) / len(tasks) * 100
                        logger.info(f"📈 Progress: {progress:.1f}% ({loaded_count + failed_count}/{len(tasks)} files)")
                        
                except Exception as e:
                    logger.error(f"❌ Failed to load {file_path}: {e}")
                    failed_count += 1
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Performance summary
        logger.info(f"✅ Parallel loading complete!")
        logger.info(f"   📄 Total documents: {len(all_documents)}")
        logger.info(f"   ✅ Successful files: {loaded_count}")
        logger.info(f"   ❌ Failed files: {failed_count}")
        logger.info(f"   ⏱️  Duration: {duration:.2f}s")
        logger.info(f"   🚀 Speed: {len(all_documents)/duration:.1f} docs/sec")
        if self.enable_cache:
            logger.info(f"   💾 Cache: {self._cache_hits} hits, {self._cache_misses} misses")
        
        return all_documents

    def load(self, data_dir: str = "data", uploaded_files: Optional[List[Any]] = None) -> None:
        """
        Optimized loading with parallel processing and smart caching.
        
        Args:
            data_dir (str): The directory to load local files from.
            uploaded_files (Optional[List]): A list of files uploaded by the user.
        """
        loaded_docs: List = []
        logger.info(f"🚀 Starting OPTIMIZED loading from data_dir: '{data_dir}'")
        
        if data_dir:
            try:
                # Single directory scan instead of multiple traversals
                files_by_type = self._scan_directory(data_dir)
                
                # Parallel loading
                local_docs = self._load_files_parallel(files_by_type)
                loaded_docs.extend(local_docs)
                
            except Exception as e:
                logger.error(f"Error during optimized local file loading: {e}", exc_info=True)
        
        # Process uploaded files (keeping original logic for now)
        if uploaded_files:
            logger.info(f"Processing {len(uploaded_files)} uploaded files")
            try:
                uploaded_docs = self._process_uploaded_files(uploaded_files)
                loaded_docs.extend(uploaded_docs)
                logger.info(f"Uploaded file processing completed. Loaded {len(uploaded_docs)} documents")
            except Exception as e:
                logger.error(f"Error during uploaded file processing: {e}", exc_info=True)
        
        # Store all loaded documents
        self._documents = loaded_docs
        total_docs = len(loaded_docs)
        logger.info(f"✨ OPTIMIZED loading complete. Total documents loaded: {total_docs}")

    def clear_cache(self) -> None:
        """Clear the document cache."""
        global _document_cache
        cache_size = len(_document_cache)
        _document_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        logger.info(f"🗑️  Cache cleared: {cache_size} entries removed")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        return {
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'hit_rate': self._cache_hits / (self._cache_hits + self._cache_misses) if (self._cache_hits + self._cache_misses) > 0 else 0,
            'cache_size': len(_document_cache),
            'cache_enabled': self.enable_cache
        }


class RemoteLoader(Loader):
    """Loads documents from remote sources with lazy loading."""

    def __init__(self, name: str = "Remote"):
        super().__init__(name)

    def _load_web_pages_basic(self, page_urls: List[str]) -> List:
        """Load web pages using basic WebBaseLoader with lazy import."""
        docs: List = []
        if not page_urls:
            return docs
        
        # Lazy import WebBaseLoader
        WebBaseLoader = _get_cached_loader("web")
        
        try:
            loader = WebBaseLoader(page_urls)
            docs = loader.load()
            logger.info(f"Loaded {len(docs)} documents from {len(page_urls)} web pages using WebBaseLoader")
        except Exception as e:
            logger.error(f"Failed to load web pages with WebBaseLoader: {e}", exc_info=True)
        
        return docs

    def _scrape_web_pages_html(self, page_urls: List[str]) -> List:
        """Scrape web pages using AsyncHtmlLoader and Html2TextTransformer with lazy imports."""
        docs: List = []
        if not page_urls:
            return docs
        
        try:
            # Lazy import loaders
            AsyncHtmlLoader = _get_cached_loader("html")
            Html2TextTransformer = _get_cached_loader("html_transformer")
            
            html_loader = AsyncHtmlLoader(page_urls)
            html_docs = html_loader.load()
            
            html2text = Html2TextTransformer()
            docs = html2text.transform_documents(html_docs)
            logger.info(f"Scraped {len(docs)} documents from {len(page_urls)} web pages using AsyncHtmlLoader")
        except Exception as e:
            logger.error(f"Failed to scrape web pages with AsyncHtmlLoader: {e}", exc_info=True)
        
        return docs

    def _load_wikipedia_docs(self, query: str, max_docs: int = 2) -> List:
        """Load documents from Wikipedia with lazy import."""
        docs: List = []
        if not query:
            return docs
        
        try:
            # Lazy import WikipediaLoader
            WikipediaLoader = _get_cached_loader("wiki")
            
            loader = WikipediaLoader(query=query, load_max_docs=max_docs)
            docs = loader.load()
            logger.info(f"Loaded {len(docs)} documents from Wikipedia for query: '{query}'")
        except Exception as e:
            logger.error(f"Failed to load Wikipedia documents for query '{query}': {e}", exc_info=True)
        
        return docs

    def load(
        self,
        page_urls: Optional[List[str]] = None,
        wiki_query: Optional[str] = None,
        scrape_pages: bool = True, # Changed from web_scrap for clarity
        max_wiki_docs: int = 2,
    ) -> None:
        """
        Load documents from remote sources with lazy loading.

        Args:
            page_urls (Optional[List[str]]): URLs of web pages to load.
            wiki_query (Optional[str]): Wikipedia search query.
            scrape_pages (bool): Whether to use advanced scraping (AsyncHtmlLoader) or basic loading.
            max_wiki_docs (int): Maximum number of Wikipedia documents to load.
        """
        loaded_docs: List = []
        logger.info(f"Starting remote loading. URLs: {'yes' if page_urls else 'no'}, Wiki query: {'yes' if wiki_query else 'no'}")

        # Load web pages
        if page_urls:
            logger.info(f"Loading {len(page_urls)} web pages. Scraping mode: {scrape_pages}")
            try:
                if scrape_pages:
                    web_docs = self._scrape_web_pages_html(page_urls)
                else:
                    web_docs = self._load_web_pages_basic(page_urls)
                
                loaded_docs.extend(web_docs)
                logger.info(f"Web page loading completed. Loaded {len(web_docs)} documents")
            except Exception as e:
                logger.error(f"Error during web page loading: {e}", exc_info=True)

        # Load Wikipedia documents
        if wiki_query:
            logger.info(f"Loading Wikipedia documents for query: '{wiki_query}', max docs: {max_wiki_docs}")
            try:
                wiki_docs = self._load_wikipedia_docs(wiki_query, max_wiki_docs)
                loaded_docs.extend(wiki_docs)
                logger.info(f"Wikipedia loading completed. Loaded {len(wiki_docs)} documents")
            except Exception as e:
                logger.error(f"Error during Wikipedia loading: {e}", exc_info=True)

        # Store all loaded documents
        self._documents = loaded_docs
        total_docs = len(loaded_docs)
        logger.info(f"Remote loading complete. Total documents loaded: {total_docs}")


def main():
    """Main function for testing loader functionality."""
    pass


if __name__ == "__main__":
    main()
