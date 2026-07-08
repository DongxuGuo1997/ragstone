"""
Unit tests for the package's lazy import surface (PEP 562).

`import ragstone` must stay cheap: the heavy dependency tree (LangChain,
faiss, torch via the rerank extra) loads only when a name that needs it
is first touched. Each test runs in a subprocess because this suite's
own imports would otherwise pre-populate sys.modules.
"""

import subprocess
import sys


def _run(code: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
    )


class TestLazyImports:
    def test_importing_the_package_loads_no_heavy_modules(self):
        result = _run(
            "import sys; import ragstone; "
            "heavy = [m for m in sys.modules "
            " if m.startswith(('langchain', 'ragstone.rag', 'faiss', 'torch'))]; "
            "print(heavy); sys.exit(1 if heavy else 0)"
        )
        assert result.returncode == 0, f"heavy modules at import: {result.stdout}"

    def test_public_names_resolve_and_cache(self):
        result = _run(
            "import ragstone; "
            "assert ragstone.OpenAIPipeline.__name__ == 'OpenAIPipeline'; "
            "assert 'OpenAIPipeline' in vars(ragstone)  # cached after first touch"
        )
        assert result.returncode == 0, result.stderr

    def test_from_import_still_works(self):
        result = _run(
            "from ragstone import Config, get_config, PipelineError; "
            "assert callable(get_config)"
        )
        assert result.returncode == 0, result.stderr

    def test_unknown_attribute_raises_attribute_error(self):
        result = _run(
            "import ragstone\n"
            "try:\n"
            "    ragstone.does_not_exist\n"
            "except AttributeError:\n"
            "    pass\n"
            "else:\n"
            "    raise SystemExit(1)\n"
        )
        assert result.returncode == 0, result.stderr

    def test_dir_advertises_the_lazy_names(self):
        result = _run(
            "import ragstone; "
            "assert 'OpenAIPipeline' in dir(ragstone); "
            "assert 'create_vector_store_proxy' in dir(ragstone)"
        )
        assert result.returncode == 0, result.stderr
