"""Adapters plugging `llmranker`'s rankers into other RAG frameworks.

Nothing here is imported by `llmranker/__init__.py`: each adapter module
depends on a heavy third-party framework (LangChain, LlamaIndex) that most
users of this package don't have installed, so importing `llmranker` itself
never requires either. Import the specific submodule you need, e.g.
`from llmranker.integrations.langchain import LLMRankerCompressor`.
"""
