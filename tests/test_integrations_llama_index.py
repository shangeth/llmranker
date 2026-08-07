import pytest
from conftest import by_text

pytest.importorskip("llama_index.core")

from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from llmranker.integrations.llama_index import LLMRankerPostprocessor
from llmranker.llm import LLMConfig
from llmranker.rankers.pointwise import PointwiseRanker


def _nodes_with_score(texts):
    return [NodeWithScore(node=TextNode(text=t), score=None) for t in texts]


def test_postprocess_nodes_reorders_and_sets_scores(fake_llm):
    nodes = _nodes_with_score(["alpha", "beta", "gamma"])
    fake_llm.responses = by_text({"alpha": "3", "beta": "9", "gamma": "1"})

    postprocessor = LLMRankerPostprocessor(ranker=PointwiseRanker(LLMConfig(model="gpt-4o-mini")))
    result = postprocessor.postprocess_nodes(nodes, query_str="q")

    assert [n.node.get_content() for n in result] == ["beta", "alpha", "gamma"]
    assert [n.score for n in result] == [9.0, 3.0, 1.0]
    # Same NodeWithScore/TextNode objects, not copies.
    assert result[0] is nodes[1]


def test_postprocess_nodes_preserves_metadata(fake_llm):
    nodes = [
        NodeWithScore(node=TextNode(text="alpha", metadata={"k": "a"}), score=None),
        NodeWithScore(node=TextNode(text="beta", metadata={"k": "b"}), score=None),
    ]
    fake_llm.responses = by_text({"alpha": "1", "beta": "9"})

    postprocessor = LLMRankerPostprocessor(ranker=PointwiseRanker(LLMConfig(model="gpt-4o-mini")))
    result = postprocessor.postprocess_nodes(nodes, query_str="q")

    assert [n.node.metadata for n in result] == [{"k": "b"}, {"k": "a"}]


def test_postprocess_nodes_without_a_query_is_a_passthrough(fake_llm):
    nodes = _nodes_with_score(["alpha", "beta"])

    postprocessor = LLMRankerPostprocessor(ranker=PointwiseRanker(LLMConfig(model="gpt-4o-mini")))
    result = postprocessor.postprocess_nodes(nodes)

    assert result == nodes


def test_postprocess_nodes_empty_list(fake_llm):
    postprocessor = LLMRankerPostprocessor(ranker=PointwiseRanker(LLMConfig(model="gpt-4o-mini")))
    assert postprocessor.postprocess_nodes([], query_str="q") == []


def test_postprocess_nodes_via_query_bundle(fake_llm):
    nodes = _nodes_with_score(["alpha", "beta"])
    fake_llm.responses = by_text({"alpha": "1", "beta": "9"})

    postprocessor = LLMRankerPostprocessor(ranker=PointwiseRanker(LLMConfig(model="gpt-4o-mini")))
    result = postprocessor.postprocess_nodes(nodes, query_bundle=QueryBundle(query_str="q"))

    assert [n.node.get_content() for n in result] == ["beta", "alpha"]
