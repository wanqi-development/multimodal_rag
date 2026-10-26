"""统一证据账本：语料块与网络结果共用一套编号和来源形状。"""

from __future__ import annotations

from rag.contracts import Citation, EvidenceLedger, format_sources


def test_the_ledger_deduplicates_corpus_and_web_sources_independently() -> None:
    ledger = EvidenceLedger()

    corpus = ledger.issue(
        chunk_id="d1:0001",
        source_uri="a.pdf",
        snippet="正文",
        heading_path="第一章 -> 1.1",
    )
    corpus_again = ledger.issue(
        chunk_id="d1:0001",
        source_uri="a.pdf",
        snippet="正文",
        heading_path="第一章 -> 1.1",
    )
    web = ledger.issue_web(
        url="https://example.test/doc",
        title="官方文档",
        snippet="网页正文",
    )
    web_again = ledger.issue_web(
        url="https://example.test/doc",
        title="官方文档（新标题）",
        snippet="网页正文",
    )

    assert (corpus, corpus_again, web, web_again) == (1, 1, 2, 2)
    assert ledger.items[1].source_type == "corpus"
    assert ledger.items[2].source_type == "web"


def test_only_cited_evidence_is_returned() -> None:
    ledger = EvidenceLedger()
    ledger.issue(chunk_id="d1:0001", source_uri="a.pdf")
    ledger.issue_web(url="https://example.test/doc", title="官网")

    cited = ledger.cited_in("语料结论 [1]，网络补充 [2]。")

    assert [item.index for item in cited] == [1, 2]


def test_web_sources_are_marked_and_keep_their_url() -> None:
    ledger = EvidenceLedger()
    ledger.issue_web(
        url="https://example.test/doc",
        title="官方文档",
        snippet="网页正文",
    )

    rendered = format_sources(ledger.records())

    assert "来自网络 · 官方文档 · https://example.test/doc" in rendered


def test_payload_round_trip_keeps_every_source_field() -> None:
    ledger = EvidenceLedger()
    ledger.issue(
        chunk_id="d1:0001",
        source_uri="a.pdf",
        title="A 文档",
        snippet="正文",
        heading_path="第一章 -> 1.1",
    )
    ledger.issue_web(url="https://example.test/doc", title="官网", snippet="网页")

    restored = EvidenceLedger.from_payload(ledger.to_payload())

    assert restored.items[1] == Citation(
        index=1,
        source_type="corpus",
        chunk_id="d1:0001",
        source_uri="a.pdf",
        title="A 文档",
        snippet="正文",
        heading_path="第一章 -> 1.1",
    )
    assert restored.items[2].url == "https://example.test/doc"
    assert restored.items[2].source_type == "web"
