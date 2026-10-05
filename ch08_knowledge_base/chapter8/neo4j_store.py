"""
[8장 3절] 에이전트 시험 공부 하기 - Neo4j 연결과 그래프 저장/조회

LiteratureExtraction을 Neo4j의 노드·관계로 저장하고(MERGE), 시험 풀이에 쓸
인물 관계·해석 노드를 Cypher로 조회합니다. 2절의 save_to_graph와 같은 그래프 구조를 쓰되,
리스트가 비어도 나머지가 저장되도록 테마·인물·관계 등을 쿼리 단위로 나누어 실행합니다.

ingest가 끝나면 Neo4j에 다음 구조가 만들어집니다(노드와 관계의 수는 추출 결과에 따라 달라집니다).
- 노드: Author, Book, Theme, Character, Event, Interpretation, Quote
- 관계: WROTE, HAS_THEME, HAS_CHARACTER, HAS_EVENT, PARTICIPATED_IN, HAS_INTERPRETATION, HAS_QUOTE, RELATION
Neo4j Browser에서 MATCH (n) RETURN distinct labels(n), count(*) 로 확인할 수 있습니다.

필요한 환경 변수: NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD (선택: NEO4J_DATABASE, 기본값 neo4j)
"""

from __future__ import annotations

import os
from typing import Any, Mapping, Sequence

from langchain_neo4j import Neo4jGraph

from chapter8.schemas import LiteratureExtraction


def build_neo4j_graph() -> Neo4jGraph:
    url = os.environ.get("NEO4J_URI") or os.environ.get("NEO4J_URL")
    if not url:
        raise RuntimeError("NEO4J_URI(또는 NEO4J_URL) 환경 변수가 필요합니다.")
    username = os.environ["NEO4J_USERNAME"]
    password = os.environ["NEO4J_PASSWORD"]
    database = os.environ.get("NEO4J_DATABASE") or "neo4j"
    return Neo4jGraph(
        url=url,
        username=username,
        password=password,
        database=database,
        refresh_schema=False,  # 접속할 때 APOC으로 스키마를 읽지 않으므로 APOC 플러그인이 없어도 됩니다.
    )


def save_extraction_to_graph(extracted: LiteratureExtraction, graph: Neo4jGraph) -> None:
    """추출 결과를 여러 트랜잭션으로 나누어 저장(빈 리스트는 건너뜀)."""
    title = extracted.work_title
    author = extracted.author
    params_base: dict[str, Any] = {
        "author": author,
        "title": title,
        "summary": extracted.summary,
    }

    # 작가-작품 노드와 요약을 먼저 저장하고, 나머지는 리스트가 있을 때만 따로 실행합니다.
    graph.query(
        """
        MERGE (a:Author {name: $author})
        MERGE (b:Book {title: $title})
        MERGE (a)-[:WROTE]->(b)
        SET b.summary = $summary
        """,
        params=params_base,
    )

    if extracted.themes:
        graph.query(
            """
            MATCH (b:Book {title: $title})
            UNWIND $themes AS theme_name
            MERGE (t:Theme {name: theme_name})
            MERGE (b)-[:HAS_THEME]->(t)
            """,
            params={"title": title, "themes": extracted.themes},
        )

    # 같은 이름이 다른 작품에 나올 수 있어 Character는 work_title을 MERGE 키로 함께 씁니다.
    if extracted.characters:
        ch_rows = [c.model_dump() for c in extracted.characters]
        graph.query(
            """
            MATCH (b:Book {title: $title})
            UNWIND $characters AS ch
            MERGE (c:Character {name: ch.name, work_title: $title})
            SET c.role = ch.role,
                c.traits = ch.traits
            MERGE (b)-[:HAS_CHARACTER]->(c)
            """,
            params={"title": title, "characters": ch_rows},
        )

    # 관계 엣지의 속성명은 Cypher에서 혼동을 줄이도록 rtype(문자열)을 씁니다.
    if extracted.relations:
        rel_rows = [r.model_dump() for r in extracted.relations]
        graph.query(
            """
            UNWIND $relations AS r
            MATCH (c1:Character {name: r.source, work_title: $title})
            MATCH (c2:Character {name: r.target, work_title: $title})
            MERGE (c1)-[rel:RELATION {rtype: r.relation}]->(c2)
            SET rel.evidence = r.evidence_quote
            """,
            params={"title": title, "relations": rel_rows},
        )

    if extracted.events:
        ev_rows = [e.model_dump() for e in extracted.events]
        graph.query(
            """
            MATCH (b:Book {title: $title})
            UNWIND $events AS e
            MERGE (ev:Event {id: e.event_id, work_title: $title})
            SET ev.description = e.description,
                ev.impact = e.impact
            MERGE (b)-[:HAS_EVENT]->(ev)
            WITH ev, e
            UNWIND e.participants AS pname
            MATCH (c:Character {name: pname, work_title: $title})
            MERGE (c)-[:PARTICIPATED_IN]->(ev)
            """,
            params={"title": title, "events": ev_rows},
        )

    if extracted.interpretations:
        int_rows = [i.model_dump() for i in extracted.interpretations]
        graph.query(
            """
            MATCH (b:Book {title: $title})
            UNWIND $interpretations AS i
            MERGE (it:Interpretation {work_title: $title, subject: i.subject})
            SET it.meaning = i.meaning,
                it.evidence = i.evidence_quote
            MERGE (b)-[:HAS_INTERPRETATION]->(it)
            """,
            params={"title": title, "interpretations": int_rows},
        )

    if extracted.key_quotes:
        graph.query(
            """
            MATCH (b:Book {title: $title})
            UNWIND $quotes AS quote_text
            MERGE (q:Quote {text: quote_text, work_title: $title})
            MERGE (b)-[:HAS_QUOTE]->(q)
            """,
            params={"title": title, "quotes": extracted.key_quotes},
        )


# 아래 조회 함수들은 2절 GraphRAG의 쿼리(인물 관계, 해석, 작가와 주제로 작품 찾기)와 같습니다.
def fetch_character_relations(graph: Neo4jGraph, title: str) -> Sequence[Mapping[str, Any]]:
    q = """
    MATCH (b:Book {title: $title})-[:HAS_CHARACTER]->(c1:Character)
          -[r:RELATION]->(c2:Character)
    RETURN c1.name AS source, r.rtype AS relation, c2.name AS target, r.evidence AS evidence
    """
    return graph.query(q, params={"title": title})


def fetch_interpretations(graph: Neo4jGraph, title: str) -> Sequence[Mapping[str, Any]]:
    q = """
    MATCH (b:Book {title: $title})-[:HAS_INTERPRETATION]->(i:Interpretation)
    RETURN i.subject AS subject, i.meaning AS meaning, i.evidence AS evidence
    """
    return graph.query(q, params={"title": title})


def fetch_themes_for_author_book(
    graph: Neo4jGraph, author: str, theme: str
) -> Sequence[Mapping[str, Any]]:
    q = """
    MATCH (a:Author {name: $author})-[:WROTE]->(b:Book)-[:HAS_THEME]->(t:Theme {name: $theme})
    RETURN b.title AS title
    """
    return graph.query(q, params={"author": author, "theme": theme})
