"""
[8장 2절] [GraphRAG] 정보를 저장하고 불러오기 - 6. GraphRAG 활용하기

이 절에서 배우는 것:
- "A와 B의 관계는?", "어떤 사건이 갈등을 만들었나?", "이 행동의 의미는?" 같은 관계 기반 질문은 Neo4j 그래프 검색이 더 적합합니다.
- 02b의 save_to_graph로 넣은 구조(RELATION, Event, Interpretation)를 Cypher MATCH로 따라가 '관계 그 자체'를 바로 꺼냅니다.
- 자주 쓰는 쿼리는 인물과 인물의 관계, 사건과 참여 인물, 행동·생각·상징의 해석 세 가지입니다.
  에이전트에 붙일 때는 이런 쿼리를 도구(tool)로 감싸 두고 필요한 순간에만 호출하게 합니다.

실행 방법:
    uv run python ch08_knowledge_base/02d_graph_rag.py
필요한 환경 변수: 없음 (LLM을 호출하지 않습니다)
준비물: Neo4j. 실행 전에 password="YOUR_PASSWORD"를 본인 비밀번호로 바꿉니다(Neo4j Desktop은 APOC 플러그인 필요, 02a 머리말 참고).
    조회할 데이터는 3절의 ingest(cd ch08_knowledge_base && uv run python -m chapter8 ingest)로 세 작품을 같은 DB에 넣어 두면 생깁니다.
각 셀은 조회 결과를 출력하지 않으므로, 셀 단위로 실행한 뒤 results나 graph.query(...)의 반환값을 확인합니다.
"""

# .env 파일의 API 키를 환경 변수로 불러옵니다.
from dotenv import load_dotenv
load_dotenv()

# %% 6. GraphRAG 활용하기 - 작가와 주제로 작품 찾기
# (a:Author)-[:WROTE]->(b:Book)-[:HAS_THEME]->(t:Theme) 패턴 하나로 "작가 -> 작품 -> 주제" 관계를 한 번에 따라갑니다.
# graph.query()는 RETURN에서 붙인 별칭(AS title)을 키로 하는 딕셔너리의 리스트를 돌려줍니다.
from langchain_neo4j import Neo4jGraph

graph = Neo4jGraph(url="bolt://localhost:7687", username="neo4j", password="YOUR_PASSWORD")

# 관계 기반 검색
query = """
MATCH (a:Author {name: $author})-[:WROTE]->(b:Book)-[:HAS_THEME]->(t:Theme {name: $theme})
RETURN b.title AS title
"""

results = graph.query(query, params={
    "author": "현진건",
    "theme": "일제강점기"
})


# %% 6. GraphRAG 활용하기 - 인물과 인물 사이의 관계 찾기
# 작품의 인물(HAS_CHARACTER)에서 RELATION 관계를 따라가, 관계 유형(rtype)과 근거 문장(evidence)을 함께 꺼냅니다.
query = """
MATCH (b:Book {title: $title})-[:HAS_CHARACTER]->(c1:Character)-[r:RELATION]->(c2:Character)
RETURN c1.name AS source, r.rtype AS relation, c2.name AS target, r.evidence AS evidence
"""
graph.query(query, params={"title": "날개"})


# %% 6. GraphRAG 활용하기 - 사건과 참여 인물(사건-인물 관계) 찾기
# <-[:PARTICIPATED_IN]-는 인물에서 사건으로 들어오는 관계입니다. collect()로 사건마다 참여 인물 이름을 리스트로 모읍니다.
query = """
MATCH (b:Book {title: $title})-[:HAS_EVENT]->(e:Event)<-[:PARTICIPATED_IN]-(c:Character)
RETURN e.id AS event_id, e.description AS event, collect(c.name) AS participants, e.impact AS impact
"""
graph.query(query, params={"title": "동백꽃"})


# %% 6. GraphRAG 활용하기 - 행동/생각/상징의 의미(해석) 찾기
# 행동, 생각, 상징의 의미를 담은 Interpretation 노드를 꺼냅니다. 문학 시험에 빠짐없이 나오는 질문이 이 세 갈래로 모이기 때문에
# 스키마를 처음부터 이 구조에 맞게 잡았습니다.
query = """
MATCH (b:Book {title: $title})-[:HAS_INTERPRETATION]->(i:Interpretation)
RETURN i.subject AS subject, i.meaning AS meaning, i.evidence AS evidence
"""
graph.query(query, params={"title": "날개"})
