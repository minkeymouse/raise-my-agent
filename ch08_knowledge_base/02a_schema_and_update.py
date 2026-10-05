"""
[8장 2절] [GraphRAG] 정보를 저장하고 불러오기 - 1. 스키마 설계하기 ~ 2. 지식베이스 업데이트하기

이 절에서 배우는 것:
- 정보를 어떤 틀(스키마)로 저장할지는 해결하려는 과제와 데이터의 특성을 보고 사람이 설계합니다.
- 스토어에는 작품 하나를 대표하는 요약 노트(WorkProfile)와 읽으면서 계속 쌓이는 단편 메모(WorkMemo)를 둡니다.
  프로필은 같은 key로 put()해 덮어쓰고, 메모는 새 UUID key로 계속 추가합니다.
- 그래프에는 Book, Author, Theme 같은 노드와 WROTE, HAS_THEME 같은 관계를 두고, Cypher MERGE로 '없으면 만들고 있으면 재사용'합니다.

8장 2절 코드는 책의 소제목 순서대로 02a -> 02b -> 02c -> 02d 네 파일로 나눴습니다. 이 순서대로 실행합니다.

실행 방법:
    uv run python ch08_knowledge_base/02a_schema_and_update.py
필요한 환경 변수: 없음 (LLM을 호출하지 않습니다)
준비물: Neo4j (마지막 셀, 설치는 01_store_basics.py 머리말 참고). 책처럼 접속 정보를 코드에 적으므로
    실행 전에 password="YOUR_PASSWORD"를 본인 비밀번호로 바꿉니다. Aura를 쓰면 url도 neo4j+s://... 주소로 바꿉니다.
참고: Neo4jGraph는 접속할 때 APOC 프로시저로 그래프 스키마를 읽습니다. Neo4j Desktop에서 "Could not use APOC procedures"
    오류가 나면 인스턴스의 플러그인(Plugins) 메뉴에서 APOC을 설치합니다. 02b, 02d도 같습니다.
"""

# .env 파일의 API 키를 환경 변수로 불러옵니다.
from dotenv import load_dotenv
load_dotenv()

# %% 1. 스키마 설계하기 - 스토어 설계
# WorkProfile은 제목, 작가, 요약, 주제, 시기처럼 한 작품을 한 번에 잡는 요약 노트입니다("한 작품 = 한 항목").
# WorkMemo는 인물, 문체, 상징, 중요한 문장, 시험 포인트처럼 근거 단위로 쌓이는 메모입니다("한 작품 = 여러 항목").
# 저장 구조도 함께 정합니다. 프로필은 ("literature", "profiles") 아래 key=work_title, 메모는 ("literature", "memos", work_title) 아래 key=uuid입니다.
from typing import TypedDict, List, Literal

class WorkProfile(TypedDict):
    work_title: str
    author: str
    summary: str
    themes: List[str]
    period: str  # 예: "일제강점기", "근대"

class WorkMemo(TypedDict):
    work_title: str
    memo_type: Literal["quote", "character", "style", "symbol", "theme", "plot", "exam"]
    content: str
    tags: List[str]


# %% 1. 스키마 설계하기 - 그래프 데이터베이스 설계
# 스토어가 노트를 '보관'하기 좋다면, 그래프는 노트 사이의 '연결'을 따라가며 답하기 좋습니다.
# "작가 -> 작품"으로 묶고, "작품 -> 주제/인물/핵심 문장"으로 넓히고, 주제나 인물을 매개로 작품을 비교하는 질문에 맞춰 정합니다.
# 노드: Book, Author, Theme, Character, Quote
# 관계: WROTE, HAS_THEME, HAS_CHARACTER, HAS_QUOTE
# 책의 Cypher 예시("없으면 만들고(MERGE), 있으면 재사용")는 아래 '2. 지식베이스 업데이트하기 - Neo4j' 셀의 query 문자열과 같습니다.


# %% 2. 지식베이스 업데이트하기 - 스토어: 작품 프로필 덮어쓰기
# 같은 namespace와 key로 put()을 다시 호출하면 기존 값을 덮어씁니다. 프로필은 한 장짜리 요약 노트라 이 방식이 자연스럽습니다.
# 주의할 점은 부분 업데이트가 아니라 전체 교체라는 것입니다. 일부 필드만 넣으면 나머지가 사라지므로, get()으로 꺼내 고친 뒤 통째로 저장합니다.
from langgraph.store.memory import InMemoryStore

store = InMemoryStore()
ns_profiles = ("literature", "profiles")

# 1) 초기 저장
store.put(ns_profiles, "운수 좋은 날", {
    "work_title": "운수 좋은 날",
    "author": "현진건",
    "summary": "인력거꾼 김첨지가 돈을 벌어오지만, 집에서는 병든 아내의 죽음이 기다린다. 비극적 아이러니로 하루의 '운수'를 뒤집는 작품.",
    "themes": ["빈곤", "아이러니", "비극"],
    "period": "근대",
})

# 2) 업데이트(전체를 다시 저장)
profile = store.get(ns_profiles, "운수 좋은 날").value
profile["themes"] = list(set(profile["themes"] + ["도시 하층민"]))
store.put(ns_profiles, "운수 좋은 날", profile)


# %% 2. 지식베이스 업데이트하기 - 스토어: 메모 추가하기
# 메모는 시간이 갈수록 늘어나므로 한 덩어리로 합치지 않고 "한 메모 = 한 항목"으로 저장합니다.
# 메모마다 uuid4로 고유한 key를 붙이므로 덮어쓰지 않고 계속 쌓입니다.
import uuid

ns_memos = ("literature", "memos", "동백꽃")

memo_id = str(uuid.uuid4())
store.put(ns_memos, memo_id, {
    "work_title": "동백꽃",
    "memo_type": "quote",
    "content": "오늘도 또 우리 수탉이 막 쫓기었다.",
    "tags": ["도입", "갈등", "서술"],
})


# %% 2. 지식베이스 업데이트하기 - Neo4j
# 그래프 업데이트는 프로필과 메모에서 얻은 구조화된 정보를 노드와 관계로 옮겨 심는 작업입니다.
# Neo4jGraph는 그래프의 스키마를 자동으로 분석해 LLM이 이해할 수 있는 형태로 제공하므로, 에이전트가 적절한 쿼리를 만들 수 있습니다.
# 쿼리는 텍스트로 적고, Neo4jGraph의 query() 메서드로 실행합니다. $author 같은 자리에는 params의 값이 들어갑니다.
from langchain_neo4j import Neo4jGraph

graph = Neo4jGraph(
    url="bolt://localhost:7687",
    username="neo4j",
    password="YOUR_PASSWORD",
)

# MERGE는 노드나 관계가 없으면 만들고 있으면 재사용합니다. UNWIND는 리스트($themes)를 한 행씩 펼쳐 주제마다 MERGE를 반복합니다.
# 참고: MERGE 바로 뒤에 UNWIND를 쓰는 문법은 Cypher 25에서만 허용됩니다. Cypher 5 DB(Neo4j 5.x 등)에서 "WITH is required between
#       MERGE and UNWIND" 오류가 나면, Neo4j 2025.06 이상에서 쿼리 맨 앞에 CYPHER 25를 붙여 실행합니다.
query = """
MERGE (a:Author {name: $author})
MERGE (b:Book {title: $title})
MERGE (a)-[:WROTE]->(b)
UNWIND $themes AS t
  MERGE (theme:Theme {name: t})
  MERGE (b)-[:HAS_THEME]->(theme)
"""

graph.query(query, params={
    "author": "이상",
    "title": "날개",
    "themes": ["소외", "자아", "불안"],
})
