"""
[8장 2절] [GraphRAG] 정보를 저장하고 불러오기 - 3. 에이전트의 지식베이스 업데이트

이 절에서 배우는 것:
- 02a는 사람이 직접 고른 정보를 저장하는 룰 기반 방식이었습니다. 그래프 데이터베이스는 LLM이 직접 의미를 추출하고
  그 관계를 저장할 때 빛을 발합니다.
- 문학 문제가 자주 묻는 인물 관계, 사건의 영향, 행동·생각의 의미를 필드로 정한 '관계 중심 노트' 스키마(LiteratureExtraction)를 만듭니다.
- with_structured_output으로 소설 발췌를 스키마에 맞는 구조화된 데이터로 받고, Cypher 쿼리(save_to_graph)로 Neo4j에 저장합니다.
- 이 과정을 LangGraph 노드로 만들어, 에이전트가 작품을 읽을 때마다 그래프가 업데이트되게 합니다.

실행 방법:
    uv run python ch08_knowledge_base/02b_extract_to_graph.py
필요한 환경 변수: OPENAI_API_KEY
준비물: Neo4j. 실행 전에 password="YOUR_PASSWORD"를 본인 비밀번호로 바꿉니다(Neo4j Desktop은 APOC 플러그인 필요, 02a 머리말 참고).
표시: [보충] 실행을 위해 더한 코드, [수정] 책 코드의 오류를 고친 곳, [설명용 코드] 실행되지 않는 설명용 조각
"""

# .env 파일의 API 키를 환경 변수로 불러옵니다.
from dotenv import load_dotenv
load_dotenv()

from functools import partial  # [보충] 마지막 셀의 [수정]에서 노드 함수에 인자를 바인딩할 때 사용

# %% 3. 에이전트의 지식베이스 업데이트 - 관계 중심 추출 스키마
# 시험 풀이에 도움 되는 노트는 단순 요약이나 주제보다 구체적이어야 합니다. 문학 문제는 인물 A와 B의 관계, 사건이 누구에게 준 영향,
# 행동·생각이 갖는 의미를 자주 묻기 때문에 Character, CharacterRelation, Event, Interpretation을 따로 정의해 LiteratureExtraction에 담습니다.
# 필드를 미리 정해 두면 빈 칸이나 엉뚱한 타입이 줄고, 그래프에 넣을 때도 같은 모양으로 반복하기 쉽습니다. Field의 description은 LLM이 읽는 설명입니다.
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

class Character(BaseModel):
    name: str = Field(description="등장인물 이름(또는 지칭)")
    role: Optional[str] = Field(default=None, description="서술자/주인공/조연 등 역할")
    traits: List[str] = Field(default_factory=list, description="성격/상태/특징 키워드")

class CharacterRelation(BaseModel):
    source: str = Field(description="관계의 주체 인물 이름")
    target: str = Field(description="관계의 대상 인물 이름")
    # 관계 유형은 Literal로 보기를 제한하고, 판단하기 어려우면 unknown을 고르게 합니다.
    relation: Literal["conflict", "affection", "dependence", "control", "family", "friendship", "unknown"] = Field(
        description="관계 유형(필요시 unknown)"
    )
    evidence_quote: Optional[str] = Field(default=None, description="관계를 뒷받침하는 짧은 근거 문장(가능하면 원문)")

class Event(BaseModel):
    event_id: str = Field(description="사건 식별자(짧은 문자열)")
    description: str = Field(description="사건 요약(무슨 일이 일어났는지)")
    participants: List[str] = Field(description="사건에 관여한 인물 이름들")
    impact: List[str] = Field(default_factory=list, description="사건이 만든 변화/결과(감정/행동/관계 변화)")

class Interpretation(BaseModel):
    subject: str = Field(description="행동/생각/상징 등 해석 대상(짧게)")
    meaning: str = Field(description="그 의미(주제/인물 상태/서술 효과와 연결)")
    evidence_quote: Optional[str] = Field(default=None, description="근거 문장(가능하면 원문)")

class LiteratureExtraction(BaseModel):
    """문학 작품에서 '관계'까지 포함해 추출할 구조화된 정보"""
    work_title: str = Field(description="작품 제목")
    author: str = Field(description="작가 이름")
    summary: str = Field(description="작품(또는 발췌)의 간단 요약")
    themes: List[str] = Field(default_factory=list, description="주제 키워드")
    characters: List[Character] = Field(default_factory=list, description="등장인물 목록")
    relations: List[CharacterRelation] = Field(default_factory=list, description="등장인물 간 관계")
    events: List[Event] = Field(default_factory=list, description="사건(인물-사건 연결 중심)")
    interpretations: List[Interpretation] = Field(default_factory=list, description="행동/생각/상징의 의미 해석(근거 포함)")
    key_quotes: List[str] = Field(default_factory=list, description="핵심 근거 문장 1~5개")


# %% 3. 에이전트의 지식베이스 업데이트 - with_structured_output으로 정보 추출
# with_structured_output은 LLM이 자유 형식 텍스트 대신 미리 정해 둔 스키마(LiteratureExtraction)에 맞는 데이터를 돌려주도록 강제합니다.
# 결과 extracted는 LiteratureExtraction 객체이므로 extracted.characters처럼 필드로 꺼내 쓰고, 바로 Neo4j 그래프로 옮길 수 있습니다.
# LLM에 구조화 출력 기능 추가
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
structured_llm = llm.with_structured_output(LiteratureExtraction)

# 작품 텍스트 일부를 분석
text_sample = """
오늘도 또 우리 수탉이 막 쫓기었다. 내가 점심을 먹고 나무를 하러 갈 양으로 나올 때이었다.
산으로 올라서려니까 등뒤에서 푸르득푸드득, 하고 닭의 횃소리가 야단이다.
"""

extracted = structured_llm.invoke([
    HumanMessage(f"다음 문학 작품 발췌를 분석하여 구조화된 정보를 추출해주세요:\n\n{text_sample}")
])


# %% 3. 에이전트의 지식베이스 업데이트 - 추출한 정보를 Neo4j 그래프에 저장
# save_to_graph는 추출 결과를 Cypher 쿼리 하나로 저장합니다. 작가-작품, 주제, 인물, 인물 관계(RELATION), 사건(Event)과 참여 인물,
# 해석(Interpretation), 핵심 문장(Quote)을 차례로 MERGE합니다. model_dump()로 Pydantic 객체를 딕셔너리로 바꿔 params로 넘깁니다.
from langchain_neo4j import Neo4jGraph

graph = Neo4jGraph(
    url="bolt://localhost:7687",
    username="neo4j",
    password="YOUR_PASSWORD",
)

def save_to_graph(extracted: LiteratureExtraction, graph: Neo4jGraph):
    """추출한 정보를 Neo4j 그래프에 저장"""
    # 참고: 리스트가 비면 UNWIND가 행을 만들지 않아 뒤쪽 절이 건너뛰어집니다(책 설명). 또 WITH ev, e에서 b가 빠져 해석·인용·요약이
    #       Book이 아닌 새 노드에 붙고, Cypher 25에서만 실행됩니다(02a 참고). 3절 chapter8/neo4j_store.py는 쿼리를 나눠 이를 피합니다.

    # 작가와 작품 노드 생성 및 관계 연결
    query = """
    MERGE (a:Author {name: $author})
    MERGE (b:Book {title: $title})
    MERGE (a)-[:WROTE]->(b)

    // 주제 노드 생성 및 연결
    UNWIND $themes AS theme_name
        MERGE (t:Theme {name: theme_name})
        MERGE (b)-[:HAS_THEME]->(t)

    // 인물 노드: 작품별로 이름이 겹칠 수 있어 work_title을 MERGE 키에 포함(저장소 chapter8과 동일한 아이디어)
    UNWIND $characters AS ch
        MERGE (c:Character {name: ch.name, work_title: $title})
        SET c.role = ch.role,
            c.traits = ch.traits
        MERGE (b)-[:HAS_CHARACTER]->(c)

    // 인물-인물 관계: Cypher에서 `type`은 예약어에 가깝기 때문에 속성명은 rtype을 씁니다.
    UNWIND $relations AS r
        MATCH (c1:Character {name: r.source, work_title: $title})
        MATCH (c2:Character {name: r.target, work_title: $title})
        MERGE (c1)-[rel:RELATION {rtype: r.relation}]->(c2)
        SET rel.evidence = r.evidence_quote

    // 사건 노드 + 참여 관계
    UNWIND $events AS e
        MERGE (ev:Event {id: e.event_id, work_title: $title})
        SET ev.description = e.description,
            ev.impact = e.impact
        MERGE (b)-[:HAS_EVENT]->(ev)
        WITH ev, e
        UNWIND e.participants AS p
            MATCH (c:Character {name: p, work_title: $title})
            MERGE (c)-[:PARTICIPATED_IN]->(ev)

    // 해석 노드 (행동/생각/상징의 의미)
    UNWIND $interpretations AS i
        MERGE (it:Interpretation {work_title: $title, subject: i.subject})
        SET it.meaning = i.meaning,
            it.evidence = i.evidence_quote
        MERGE (b)-[:HAS_INTERPRETATION]->(it)

    // 핵심 문장을 Quote 노드로 저장
    UNWIND $quotes AS quote_text
        MERGE (q:Quote {text: quote_text, work_title: $title})
        MERGE (b)-[:HAS_QUOTE]->(q)

    SET b.summary = $summary
    """

    graph.query(query, params={
        "author": extracted.author,
        "title": extracted.work_title,
        "themes": extracted.themes,
        "characters": [c.model_dump() for c in extracted.characters],
        "relations": [r.model_dump() for r in extracted.relations],
        "events": [e.model_dump() for e in extracted.events],
        "interpretations": [i.model_dump() for i in extracted.interpretations],
        "quotes": extracted.key_quotes,
        "summary": extracted.summary,
    })

# 추출한 정보를 그래프에 저장
save_to_graph(extracted, graph)


# %% 3. 에이전트의 지식베이스 업데이트 - LangGraph 노드로 만들기
# 추출과 저장을 노드 하나로 묶으면, 에이전트가 작품을 읽을 때마다 엔티티(작가, 작품, 주제, 인물)와 관계가 그래프에 저장됩니다.
# 노드는 마지막 HumanMessage를 작품 텍스트로 보고, 저장을 마치면 AIMessage로 알립니다. 아래 "[작품 텍스트 내용...]"은 책의 개념 예시로,
# 실제로는 이 자리에 작품 본문을 넣습니다.
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_neo4j import Neo4jGraph

def extract_and_save_to_graph(
    state: MessagesState,
    *,
    graph: Neo4jGraph,
    structured_llm
):
    """작품 텍스트를 읽고 그래프에 저장하는 노드"""

    # 마지막 메시지에서 텍스트 추출
    last_msg = state["messages"][-1]
    if isinstance(last_msg, HumanMessage):
        text_content = last_msg.content

        # 구조화된 정보 추출
        extracted = structured_llm.invoke([
            HumanMessage(f"다음 작품 텍스트를 분석하여 구조화된 정보를 추출해주세요:\n\n{text_content}")
        ])

        # 그래프에 저장
        save_to_graph(extracted, graph)

        return {
            "messages": [
                AIMessage(content=f"'{extracted.work_title}' 작품 정보를 그래프에 저장했습니다.")
            ]
        }

    return {"messages": []}

# 그래프 구성
builder = StateGraph(MessagesState)
# [수정] LangGraph는 graph, structured_llm 인자를 채워 주지 않아 invoke에서 TypeError가 납니다.
#        책의 주석대로 functools.partial로 두 값을 바인딩해 등록합니다. (책: 아래 주석 줄)
# builder.add_node("extract_and_save", extract_and_save_to_graph)
builder.add_node("extract_and_save", partial(extract_and_save_to_graph, graph=graph, structured_llm=structured_llm))
builder.add_edge(START, "extract_and_save")
builder.add_edge("extract_and_save", END)

# 그래프 컴파일 — 실제로는 Neo4jGraph·structured_llm을 바인딩해야 합니다(아래 invoke는 개념 예시).
graph_agent = builder.compile()

# 실행
result = graph_agent.invoke({
    "messages": [HumanMessage(content="[작품 텍스트 내용...]")]
})
