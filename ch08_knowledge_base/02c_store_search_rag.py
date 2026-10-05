"""
[8장 2절] [GraphRAG] 정보를 저장하고 불러오기 - 4. 에이전트 지식베이스 쿼리 ~ 5. RAG 활용하기

이 절에서 배우는 것:
- 지식베이스 검색은 두 갈래입니다. 위치를 아는 정보는 직접 검색으로, 어디 있을지 감만 오는 정보는 벡터 검색으로 찾습니다.
- 직접 검색: get(namespace, key)로 항목 하나를, search(namespace)로 네임스페이스의 항목 전체를 꺼냅니다. 빠르고 정확합니다.
- 벡터 검색: 스토어를 만들 때 임베딩 인덱스를 설정하고 search()에 query=를 넘기면, 정확한 키워드가 없어도 의미가 비슷한 노트를 찾습니다.
- RAG(검색 증강 생성): 검색한 노트를 컨텍스트로 LLM에 넘겨, 되찾은 정보를 근거로 답하게 합니다.

실행 방법:
    uv run python ch08_knowledge_base/02c_store_search_rag.py
필요한 환경 변수: OPENAI_API_KEY (Neo4j는 쓰지 않습니다)
책의 조회 예시는 새로 만든 빈 스토어를 검색하므로 검색 결과는 비어 있고, 마지막 RAG 답변만 출력됩니다.
"""

# .env 파일의 API 키를 환경 변수로 불러옵니다.
from dotenv import load_dotenv
load_dotenv()

# %% 4. 에이전트 지식베이스 쿼리 - get(): 정확한 키로 찾기
# 직접 검색은 어디에 뭐가 있는지 알 때 가장 빠른 방법입니다. 작품 제목을 정확히 알면 프로필을 바로 가져옵니다.
# 네임스페이스에 user_id를 넣어 학생별로 노트를 나눴습니다. get()은 항목이 없으면 None을 돌려주므로 if profile:로 확인합니다.
from langgraph.store.memory import InMemoryStore

store = InMemoryStore()
user_id = "student_001"
ns_profiles = ("literature", "profiles", user_id)

# 작품 프로필 가져오기
profile = store.get(ns_profiles, "운수 좋은 날")
if profile:
    print(f"작가: {profile.value['author']}")
    print(f"주제: {profile.value['themes']}")


# %% 4. 에이전트 지식베이스 쿼리 - search(): 네임스페이스의 모든 항목 가져오기
# 특정 작가의 모든 작품이나 특정 작품의 모든 메모처럼 묶음을 가져올 때는 search(namespace)를 씁니다.
# 결과는 검색 항목(SearchItem)의 리스트이고, 저장한 딕셔너리는 각 항목의 .value에 들어 있습니다.
# 특정 작품의 모든 메모 가져오기
user_id = "student_001"
ns_memos = ("literature", "memos", user_id, "날개")
all_memos = list(store.search(ns_memos))

for memo in all_memos:
    print(f"타입: {memo.value['memo_type']}")
    print(f"내용: {memo.value['content']}")
    print("---")


# %% 4. 에이전트 지식베이스 쿼리 - 필터링: 메모 타입별로 찾기
# 메모 컬렉션에서 특정 타입만 골라내려면 검색 결과를 value의 memo_type으로 걸러 냅니다.
# "quote" 타입 메모만 찾기
quote_memos = [
    memo for memo in store.search(ns_memos)
    if memo.value.get("memo_type") == "quote"
]


# %% 4. 에이전트 지식베이스 쿼리 - 벡터 인덱스 설정
# "일제강점기 시대의 고통을 다룬 작품"처럼 의미로 찾으려면 벡터 검색이 필요합니다.
# 벡터 검색을 쓰려면 Store를 만들 때 index에 임베딩 모델(embed)과 벡터 차원(dims)을 설정합니다.
from langgraph.store.memory import InMemoryStore
from langchain.embeddings import init_embeddings

# 임베딩 모델 초기화
embeddings = init_embeddings("openai:text-embedding-3-small")

# 벡터 인덱스가 있는 Store 생성
store = InMemoryStore(
    index={
        "embed": embeddings,
        "dims": 1536,  # text-embedding-3-small의 차원
    }
)


# %% 4. 에이전트 지식베이스 쿼리 - 의미 기반 검색
# search()에 query를 넘기면 벡터 검색이 되고, 결과마다 유사도 점수(score)가 붙습니다. limit은 돌려받을 최대 개수입니다.
# search()의 첫 인자는 네임스페이스 접두사(prefix)라서 ("literature", "memos", user_id) 아래의 작품별 메모 서랍까지 함께 검색합니다.
user_id = "student_001"
ns_memos = ("literature", "memos", user_id)

# "일제강점기 시대의 고통을 다룬 작품"과 의미적으로 비슷한 메모 찾기
results = list(store.search(
    ns_memos,
    query="일제강점기 시대의 고통을 다룬 작품",
    limit=3
))

for result in results:
    print(f"작품: {result.value['work_title']}")
    print(f"내용: {result.value['content']}")
    print(f"유사도 점수: {result.score:.3f}")
    print("---")


# %% 4. 에이전트 지식베이스 쿼리 - 필터 + 벡터 검색 조합
# 네임스페이스를 작품까지 좁힌 뒤 벡터 검색하면 그 작품의 메모 안에서만 의미가 가까운 것을 찾습니다.
# "날개" 작품의 메모만 검색
user_id = "student_001"
ns_work = ("literature", "memos", user_id, "날개")
results = list(store.search(
    ns_work,
    query="가난과 비극",
    limit=5
))


# %% 5. RAG 활용하기
# 검색한 노트를 바탕으로 답변을 만드는 간단한 그래프입니다. 노드 함수에 store: BaseStore 인자를 두면
# compile(store=store)로 넘긴 스토어를 LangGraph가 넣어 주고, config: RunnableConfig로는 invoke에 넘긴 user_id를 읽습니다.
# 검색으로 되찾은 정보를 근거로 답하게 하므로 환각을 줄이고 출처를 맞추기 쉽습니다.
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.store.base import BaseStore
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def answer_with_knowledge(
    state: MessagesState,
    config: RunnableConfig,
    *,
    store: BaseStore
):
    """지식베이스를 검색하여 질문에 답하는 노드"""
    user_id = config.get("configurable", {}).get("user_id", "default_user")

    # 사용자의 질문 추출
    last_msg = state["messages"][-1]
    if isinstance(last_msg, HumanMessage):
        question = last_msg.content

        # 벡터 검색으로 관련 메모 찾기
        ns_memos = ("literature", "memos", user_id)
        relevant_memos = list(store.search(
            ns_memos,
            query=question,
            limit=5
        ))

        # 검색 결과를 컨텍스트로 구성
        context = "관련 노트:\n"
        for memo in relevant_memos:
            context += f"- [{memo.value['work_title']}] {memo.value['content']}\n"

        # LLM에게 컨텍스트와 함께 질문 전달
        system_msg = SystemMessage(content=f"""당신은 문학 작품에 대해 답변하는 도우미입니다.

{context}

위 노트를 참고하여 사용자의 질문에 답변하세요.""")

        response = llm.invoke([system_msg, last_msg])
        return {"messages": [response]}

    return state

# 그래프 구성
builder = StateGraph(MessagesState)
builder.add_node("answer", answer_with_knowledge)
builder.add_edge(START, "answer")
builder.add_edge("answer", END)

graph = builder.compile(store=store)

# 실행
config = {"configurable": {"user_id": "student_001"}}
result = graph.invoke({
    "messages": [HumanMessage(content="일제강점기 시대의 고통을 다룬 작품을 알려줘")]
}, config=config)

print(result["messages"][-1].content)
