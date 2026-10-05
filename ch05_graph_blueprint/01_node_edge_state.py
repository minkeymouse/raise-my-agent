"""
[5장 1절] [노드, 엣지, 상태] 한눈에 잡는 실행 뼈대

이 절에서 배우는 것:
- LangGraph의 세 요소를 "노드 = 할 일, 엣지 = 순서, 상태 = 기억"으로 이해합니다.
- TypedDict와 MessagesState로 노드끼리 주고받는 공유 가방, 상태(State)를 정의합니다.
- 노드는 상태를 받아 '바뀌어야 할 부분'만 딕셔너리로 돌려주는 함수입니다.
- add_node, add_edge, add_conditional_edges로 검색 -> 답변 뼈대(Graph)를 조립하고 실행 흐름을 확인합니다.

실행 방법:
    uv run python ch05_graph_blueprint/01_node_edge_state.py
필요한 환경 변수: 없음 (LLM을 호출하지 않습니다)
"""

# .env 파일의 API 키를 환경 변수로 불러옵니다.
from dotenv import load_dotenv
load_dotenv()

# %% 2. 상태(State): 생각의 공유 메모리
# 가장 먼저 "무엇을 기억할 것인가?"를 정합니다. 모든 노드는 하나의 공유된 상태를 돌려 보며 일하므로
# 가방을 잘 정리해 두어야 합니다. TypedDict로 필드를 직접 정의할 수도 있고, 채팅 모델 에이전트라면
# 대화 기록(messages)을 자동으로 관리해 주는 MessagesState를 상속받아 필요한 필드만 더하는 방식을 주로 씁니다.
from typing import TypedDict, Annotated, List
from langgraph.graph import MessagesState

# 1. 가장 기초적인 형태의 상태
class BasicState(TypedDict):
    question: str       # 사용자의 질문
    answer: str         # 에이전트의 답변
    context: List[str]  # 검색된 정보들

# 2. (추천) LangGraph가 미리 만들어둔 '메시지 전용' 상태 상속받기
class AgentState(MessagesState):
    # MessagesState에는 이미 'messages'라는 필드가 들어있습니다.
    # 여기에 우리가 필요한 추가 필드만 더하면 됩니다.
    next_step: str      # 다음에 무엇을 할지 결정하는 플래그


# %% 3. 노드(Node): 실행의 주체
# 노드는 그래프에서 실제 행동(Action)을 맡는, '상태를 입력받아 변경된 상태를 반환하는 함수'입니다.
# 항상 현재 상태(state)를 인자로 받고, 바뀌어야 할 부분만 딕셔너리로 돌려줍니다.
# 돌려준 값을 상태에 반영하는 일은 LangGraph가 맡습니다.
def search_node(state: AgentState) -> AgentState:
    """정보를 검색하는 노드 (행동)"""
    print("검색을 수행합니다...")

    # ... 실제 검색 로직 (생략) ...
    found_data = "LangGraph는 에이전트 프레임워크입니다."

    # 중요: 기존 상태를 덮어쓰는 게 아니라, '업데이트할 내용'만 반환합니다.
    return {"context": [found_data], "next_step": "answer"}

def answer_node(state: AgentState) -> AgentState:
    """답변을 생성하는 노드 (행동)"""
    print("답변을 작성합니다...")

    return {"answer": "검색 결과에 따르면..."}


# %% 4. 엣지(Edge)와 그래프 조립
# 부품(State, Node)을 엣지로 연결해 뼈대를 완성합니다. 순서는 빌더 생성 -> 노드 배치 -> 엣지 연결 -> 컴파일입니다.
# add_edge: 작업장 사이를 잇는 도로로, 무조건 다음 노드로 넘어갑니다. START와 END는 그래프의 시작점과 끝점입니다.
# add_conditional_edges: 함수(check_result)가 돌려준 값을 경로 지도에서 찾아 방향을 트는 갈림길입니다.
# 이렇게 길을 정하는 함수를 라우터(Router)라고 부르며, 자세한 내용은 2절에서 다룹니다.
from langgraph.graph import StateGraph, START, END

# 1. 그래프 빌더 생성 (설계도판 깔기)
workflow = StateGraph(AgentState)

# 2. 노드 배치 (스티커 붙이기)
workflow.add_node("search", search_node)
workflow.add_node("answer", answer_node)

# 3. 엣지 연결 (선 긋기)
# START는 에이전트가 처음 시작하는 지점입니다.
workflow.add_edge(START, "search")

# 4. 조건부 엣지 (갈림길 만들기)
# "search 노드가 끝나면, check_result 함수를 보고 길을 정해라"
def check_result(state: AgentState):
    if state.get("next_step") == "answer":
        return "answer" # 답변 작성으로 이동
    return "search"     # 다시 검색 (Loop)

workflow.add_conditional_edges(
    "search",           # 출발 노드
    check_result,       # 길을 정하는 로직 (기초적인 라우터이기도 합니다!)
    {
        "search": "search",
        "answer": "answer"
    }
)

# 5. 종료 지점 연결
workflow.add_edge("answer", END)

# 6. 컴파일 (설계도 확정)
app = workflow.compile()


# %% 5. 실행 결과 확인
# 초기 상태를 넣고 stream으로 실행하면 노드 하나가 끝날 때마다 {노드 이름: 그 노드가 돌려준 업데이트}가 나옵니다.
# search_node는 검색이 끝나면 항상 next_step을 "answer"로 바꾸므로, 라우터가 answer로 길을 정하고
# answer 노드 다음에 END에 닿아 그래프가 끝납니다.
# 초기 상태 주입 및 실행
inputs = {"question": "LangGraph가 뭐야?"}

for event in app.stream(inputs):
    for key, value in event.items():
        print(f"   Node: {key}")
        print(f"   State Update: {value}")
        print("---")

# [책의 실행 결과]
# 검색을 수행합니다...
#    Node: search
#    State Update: {'context': ['LangGraph는...'], 'next_step': 'answer'}
#
# 답변을 작성합니다...
#    Node: answer
#    State Update: {'answer': '검색 결과에 따르면...'}
#
# 참고: 상태에는 정의된 키만 반영됩니다. AgentState에는 context, answer가 없어 LangGraph가 이 값을 버리므로
#       실제로는 State Update가 {'next_step': 'answer'}와 None으로 출력됩니다.
