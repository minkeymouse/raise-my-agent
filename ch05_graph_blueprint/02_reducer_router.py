"""
[5장 2절] [리듀서, 라우터] 어떻게 바꾸고 어디로 갈까?

이 절에서 배우는 것:
- 새 정보가 예전 기억을 덮어쓰는 '건망증'은 리듀서(Reducer)로, 정해진 순서로만 움직이는 '직진 본능'은 라우터(Router)로 풉니다.
- 리듀서: Annotated[list, operator.add]를 붙인 필드는 덮어쓰지 않고 뒤에 이어 쌓습니다.
- 라우터: 상태를 읽고 다음 행선지 이름만 돌려주는 조건 함수를 add_conditional_edges로 그래프에 등록합니다.
- 둘을 결합해 최대 3번까지 재검색하는 '끈질긴 검색 에이전트'를 실행합니다.

실행 방법:
    uv run python ch05_graph_blueprint/02_reducer_router.py
필요한 환경 변수: 없음 (LLM을 호출하지 않습니다)
표시: [보충] 실행을 위해 더한 코드, [수정] 책 코드의 오류를 고친 곳, [설명용 코드] 실행되지 않는 설명용 조각
"""

# .env 파일의 API 키를 환경 변수로 불러옵니다.
from dotenv import load_dotenv
load_dotenv()

# %% 1. 리듀서(Reducer): 기억을 합치는 기술 - 코드 구현: Annotated의 마법
# 상태는 기본적으로 덮어쓰기(칠판) 방식입니다. 리듀서는 "새 정보가 들어오면 기존 정보와 어떻게 합칠까?"를 정하는 규칙으로,
# Annotated[list, operator.add]를 붙이면 그 필드는 일기장처럼 기존 리스트 뒤에 이어 씁니다.
# operator.add는 더하기 연산을 함수 형태로 넘깁니다. 연산 결과가 아니라 '나중에 수행할 연산' 자체를 전달하는 것입니다.
# 참고: 1절의 MessagesState도 내부에 messages: Annotated[list, add_messages]로 정의되어 있어 대화 기록이 계속 쌓입니다.
import operator
from typing import Annotated, TypedDict

class MemoryState(TypedDict):
    # 1. 일반 필드: 새로운 값이 들어오면 기존 값을 덮어씁니다.
    summary: str

    # 2. 리듀서 필드: 새로운 값이 들어오면 기존 리스트 뒤에 붙입니다.
    # operator.add는 "기존 리스트 + 새 리스트"를 수행합니다.
    chat_history: Annotated[list, operator.add]


# %% 2. 라우터(Router): 길을 선택하는 눈
# 라우터는 조건부 엣지에서 '다음에 어느 노드로 갈지'를 판단하는 로직입니다. 라우터의 3박자는 다음과 같습니다.
# (1) 조건 함수: 상태를 보고 행선지 이름(문자열)을 반환  (2) 경로 지도: 그 이름과 실제 노드를 잇는 사전
# (3) 연결: add_conditional_edges로 그래프에 등록. 라우터 함수는 상태를 바꾸지 않고 읽기만 합니다.
# [보충] 아래 예시는 1절의 AgentState와 그래프 빌더(workflow)를 이어서 사용하므로 1절의 코드를 옮겨 둡니다.
#        책처럼 라우터를 등록만 하고 실행하지는 않으므로 이 셀은 출력이 없습니다.
from langgraph.graph import MessagesState
from langgraph.graph import StateGraph, START, END

class AgentState(MessagesState):
    # MessagesState에는 이미 'messages'라는 필드가 들어있습니다.
    # 여기에 우리가 필요한 추가 필드만 더하면 됩니다.
    next_step: str      # 다음에 무엇을 할지 결정하는 플래그

# 1. 그래프 빌더 생성 (설계도판 깔기)
workflow = StateGraph(AgentState)

# 여기부터 책의 라우터 예시입니다. 에이전트가 '도구를 쓸까 말까?'를 결정합니다.
from typing import Literal

# 1. 조건 함수: 상태를 보고 '다음 행선지 이름'을 반환
def route_step(state: AgentState) -> Literal["tools", "end"]:
    # 만약 LLM이 도구 사용을 요청했다면?
    if state.get("tool_calls"):
        return "tools"

    # 도구 쓸 일이 없다면?
    return "end"

# 2. 그래프에 연결
# "agent 노드가 끝나면 route_step 함수를 실행해서 길을 정해라!"
workflow.add_conditional_edges(
    "agent",            # 출발 노드
    route_step,         # 길잡이 함수 (Router)
    {
        "tools": "tool_node",  # 함수가 'tools'를 뱉으면 -> tool_node로 이동
        "end": END             # 함수가 'end'를 뱉으면 -> 종료
    }
)

# 참고: 개념 설명용 예시라 상태의 tool_calls 키를 읽습니다. MessagesState 기반 그래프에서는 도구 호출 요청이
#       마지막 메시지에 들어 있으므로 state["messages"][-1].tool_calls를 확인합니다(3절의 should_continue).


# %% 3. 예제: 반복해서 검색하는 끈질긴 에이전트
# 리듀서와 라우터를 결합해, 정보를 찾을 때까지 최대 3번 재검색하는 에이전트를 만듭니다.
# results는 리듀서로 누적하고 try_count는 덮어씁니다. search 노드는 횟수를 1 늘리고 결과를 리스트로 돌려줍니다.
# 라우터 should_continue는 try_count가 3 미만이면 자기 자신(search)으로 돌아가고(Loop), 아니면 종료합니다.
import operator
from typing import Annotated, TypedDict, Literal
from langgraph.graph import StateGraph, START, END

# 1. 상태 정의 (리듀서 활용)
class SearchState(TypedDict):
    query: str
    # 검색 결과를 계속 누적하기 위해 리듀서 사용
    results: Annotated[list, operator.add]
    try_count: int # 시도 횟수 (덮어쓰기)

# 2. 노드 정의
def search_node(state: SearchState):
    print(f" {state['try_count']}번째 검색 시도 중...")
    # (가상의 검색 결과)
    new_result = f"{state['try_count']}번째 찾은 정보"

    # 횟수는 1 증가시키고, 결과는 리스트로 반환 (Append 됨)
    return {"try_count": state["try_count"] + 1, "results": [new_result]}

# 3. 라우터 정의 (로직)
def should_continue(state: SearchState) -> Literal["retry", "finish"]:
    # 3번 미만이면 다시 검색(retry), 3번 다 채웠으면 종료(finish)
    if state["try_count"] < 3:
        return "retry"
    return "finish"

# 4. 그래프 조립
builder = StateGraph(SearchState)
builder.add_node("search", search_node)

builder.add_edge(START, "search")

# 조건부 엣지 등록
builder.add_conditional_edges(
    "search",
    should_continue,
    {
        "retry": "search",  # 다시 자기 자신을 호출 (Loop)
        "finish": END
    }
)

# 5. 실행
# compile로 설계도를 확정하고, invoke는 그래프를 END까지 실행한 뒤 최종 상태를 돌려줍니다.
app = builder.compile()
initial_state = {"query": "LangGraph", "results": [], "try_count": 0}
final_state = app.invoke(initial_state)

print("--- 최종 결과 ---")
print(final_state["results"])
# 결과: ['0번째 찾은 정보', '1번째 찾은 정보', '2번째 찾은 정보'] 가 모두 쌓여있음!
