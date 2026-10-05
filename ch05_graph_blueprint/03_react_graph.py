"""
[5장 3절] [ReAct] 에이전트의 사고 과정

이 절에서 배우는 것:
- ReAct(Reasoning + Acting)는 "생각 -> 행동 -> 관찰"의 루프를 돌며 문제를 푸는 대표적인 에이전트 행동 패턴입니다.
- 생각은 agent 노드(LLM), 판단은 조건부 엣지(라우터), 행동은 ToolNode, 관찰은 상태 업데이트가 맡습니다.
- bind_tools, ToolNode, 라우터, tools -> agent 루프 엣지로 ReAct 그래프를 직접 조립합니다.
- 앞에서 쓴 create_agent는 이 ReAct 구조를 더 상세하게 구현해 둔 함수입니다.

실행 방법:
    uv run python ch05_graph_blueprint/03_react_graph.py
필요한 환경 변수: OPENAI_API_KEY (리포지토리 루트의 .env 파일에 넣어 둡니다)
"""

# .env 파일의 API 키를 환경 변수로 불러옵니다.
from dotenv import load_dotenv
load_dotenv()

# %% 2. ToolNode의 활용 - 검색 도구 정의
# 4장 자료조사 에이전트의 검색 도구를 예로 들되, 실제 검색 대신 정해진 문장을 돌려줍니다.
# @tool 데코레이터가 함수를 도구로 만들고, docstring은 모델이 도구의 쓰임새를 판단하는 설명이 됩니다.
from langchain_core.tools import tool

@tool
def search(query: str):
    """웹에서 정보를 검색합니다."""
    # (실제 검색 로직 대신 예시 반환)
    if "날씨" in query:
        return "오늘 서울의 날씨는 맑음, 기온은 20도입니다."
    return "검색 결과 없음"

tools = [search]


# %% 2. ToolNode의 활용 - 도구 실행 노드 만들기
# ReAct를 따르지 않으면 도구를 실행하는 노드와 엣지, 언제 어떤 도구를 쓸지를 일일이 구현해야 합니다.
# LangGraph의 ToolNode는 도구 리스트만 받으면, LLM이 도구 호출(tool_calls)을 요청할 때 해당 함수를 실행하고
# 결과를 메시지로 만들어 줍니다. 도구 선택은 에이전트가 알아서 한다고 가정하기 때문에 이렇게 단순해집니다.
from langgraph.prebuilt import ToolNode

# 도구 리스트만 넣어주면 알아서 실행 노드가 만들어집니다.
tool_node = ToolNode(tools)


# %% 3. ReAct 그래프 조립하기
# 1~2절에서 배운 노드, 엣지, 상태, 라우터로 ReAct 루프를 만듭니다.
# agent 노드(생각)의 응답에 tool_calls가 있으면 라우터가 tools 노드(행동)로 보내고, 없으면 END로 끝냅니다.
# tools -> agent 엣지가 루프를 만들어, 도구 결과(관찰)를 본 에이전트가 다시 생각하고 최종 답변을 합니다.
# MessagesState의 messages에는 add_messages 리듀서가 있어 agent와 tools가 돌려준 메시지가 대화 기록에 쌓입니다.
from typing import Literal
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START, END

# 1. 모델 준비 (도구 바인딩)
# bind_tools를 해야 모델이 "아, 나한테 이런 도구가 있구나"라고 인지합니다.
model = ChatOpenAI(model="gpt-4o").bind_tools(tools)

# 2. 에이전트 노드 정의 (생각하는 뇌)
def agent_node(state: MessagesState):
    messages = state["messages"]
    response = model.invoke(messages)
    # LLM의 생각(응답)을 대화 기록에 추가합니다.
    return {"messages": [response]}

# 3. 라우터 정의 (판단하는 눈)
# LLM의 응답에 'tool_calls'(도구 호출 요청)가 있는지 확인합니다.
def should_continue(state: MessagesState) -> Literal["tools", "end"]:
    last_message = state["messages"][-1]

    # 도구를 쓰겠다고 했으면 -> 도구 노드로
    if last_message.tool_calls:
        return "tools"

    # 할 말이 끝났으면 -> 종료
    return "end"

# 4. 그래프 빌더 생성
workflow = StateGraph(MessagesState)

# 5. 노드 추가
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node) # prebuilt ToolNode 사용!

# 6. 엣지 연결 (핵심!)
workflow.add_edge(START, "agent")

# (1) agent가 생각한 뒤, 라우터를 통해 길을 정합니다.
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        "end": END
    }
)

# (2) 도구를 쓰고 나면, 반드시 다시 agent에게 보고하러 갑니다. (Loop)
# 그래야 에이전트가 도구 결과를 보고 최종 답변을 할 수 있으니까요.
workflow.add_edge("tools", "agent")

# 7. 컴파일
app = workflow.compile()


# %% 에이전트 실행 결과
# "서울 날씨 알려줘"를 넣으면 책의 설명대로 다음 흐름을 따릅니다.
# START -> agent(search 도구를 쓰기로 하고 tool_calls 생성) -> 라우터 "tools" -> tools(search 실행)
# -> agent(도구 결과를 보고 답변) -> 라우터 "end" -> END
# stream은 노드가 끝날 때마다 이벤트를 내보내므로, 실행된 노드 이름으로 이 생각의 루프를 따라가 볼 수 있습니다.
inputs = {"messages": [("user", "서울 날씨 알려줘")]}

for event in app.stream(inputs):
    for key, value in event.items():
        print(f"Node: {key}")
        # print(value) # 상태 변화 확인용
