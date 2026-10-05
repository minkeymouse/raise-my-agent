"""
[6장 1절] [Sequential & Parallel pattern] 재밌는 가정 시간

이 절에서 배우는 것:
- 서브 에이전트는 create_agent로 만들고, 그래프의 각 노드 안에서 invoke하는 것이 멀티 에이전트의 기초입니다.
- 순차 패턴(피자빵 만들기): 앞 에이전트의 결과가 다음 에이전트의 입력이 되도록 edge(A, B), edge(B, C)로 잇습니다.
- 병렬 패턴(조별 과제): START에서 여러 노드로 엣지를 뻗어(Fan-out) 서로 독립적인 일을 동시에 처리합니다.
- 상태의 어떤 값은 덮어쓰고, 어떤 값은 리듀서로 모을지 구분합니다.

실행 방법 (6장의 모든 명령은 리포지토리 루트에서 실행합니다):
    uv run python ch06_multi_agent_school/01_sequential_parallel.py
필요한 환경 변수: OPENAI_API_KEY (모델: gpt-4o-mini)
    6장의 모든 예제가 ChatOpenAI를 쓰므로, 키를 리포지토리 루트의 .env 파일에 적어 둡니다.
표시: [보충] 실행에 필요한 코드, [수정] 실행에 맞게 고친 코드, [설명용 코드] 실행되지 않는 설명용 조각
"""

# .env 파일의 API 키를 환경 변수로 불러옵니다.
from dotenv import load_dotenv
load_dotenv()

# %% 1. 차근차근 이어달리기: 순차 패턴 - 피자빵 만들기 구현: 상태와 전문가 에이전트 생성 함수
# 순차 패턴은 한 에이전트의 결과(Output)가 다음 에이전트의 입력(Input)으로 그대로 전달되는 방식입니다.
# PizzaState에는 각 단계의 작업 보고(status, 리스트)와 지금의 상품(current_product, 텍스트)을 담습니다.
# create_specialist는 create_agent를 감싸, 이름과 역할만 바꿔 전문가 에이전트를 만들어 주는 함수입니다.
# 서브 에이전트까지 LangGraph로 짜면 복잡하고 유지보수가 어려워지기 쉬우므로, 서브 에이전트는 create_agent로 만듭니다.
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List

# 1. 상태 정의: 피자의 현재 상태를 기록합니다.
class PizzaState(TypedDict):
    status: List[str]
    current_product: str

# 2. 전문가 에이전트 생성 함수
def create_specialist(name: str, role: str):
    return create_agent(
        model=ChatOpenAI(model="gpt-4o-mini"),
        tools=[], # 도구 없이 LLM 자체 능력만 사용
        # 프롬프트는 name과 role로 역할에 맞게 동적으로 구성합니다. 예시가 간단해 미들웨어는 쓰지 않았습니다.
        system_prompt=f"당신은 {name}입니다. 당신의 역할은 {role}입니다. 작업을 수행하고 결과를 한 문장으로 보고하세요."
    )


# %% 1. 순차 패턴 - 반죽 노드
# 노드 안에서 전문가 에이전트를 만들고 invoke로 실행합니다. 멀티 에이전트를 만드는 기초적인 방식입니다.
# 노드가 끝나면 에이전트의 답변과 바뀐 상태를 반환해야 다음 노드가 이어서 일할 수 있습니다.
# 3. 각 단계별 에이전트 노드 정의
def knead_node(state: PizzaState):
    kneader = create_specialist("반죽 담당", "밀가루와 재료를 섞어 탄력있는 도우를 만드는 것")
    response = kneader.invoke({"messages": [{"role": "user", "content": f"현재 상태: {state.get('current_product', '밀가루')}. 반죽을 만들어 주세요."}]})
    # response["messages"][-1]은 에이전트의 최종 답변입니다. status에는 리듀서가 없어 반환한 값으로 덮어쓰므로,
    # 기존 목록에 새 보고를 붙인 전체 목록을 돌려줍니다. current_product도 다음 단계의 상품으로 덮어씁니다.
    return {"status": state["status"] + [response["messages"][-1].content], "current_product": "반죽된 도우"}


# %% 1. 순차 패턴 - 소스 노드와 토핑 노드
# 반죽 노드와 같은 방식입니다. 앞 노드가 남긴 current_product를 받아 새 전문가에게 맡기고, 그 결과로 상태를 갱신합니다.
# 순차 패턴에서는 상태를 일관되게 전달하고, 필요한 정보가 추가되거나 덮어써지도록 관리하는 것이 중요합니다.
def sauce_node(state: PizzaState):
    saucer = create_specialist("소스 전문가", "빵에 토마토 소스를 꼼꼼히 바르는 것")
    response = saucer.invoke({"messages": [{"role": "user", "content": f"현재 상태: {state['current_product']}. 소스를 발라주세요."}]})
    return {"status": state["status"] + [response["messages"][-1].content], "current_product": "소스 발린 빵"}

def topping_node(state: PizzaState):
    topper = create_specialist("토핑 전문가", "햄과 치즈를 듬뿍 올리는 것")
    response = topper.invoke({"messages": [{"role": "user", "content": f"현재 상태: {state['current_product']}. 토핑을 올려주세요."}]})
    return {"status": state["status"] + [response["messages"][-1].content], "current_product": "토핑된 피자"}


# %% 1. 순차 패턴 - 그래프 조립
# add_node로 노드를 등록하고, add_edge로 START -> kneader -> saucer -> topper -> END를 한 줄로 잇습니다.
# 엣지가 한 줄로 이어져 있으므로 앞 단계가 끝나야 다음 단계가 실행됩니다.
# 4. 그래프 조립 (순서대로 연결!)
workflow = StateGraph(PizzaState)

workflow.add_node("kneader", knead_node)
workflow.add_node("saucer", sauce_node)
workflow.add_node("topper", topping_node)

# 순차 연결: START -> Baker -> Saucer -> Topper -> END
workflow.add_edge(START, "kneader")
workflow.add_edge("kneader", "saucer")
workflow.add_edge("saucer", "topper")
workflow.add_edge("topper", END)

app = workflow.compile()


# %% 1. 순차 패턴 - 실행
# 컴파일한 그래프도 invoke로 실행하지만, 개별 에이전트와 달리 입력은 PizzaState 모양의 딕셔너리입니다.
# 순차 패턴은 어느 단계에서 문제가 생겼는지 지켜보기 쉽고 에이전트가 권한 밖의 일을 하지 못하게 막기 좋습니다.
# 대신 한 단계가 끝나야 다음 단계가 실행되므로 효율이 떨어집니다. 이를 보완하는 것이 병렬 패턴입니다.
# 5. 실행
result = app.invoke({"status": [], "current_product": "밀가루 반죽"})
print("=== 피자 만들기 과정 ===")
for step in result["status"]:
    print(step)

# [책의 실행 결과]
# === 피자 만들기 과정 ===
# 제빵사: 밀가루 반죽을 오븐에 넣어 노릇노릇하고 바삭하게 구워냈습니다.
# 소스 전문가: 구운 빵 위에 신선한 토마토 소스를 골고루 펴 발랐습니다.
# 토핑 전문가: 소스가 발린 빵 위에 햄과 모짜렐라 치즈를 풍성하게 올렸습니다.


# %% 2. 다 같이 동시에!: 병렬 패턴 - 조별 과제 구현: 상태와 서브 에이전트 정의
# 병렬 패턴은 여러 에이전트가 각자의 일을 동시에, 독립적으로 수행하는 방식입니다.
# 서브 에이전트끼리 간섭하면 안 되므로 역사 담당과 과학 담당은 각자 맡은 내용만 조사합니다.
# 흩어진 결과를 취합(Aggregation)하는 것은 조장의 몫이며, 여기서는 두 결과를 sections 하나에 모읍니다.
# [수정] 아래 ReportState의 리듀서(operator.add)에 필요한 import입니다.
import operator
from typing import Annotated

# 1. 상태 정의
class ReportState(TypedDict):
    topic: str
    # [수정] 두 노드가 같은 단계에서 sections를 갱신하므로, 리스트를 이어 붙이는 리듀서(operator.add)가 필요합니다. (책: 아래 주석 줄)
    # sections: List[str]
    sections: Annotated[List[str], operator.add]

# 2. 각 서브 에이전트 정의
history_agent = create_agent(
    model=ChatOpenAI(model="gpt-4o-mini"),
    tools=[],
    system_prompt="당신은 역사학자입니다. 주어진 주제의 역사적 배경을 한 줄로 요약하세요."
)

science_agent = create_agent(
    model=ChatOpenAI(model="gpt-4o-mini"),
    tools=[],
    system_prompt="당신은 과학자입니다. 주어진 주제의 과학적 원리를 한 줄로 설명하세요."
)


# %% 2. 병렬 패턴 - 노드 정의
# 한 노드(START)에서 뻗어 나가므로 두 노드는 같은 ReportState에서 topic을 읽어 각자의 에이전트에게 맡깁니다.
# 각 노드는 자기 결과 한 줄만 리스트에 담아 반환하고, 이를 하나로 모으는 일은 sections의 리듀서가 합니다.
# 3. 노드 정의 (에이전트 호출)
def history_node(state: ReportState):
    response = history_agent.invoke({"messages": [{"role": "user", "content": state["topic"]}]})
    return {"sections": [f"[역사] {response['messages'][-1].content}"]}

def science_node(state: ReportState):
    response = science_agent.invoke({"messages": [{"role": "user", "content": state["topic"]}]})
    return {"sections": [f"[과학] {response['messages'][-1].content}"]}


# %% 2. 병렬 패턴 - 그래프 조립과 실행
# START에서 두 노드로 엣지를 각각 연결하는 것을 Fan-out이라고 합니다. historian과 scientist가 같은 단계에서 함께 실행됩니다.
# 어느 에이전트가 먼저 끝날지 모르므로 같은 키에 쓰는 값을 합치는 리듀서를 알맞게 정의해야 합니다.
# 여러 에이전트를 동시에 호출하므로 빠른 대신 토큰 사용량과 비용이 늘 수 있고, 취합 과정을 세심하게 제어해야 합니다.
# 4. 그래프 조립 (동시 실행!)
workflow = StateGraph(ReportState)

workflow.add_node("historian", history_node)
workflow.add_node("scientist", science_node)

# START에서 두 갈래로 나뉩니다 (Fan-out)
workflow.add_edge(START, "historian")
workflow.add_edge(START, "scientist")

# 결과를 하나로 모으는 노드 (Fan-in) - 여기서는 간단히 END로 연결
workflow.add_edge("historian", END)
workflow.add_edge("scientist", END)

app = workflow.compile()

# 5. 실행
result = app.invoke({"topic": "전기 자동차", "sections": []})
print(f"=== '{result['topic']}' 보고서 ===")
for section in result["sections"]:
    print(section)

# [책의 실행 결과] (순서는 바뀔 수 있음)
# === '전기 자동차' 보고서 ===
# [과학] 전기 에너지를 배터리에 저장하고 모터를 회전시켜 구동하는 원리입니다.
# [역사] 19세기 초반에 처음 발명되었으나 내연기관에 밀렸다가 최근 친환경 이슈로 다시 부상했습니다.
