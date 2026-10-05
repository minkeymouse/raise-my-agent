"""
[7장 2절] [Coordinator Pattern] 우리 반의 든든한 사령관

이 절에서 배우는 것:
- 코디네이터 패턴은 조장(오케스트레이터 LLM) 하나가 모든 의사결정을 맡는 중앙 집권형 구조입니다.
  전문가끼리는 직접 통신하지 않고(Star 구조), 모든 요청과 결과가 조장을 거칩니다.
- 조장이 누구를 부를지 고르는 유일한 기준은 프롬프트에 적은 에이전트 명세(이름과 설명)입니다.
- supervisor 노드가 next_node에 노드 이름을 적으면 add_conditional_edges가 그 전문가에게 일을 넘기고,
  전문가는 조장에게 돌아가지 않고 END로 끝납니다(한 번만 위임하는 최소 예시).
- stream으로 노드가 끝날 때마다 바뀐 상태 조각을 받아 어느 전문가가 응답했는지 확인합니다.

실행 방법:
    uv run python ch07_orchestration/02_coordinator.py
필요한 환경 변수: OPENAI_API_KEY (리포지토리 루트의 .env 파일에 넣어 둡니다)
"""

# .env 파일의 API 키를 환경 변수로 불러옵니다.
from dotenv import load_dotenv
load_dotenv()

# %% 3. 실전! 랭그래프로 우리 반 축제 부스 지휘하기 - 공통 import와 모델
# 공통 import와 모델은 한 번만 두고, 상태 -> 조장 -> 전문가 -> 그래프 조립 -> 실행 순으로 읽습니다.
# StateGraph는 그래프를 조립하는 빌더, START와 END는 시작과 끝 표식입니다.
# temperature=0은 조장이 노드 이름 문자열을 고를 때 출력이 덜 흔들리게 하려는 선택이지만,
# 그렇다고 항상 올바른 이름만 나오는 것은 아닙니다.
from typing import TypedDict, Annotated, List
import operator
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

llm = ChatOpenAI(model="gpt-4o", temperature=0)


# %% 상태 FestivalState — 무엇을 공유할지
# 조장과 전문가 노드가 함께 보는 상태입니다. 노드가 돌려준 dict 조각이 이 상태에 합쳐집니다.
# Annotated[..., operator.add]는 "덮어쓰지 말고 리스트를 합쳐라"는 표시입니다. 대화 본문만 다룰 때는
# 5장의 add_messages가 더 흔하지만, 여기서는 형태를 단순히 보이려고 operator.add를 썼습니다.
class FestivalState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    # supervisor가 "다음엔 어느 노드로 갈지" 적어 두는 곳으로, 라우팅 함수가 곧바로 읽습니다.
    next_node: str
    # menu와 booth_name은 전문가 노드가 꺼내 쓰는 부스 설정 값이고, 처음 입력할 때 같이 채워 줍니다.
    menu: str
    booth_name: str


# %% supervisor_node — 조장이 다음 노드 이름만 고르기
# 조장(LLM)은 시스템 프롬프트에 적힌 담당자 명세를 읽고 누구를 부를지 추론합니다.
# 명세가 구체적일수록(무엇을 하는지와 함께 무엇은 하지 않는지까지) 라우팅이 정확해집니다.
# 반환은 {"next_node": ...}뿐이라 messages에는 손대지 않습니다. 조장의 판단은 next_node에만 남습니다.
def supervisor_node(state: FestivalState):
    system_prompt = SystemMessage(content=(
        "당신은 우리 반 축제 부스의 총괄 조장입니다. "
        "사용자의 질문을 분석해 다음 담당자 중 누구를 부를지 결정하세요.\n"
        "- 재료/비용 관련 질문 -> 'supply_expert'\n"
        "- 요리/레시피 관련 질문 -> 'cook_expert'\n"
        "- 홍보/이벤트 관련 질문 -> 'design_expert'\n"
        "다른 말은 하지 말고 딱 해당 담당자의 이름(영어)만 출력하세요."
    ))

    # 마지막 메시지(사용자 질문)만 조장에게 넘깁니다. 긴 대화 전체를 보여 주려면 messages를 통째로 넣습니다.
    last_msg = state["messages"][-1].content
    response = llm.invoke([system_prompt, HumanMessage(content=last_msg)])
    # strip().lower(): 아래 path_map의 키와 글자를 맞추기 위한 최소한의 정규화입니다.
    return {"next_node": response.content.strip().lower()}


# %% 전문가 노드 — 일 처리 후 messages에 한 줄 얹기
# 각 노드는 messages에 새 메시지 하나만 얹어 돌려주고, 리듀서(operator.add)가 기존 대화 뒤에 붙입니다.
# 문장은 예시라 하드코딩했고, 실제로는 이 자리에서 LLM을 부르거나 도구를 호출하게 만들면 됩니다.
# 전문가의 말을 HumanMessage로 넣은 것은 설명을 짧게 하려는 선택이고, 말 주인을 로그에 남기려면
# AIMessage에 name="supply_expert"처럼 붙이는 쪽이 더 자주 쓰입니다.
def supply_node(state: FestivalState):
    print("-> [재료 담당] 전문가가 분석을 시작합니다.")
    return {"messages": [HumanMessage(content=f"{state['menu']}를 위한 떡과 고추장, 어묵 예산을 5만 원으로 책정했습니다!")]}

def cook_node(state: FestivalState):
    print("-> [요리 담당] 전문가가 레시피를 점검합니다.")
    return {"messages": [HumanMessage(content=f"{state['menu']}의 핵심인 '식지 않는 비법 소스'와 화구를 준비했습니다!")]}

def design_node(state: FestivalState):
    print("-> [홍보 담당] 전문가가 마케팅 계획을 세웁니다.")
    return {"messages": [HumanMessage(content=f"'{state['booth_name']}' 부스 홍보를 위해 학교 정문에 대형 포스터를 부착했습니다!")]}


# %% 그래프 조립 — add_conditional_edges가 하는 일
# add_node의 첫 인자는 Studio와 로그에 찍히는 노드 이름이고, START에서 supervisor로 엣지를 두어 항상 조장부터 시작합니다.
# add_conditional_edges: supervisor가 끝난 뒤 라우팅 함수가 돌려준 문자열을 path_map의 키와 맞춰 다음 노드를 고릅니다.
# 세 전문가는 END로만 이어져 있어 한 번 위임하면 거기서 끝납니다. 조장에게 다시 묻게 하려면
# 전문가에서 supervisor로 엣지를 돌리고, 조장 프롬프트에 "끝내도 되면 특정 이름을 내라" 같은 종료 규칙을 더합니다.
workflow = StateGraph(FestivalState)
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("supply_expert", supply_node)
workflow.add_node("cook_expert", cook_node)
workflow.add_node("design_expert", design_node)

workflow.add_edge(START, "supervisor")

workflow.add_conditional_edges(
    "supervisor",
    # 라우팅 함수: 상태에 적어 둔 next_node를 그대로 읽습니다.
    lambda x: x["next_node"],
    # path_map: 라우팅 함수가 돌려준 이름 -> 실제로 이동할 노드
    {
        "supply_expert": "supply_expert",
        "cook_expert": "cook_expert",
        "design_expert": "design_expert",
    },
)

workflow.add_edge("supply_expert", END)
workflow.add_edge("cook_expert", END)
workflow.add_edge("design_expert", END)

app = workflow.compile()


# %% 조장이 엉뚱한 이름을 내면? — 라우팅 실패 막기
# supervisor가 path_map에 없는 이름(예: 오타 "supplyy_expert"나 "잘 모르겠음")을 내면 KeyError로 그 자리에서 멈춥니다.
# 운영 코드라면 라우터 함수에 짧은 가드를 두고, path_map에 "fallback": "fallback_node"를 더해
# fallback_node가 조장에게 다시 묻거나 사과 메시지를 남기고 END로 가게 합니다.
# 책의 안내대로 이 가드는 정의만 하고 본문 그래프(app)에는 연결하지 않습니다.
ALLOWED = {"supply_expert", "cook_expert", "design_expert"}

def route_from_supervisor(state: FestivalState) -> str:
    name = state["next_node"]
    return name if name in ALLOWED else "fallback"  # 허용 외엔 fallback


# %% stream으로 노드별 업데이트 보기 - 테스트 1: 홍보 문의
# invoke는 그래프가 끝난 뒤 최종 상태를 한 번에 돌려주고, stream은 노드 하나가 끝날 때마다
# {노드이름: 이번에 바뀐 상태 조각}을 흘려줍니다. 이벤트의 키가 곧 노드 이름이라 어느 단계의 출력인지 바로 보입니다.
# supervisor는 next_node만 돌려주므로 이 루프에서 출력이 없고, design_expert가 messages를 얹을 때 한 줄이 찍힙니다.
# 초기 입력의 "next_node": ""는 TypedDict 키를 채워 두면서 "아직 라우팅 전"이라는 뜻을 드러냅니다.
print("\n[테스트 1: 홍보 문의]")
input_1 = {
    "messages": [HumanMessage(content="우리 부스 이름 알리려면 뭐부터 해야 돼?")],
    "menu": "치즈 떡볶이",
    "booth_name": "치즈가 쭈욱-",
    "next_node": "",
}
for event in app.stream(input_1):
    for node, value in event.items():
        if "messages" in value:
            print(f"[{node}] 마지막 메시지: {value['messages'][-1].content}")

# [책의 실행 결과]
# [테스트 1: 홍보 문의]
# [design_expert] 마지막 메시지: '치즈가 쭈욱-' 부스 홍보를 위해 학교 정문에 대형 포스터를 부착했습니다!
#
# 참고: 실제로 실행하면 design_node의 print 문 줄("-> [홍보 담당] ...")도 마지막 메시지 앞에 함께 찍힙니다.


# %% stream으로 노드별 업데이트 보기 - 테스트 2: 재료 문의
# 이번에는 재료 문의를 넣어 supervisor가 supply_expert로 라우팅하는지 확인합니다.
# 조장이 고른 next_node까지 함께 보고 싶다면 루프에 if "next_node" in value: 분기를 하나 더하면 됩니다.
print("\n" + "-" * 30)

print("[테스트 2: 재료 문의]")
input_2 = {
    "messages": [HumanMessage(content="떡볶이 재료 사는 데 돈 얼마나 들어?")],
    "menu": "치즈 떡볶이",
    "booth_name": "치즈가 쭈욱-",
    "next_node": "",
}
for event in app.stream(input_2):
    for node, value in event.items():
        if "messages" in value:
            print(f"[{node}] 마지막 메시지: {value['messages'][-1].content}")

# [책의 실행 결과]
#
# ------------------------------
# [테스트 2: 재료 문의]
# [supply_expert] 마지막 메시지: 치즈 떡볶이를 위한 떡과 고추장, 어묵 예산을 5만 원으로 책정했습니다!
#
# 참고: 테스트 1과 마찬가지로 "-> [재료 담당] 전문가가 분석을 시작합니다." 줄도 함께 찍힙니다.
