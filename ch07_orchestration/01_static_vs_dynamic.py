"""
[7장 1절] [Orchestration / Orchestrator] 지혜로운 담임 선생님: 설계도와 지휘자의 조화

이 절에서 배우는 것:
- 오케스트레이션은 에이전트가 언제, 어떻게, 어떤 정보로 움직일지 정한 협력의 설계도이고,
  오케스트레이터(보통 LLM)는 그 설계도를 들고 실제로 판단하고 지시하는 담임 선생님입니다.
- 6장은 개발자가 흐름을 고정한 정적(결정론적) 오케스트레이션, 7장은 LLM이 다음 단계를 고르는
  동적(추론 기반) 오케스트레이션입니다. 차이는 판단의 주도권을 누가 쥐고 있느냐입니다.
- 그래프 조립 코드로 비교하면 add_edge는 길을 고정하고, add_conditional_edges는 상태 값으로 길을 고릅니다.
- 오케스트레이터는 관찰(의도 파악) -> 계획(적임자 선택) -> 검토(피드백)의 사고 루틴을 거칩니다.

실행 방법 (7장의 모든 명령은 리포지토리 루트에서 실행합니다):
    uv run python ch07_orchestration/01_static_vs_dynamic.py
필요한 환경 변수: 없음 (LLM을 호출하지 않습니다. 7장에서 API 키가 필요한 파일은 2절의 02_coordinator.py뿐입니다)
참고: 엣지 연결 부분만 보여 주는 조각이라, 엣지를 등록할 뿐 컴파일하거나 실행하지 않아 출력이 없습니다.
      노드까지 갖춘 전체 코드는 ch06_multi_agent_school/01_sequential_parallel.py와 이 폴더의 02_coordinator.py에 있습니다.
표시: [보충] 실행에 필요한 코드, [수정] 실행에 맞게 고친 코드, [설명용 코드] 실행되지 않는 설명용 조각
"""

# .env 파일의 API 키를 환경 변수로 불러옵니다.
from dotenv import load_dotenv
load_dotenv()

# [보충] 아래 조각 코드의 그래프 빌더(workflow)를 만드는 데 필요한 import
from typing import TypedDict, Annotated, List
import operator
from langgraph.graph import StateGraph
from langchain_core.messages import BaseMessage


# %% 3. 정적 워크플로우 vs 동적 오케스트레이션 - 코드 한 줄로 보는 정적 vs 동적: 6장 정적 오케스트레이션
# 6장 1절 피자빵 워크플로우의 엣지 연결 부분입니다. 지휘 주체는 개발자이고, 판단은 모델이 아니라 코드가 합니다.
# add_edge(A, B)는 "A가 끝나면 무조건 B로 가라"는 규칙이라, 흐름은 항상 같은 자리(saucer -> topper)로 갑니다.
# [보충] 6장 1절의 PizzaState와 그래프 빌더
class PizzaState(TypedDict):
    status: List[str]
    current_product: str

workflow = StateGraph(PizzaState)

# 6장 — 정적 오케스트레이션
# 개발자가 "누가 끝나면 누구로 간다"를 코드에 직접 박는다.
workflow.add_edge("kneader", "saucer")
workflow.add_edge("saucer", "topper")


# %% 3. 정적 워크플로우 vs 동적 오케스트레이션 - 코드 한 줄로 보는 정적 vs 동적: 7장 동적 오케스트레이션
# 2절에서 만들 축제 부스 워크플로우의 엣지 연결 부분입니다. 지휘 주체는 LLM이고, 개발자는 에이전트 명세만 건넵니다.
# add_conditional_edges: supervisor가 끝난 뒤 함수가 돌려준 값(next_node)으로 다음 노드를 고릅니다.
# 같은 supervisor에서 출발해도 그날의 입력에 따라 supply_expert로 갈지 cook_expert로 갈지가 달라집니다.
# [보충] 2절의 FestivalState와 그래프 빌더
class FestivalState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    next_node: str
    menu: str
    booth_name: str

workflow = StateGraph(FestivalState)

# 7장 — 동적 오케스트레이션
# 다음 노드 이름은 supervisor 노드의 LLM이 매 턴 채워 넣는다.
workflow.add_conditional_edges(
    "supervisor",
    lambda state: state["next_node"], # ← 이 값을 supervisor의 LLM이 결정
    # path_map: 함수가 돌려준 이름 -> 실제로 이동할 노드
    {
        "supply_expert": "supply_expert",
        "cook_expert": "cook_expert",
        "design_expert": "design_expert",
    },
)
