"""
[7장 3절] [Hierarchical Pattern] 권한을 위임하는 부장 선생님

이 절에서 배우는 것:
- 계층적 패턴은 상위가 큰 계획을 세워 중간 관리자에게 위임하고, 중간 관리자가 자기 팀을 다시 조율하는 트리(Tree) 구조입니다.
- 코디네이터(Star, 1단)와 달리 '위임의 위임'이 가능해, 각 단의 LLM은 자기 바로 아래만 들여다봅니다.
- 계획(Plan) -> 위임(Delegate) -> 취합(Aggregate) 순서로 'AI 윤리' 보고서를 두 섹션(Intro, Conclusion)으로 나눠 만듭니다.
- 컴파일된 서브그래프(섹션 팀)를 wrapper로 감싸 상위 그래프의 노드로 넣고, fan-out / fan-in으로 병렬 위임과 취합을 합니다.
- 전문가가 5~10명을 넘는 등 규모가 커질 때 비로소 이점이 드러나고, 전문가가 셋, 넷이면 코디네이터가 유지하기 쉽습니다.

실행 방법:
    uv run python ch07_orchestration/03_hierarchical.py
필요한 환경 변수: 없음 (LLM을 호출하지 않습니다)
"""

# .env 파일의 API 키를 환경 변수로 불러옵니다.
from dotenv import load_dotenv
load_dotenv()

# %% 4. 실전! 보고서 생성 에이전트 만들기 - 상태 정의 — 상위와 하위가 무엇을 공유할까
# 상위 그래프(ReportState)와 하위 그래프인 섹션 팀(SectionState)이 상태를 따로 가집니다.
# drafts의 Annotated[List[str], operator.add]가 핵심입니다. 두 섹션 팀이 동시에 끝나도 결과가
# 덮어써지지 않고 누적되게 하는 리듀서로, 5장에서 본 add_messages와 같은 종류의 합치기 규칙입니다.
from typing import TypedDict, Annotated, List
import operator
from langgraph.graph import StateGraph, START, END

# 상위 그래프의 상태 — 사용자 요청과 모든 팀의 결과 모음
class ReportState(TypedDict):
    topic: str
    drafts: Annotated[List[str], operator.add]   # 각 팀의 초안이 누적

# 하위 그래프(섹션 팀)의 상태 — 자기 섹션 정보만 안다
class SectionState(TypedDict):
    topic: str
    section_name: str
    research: str
    drafts: Annotated[List[str], operator.add]


# %% 섹션 팀 내부 노드 — 각 팀은 자기 영역만 안다
# research -> write 두 노드가 섹션 팀 하나를 이룹니다. 각 노드는 자기 입력(section_name, topic, research)만 보고
# 결과를 돌려주며, 다른 섹션 팀이 무엇을 하는지는 모릅니다. 이 분리가 계층적 패턴의 정수입니다.
def research_node(state: SectionState):
    # 실전에서는 검색 도구 호출. 여기선 짧게.
    return {"research": f"[{state['section_name']}] {state['topic']} 관련 자료"}

def write_node(state: SectionState):
    text = f"[{state['section_name']}] {state['research']}를 바탕으로 작성한 초안"
    # 리스트로 돌려주면 drafts 리듀서(operator.add)가 기존 초안 뒤에 이어 붙입니다.
    return {"drafts": [text]}


# %% 섹션 팀 서브그래프 빌더
# 서브그래프: LangGraph에서는 컴파일된 그래프를 다른 그래프의 노드처럼 끼워 넣을 수 있습니다.
# build_section_team은 컴파일된 작은 그래프와, 상위 상태와 매핑하는 wrapper(run)를 한 번에 묶어 돌려줍니다.
# 같은 빌더로 Intro 팀, Conclusion 팀처럼 여러 인스턴스를 찍어 내므로, 같은 구조가 트리의 여러 가지에 복제됩니다.
def build_section_team(section_name: str):
    sub = StateGraph(SectionState)
    sub.add_node("research", research_node)
    sub.add_node("write", write_node)
    sub.add_edge(START, "research")
    sub.add_edge("research", "write")
    sub.add_edge("write", END)
    compiled = sub.compile()

    # 상위(ReportState) ↔ 하위(SectionState) 상태 매핑
    # 상위·하위 상태의 키가 다르면 LangGraph가 자동으로 매핑해 주지 않으므로, 상위 상태에 없는 키
    # (section_name, research)는 wrapper에서 명시적으로 채워 넣고, 결과는 drafts만 위로 올립니다.
    def run(state: ReportState):
        result = compiled.invoke({
            "topic": state["topic"],
            "section_name": section_name,
            "research": "",
            "drafts": [],
        })
        return {"drafts": result["drafts"]}
    return run


# %% 상위 그래프 조립 — Plan → 두 팀 fan-out → Aggregate
# add_node("intro_team", build_section_team("Intro"))에는 보통의 함수 대신 컴파일된 서브그래프를 감싼 함수가 들어갑니다.
# 상위 그래프에서는 한 노드가 일을 처리하는 것처럼 보이지만, 그 안에서 또 다른 작은 그래프가 돕니다(위임의 위임).
# planner 다음에 엣지가 둘로 갈라지면(fan-out) 두 팀이 병렬로 시작하고, 두 팀이 모두 끝난 다음에야
# aggregator가 호출됩니다(fan-in). 두 팀의 결과는 drafts 리듀서가 자동으로 누적합니다.

# 계획(Plan): 상위는 전체를 어떻게 쪼갤지만 정하고, 구체적인 실행은 섹션 팀에 맡깁니다(예제는 분해 계획을 출력만 합니다).
def plan_node(state: ReportState):
    print(f"[Plan] '{state['topic']}'을 Intro / Conclusion 두 섹션으로 분해")
    return {}

# 취합(Aggregate): 두 팀이 올린 결과가 drafts에 모인 뒤 호출됩니다(예제는 모인 섹션 개수만 출력합니다).
def aggregate_node(state: ReportState):
    print(f"[Aggregate] 섹션 {len(state['drafts'])}개 통합 → 최종 보고서")
    return {}

main = StateGraph(ReportState)
main.add_node("planner", plan_node)
main.add_node("intro_team", build_section_team("Intro"))
main.add_node("conclusion_team", build_section_team("Conclusion"))
main.add_node("aggregator", aggregate_node)

main.add_edge(START, "planner")
main.add_edge("planner", "intro_team")
main.add_edge("planner", "conclusion_team")     # fan-out (병렬 위임)
main.add_edge("intro_team", "aggregator")
main.add_edge("conclusion_team", "aggregator")  # fan-in (취합)
main.add_edge("aggregator", END)

app = main.compile()


# %% 실행
# Plan과 Aggregate는 한 번씩 찍히고, 그 사이에 두 섹션 팀이 동시에 실행됩니다.
# 섹션을 늘리려면 main.add_node("body_team", build_section_team("Body")) 한 줄과 엣지 두 줄만 더하면 됩니다.
result = app.invoke({"topic": "AI 윤리", "drafts": []})
for d in result["drafts"]:
    print("- ", d)

# [책의 실행 결과]
# [Plan] 'AI 윤리'을 Intro / Conclusion 두 섹션으로 분해
# [Aggregate] 섹션 2개 통합 → 최종 보고서
# -  [Intro] [Intro] AI 윤리 관련 자료를 바탕으로 작성한 초안
# -  [Conclusion] [Conclusion] AI 윤리 관련 자료를 바탕으로 작성한 초안
#
# 참고: 같은 단계에서 병렬로 실행된 노드의 결과는 노드 이름의 알파벳 순서로 합쳐지므로,
#       실제로는 conclusion_team의 초안이 intro_team의 초안보다 먼저 출력됩니다.
