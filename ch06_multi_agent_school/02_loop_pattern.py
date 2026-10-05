"""
[6장 2절] [Loop pattern] 공부는 될 때까지!

이 절에서 배우는 것:
- 루프 패턴(나머지 공부): 학생 에이전트가 풀고 채점자 에이전트가 PASS/FAIL을 판정해, 합격할 때까지 반복합니다.
- 조건부 엣지(add_conditional_edges)로 다음 노드를 고르고, '최대 3번'과 같은 탈출 조건을 반드시 둡니다.
- 리뷰 및 반복 개선 패턴(글짓기 수업): 작성자는 글만 쓰고, 비평가는 구체적인 피드백만 주도록 역할을 나눕니다.
- 두 패턴은 그래프 구조가 같고, 에이전트의 프롬프트 설계와 피드백의 구체성이 다릅니다.

실행 방법:
    uv run python ch06_multi_agent_school/02_loop_pattern.py
필요한 환경 변수: OPENAI_API_KEY (모델: gpt-4o)
"""

# .env 파일의 API 키를 환경 변수로 불러옵니다.
from dotenv import load_dotenv
load_dotenv()

# %% 1. 나머지 공부: 루프 패턴 (Loop Pattern) - LangGraph로 나머지 공부 구현: 모듈과 상태, 모델 준비
# 루프 패턴은 특정 조건(시험 합격)을 만족할 때까지 작업(Action) -> 확인(Check) -> 반복(Loop)을 되풀이합니다.
# 학생과 채점자가 주고받는 메시지가 계속 쌓이므로 messages에는 리듀서(operator.add)를 붙입니다.
# attempt는 반복 횟수를 세어, 탈출 조건(최대 3번)을 판단하는 데 씁니다.
from typing import TypedDict, Annotated, List
import operator
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage, HumanMessage

# 1. 상태 정의 (대화 기록과 시도 횟수 저장)
class StudyState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    attempt: int

# 2. 모델 준비
llm = ChatOpenAI(model="gpt-4o", temperature=0.7)


# %% 1. 루프 패턴 - 학생 에이전트와 채점자 에이전트
# 종료 조건을 판단하는 별도의 서브 에이전트(채점자)를 둘 수 있다는 점이, 미들웨어나 로직 기반 루프와 가장 큰 차이입니다.
# 채점자가 'FAIL'로 돌려보내면 학생 에이전트가 문제를 다시 풉니다. 채점자 답변 속의 'PASS'가 아래 check_pass의 합격 신호입니다.
# 3. 학생 에이전트 생성 (문제를 푸는 역할)
student_agent = create_agent(
    model=llm,
    tools=[],
    system_prompt="당신은 수학 공부를 하는 학생입니다. 주어진 문제에 대해 풀이를 작성하세요."
)

# 4. 채점자 에이전트 생성 (합불 여부를 판단하는 역할)
grader_agent = create_agent(
    model=llm,
    tools=[],
    system_prompt=(
        "당신은 엄격한 수학 선생님입니다. 학생의 답을 보고 정답 여부를 판단하세요. "
        "정답이면 'PASS', 틀렸다면 'FAIL'과 함께 틀린 이유를 짧게 설명하세요."
    )
)


# %% 1. 루프 패턴 - 노드와 조건부 엣지 로직
# 두 노드는 지금까지의 대화 전체를 에이전트에게 넘기고, 에이전트의 마지막 메시지(새 답변) 하나만 반환합니다.
# check_pass는 조건부 엣지에서 쓸 함수입니다. 돌려준 문자열("pass", "give_up", "retry")로 다음에 갈 곳이 정해집니다.
# 합격 여부는 채점자 에이전트가, 시도 횟수 제한은 규칙이 판단합니다.
# 실제 시스템에서도 이처럼 규칙 기반 루프와 에이전트 기반 루프를 섞어 쓰는 것이 좋습니다.
# 5. 노드 함수 정의
def student_node(state: StudyState):
    # 학생이 마지막 메시지(문제 또는 피드백)를 보고 답변 생성
    response = student_agent.invoke({"messages": state["messages"]})
    # 새 답변은 messages 리듀서가 기존 대화 뒤에 이어 붙이고, 학생이 답할 때마다 attempt가 1 올라갑니다.
    return {
        "messages": [response["messages"][-1]],
        "attempt": state.get("attempt", 0) + 1
    }

def grader_node(state: StudyState):
    # 선생님이 채점
    response = grader_agent.invoke({"messages": state["messages"]})
    return {"messages": [response["messages"][-1]]}

# 6. 조건부 엣지 로직
def check_pass(state: StudyState):
    # 가장 최근 메시지, 곧 채점자의 판정을 봅니다.
    last_msg = state["messages"][-1].content
    if "PASS" in last_msg:
        return "pass"
    elif state["attempt"] >= 3:  # 최대 3번까지만 재시험
        return "give_up"
    else:
        return "retry"


# %% 1. 루프 패턴 - 그래프 조립
# START -> student -> grader는 일반 엣지로 잇고, grader 다음은 add_conditional_edges로 정합니다.
# 세 번째 인자의 딕셔너리가 check_pass의 반환값을 실제 노드에 연결합니다. "retry"면 student로 돌아가 루프가 생깁니다.
# LangGraph는 컴파일할 때 루프의 종료 조건이 적절한지 검증하지 않으므로, 탈출 조건은 설계자가 챙겨야 합니다.
# 7. 그래프 조립
workflow = StateGraph(StudyState)
workflow.add_node("student", student_node)
workflow.add_node("grader", grader_node)

workflow.add_edge(START, "student")
workflow.add_edge("student", "grader")

workflow.add_conditional_edges(
    "grader",
    check_pass,
    {
        "pass": END, "give_up": END, "retry": "student"
    }
)

app = workflow.compile()


# %% 1. 루프 패턴 - 실행
# 처음 입력으로 문제(HumanMessage)와 attempt=0을 넣습니다.
# 결과의 [ai] 메시지들은 한 에이전트가 아니라 학생과 채점자가 번갈아 남긴 것입니다.
# 이 정도 문제는 대개 한 번에 맞히므로, 실전에서는 채점자에게만 정답이나 상세한 채점 기준을 알려 주기도 합니다.
# 8. 실행
print("=== 나머지 공부 시작 (문제: 12 * 12는?) ===")
inputs = {"messages": [HumanMessage(content="문제: 12 곱하기 12는 얼마인가요?")], "attempt": 0}
result = app.invoke(inputs)

for msg in result["messages"]:
    print(f"[{msg.type}]: {msg.content}")

# [책의 실행 결과]
# === 나머지 공부 시작 (문제: 12 * 12는?) ===
# [human]: 문제: 12 곱하기 12는 얼마인가요?
# [ai]: 12 곱하기 12는 140입니다.
# [ai]: FAIL. 틀렸습니다. 12 곱하기 12는 144입니다. 다시 계산해보세요.
# [ai]: 아, 실수를 했네요. 12 곱하기 12는 144입니다.
# [ai]: PASS. 정답입니다.


# %% 2. 넌 학생이고 난 선생이야: 리뷰 및 반복 개선 패턴 - LangGraph로 글짓기 수업 구현: 모듈과 상태, 모델 준비
# 위 결과처럼 루프 패턴의 채점자는 정답을 직접 알려 주기도 해서, 학생과 채점자의 역할 분리가 완벽하지 않았습니다.
# 이번에는 작성자(Generator)는 오직 글을 쓰고, 비평가(Critic)는 "어디가, 왜 부족한지"만 구체적으로 알려 주도록 나눕니다.
# 필요한 모듈과 StudyState, llm을 다시 정의합니다(1번과 같은 내용입니다).
from typing import TypedDict, Annotated, List
import operator
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage, HumanMessage

# (1번에서 정의한 State 재사용)
class StudyState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    attempt: int

# 모델 준비
llm = ChatOpenAI(model="gpt-4o", temperature=0.7)


# %% 2. 리뷰 및 반복 개선 패턴 - 작성자와 비평가 에이전트
# 핵심은 프롬프트에서 역할의 경계를 정해 주는 것입니다. 비평가의 "절대로 직접 글을 고쳐주지 마세요"가 역할 분리를 보장합니다.
# 비평가가 '완벽합니다'라고 말하는 것이 이 루프의 종료 신호가 됩니다.
# 여기서는 프롬프트만 썼지만, 미들웨어, 스키마, 컨텍스트로 두 역할을 더 세밀하게 정의하는 것이 일반적입니다.
# 1. 작성자(Writer) 에이전트 생성
writer_agent = create_agent(
    model=llm,
    tools=[],
    system_prompt=(
        "당신은 작가입니다. 요청받은 주제로 짧은 글을 작성하세요. "
        "만약 비평가의 피드백이 있다면, 그 내용을 반영하여 글을 수정하세요. "
        "절대로 자기 자신의 글을 평가하지 마세요. 오직 글을 쓰는 것만이 당신의 역할입니다."
    )
)

# 2. 비평가(Critic) 에이전트 생성
critic_agent = create_agent(
    model=llm,
    tools=[],
    system_prompt=(
        "당신은 깐깐한 편집자입니다. 작가의 글을 읽고 논리적 허점, 문체, 감정 표현 등에 대해 비평하세요. "
        "절대로 직접 글을 고쳐주지 마세요. 어디가 부족한지, 왜 고쳐야 하는지만 설명하세요. "
        "글이 충분히 좋다고 판단되면 반드시 '완벽합니다'라고 말하세요."
    )
)


# %% 2. 리뷰 및 반복 개선 패턴 - 노드와 종료 조건
# 작성자가 글을 쓸 때마다(= 수정할 때마다) attempt가 1씩 올라가므로 '최대 3회 수정'이라는 조건이 자연스럽게 구현됩니다.
# should_continue는 비평가가 만족했거나 횟수를 채우면 "publish", 아니면 "revise"를 돌려줍니다.
# 반복할수록 품질은 좋아지지만 비용(Cost)과 시간(Latency)도 늘어나므로, 알맞은 종료 조건(타협점)을 두어야 합니다.
# 3. 노드 정의
def writer_node(state: StudyState):
    # 작가는 대화 기록(피드백 포함)을 바탕으로 글을 씀/수정함
    response = writer_agent.invoke({"messages": state["messages"]})
    return {
        "messages": [response["messages"][-1]],
        "attempt": state.get("attempt", 0) + 1
    }

def critic_node(state: StudyState):
    # 비평가는 작가의 글을 보고 피드백을 줌
    response = critic_agent.invoke({"messages": state["messages"]})
    return {"messages": [response["messages"][-1]]}

# 4. 종료 조건 로직
def should_continue(state: StudyState):
    last_msg = state["messages"][-1].content

    if "완벽합니다" in last_msg:
        return "publish"           # 비평가가 만족 → 출판
    elif state.get("attempt", 0) >= 3:  # 최대 3회 수정
        return "publish"           # 횟수 초과 → 현재 버전으로 출판
    else:
        return "revise"            # 피드백 반영하여 수정 요청


# %% 2. 리뷰 및 반복 개선 패턴 - 그래프 조립
# 그래프 구조는 1번 루프 패턴과 같습니다. critic 다음을 조건부 엣지로 정해, "revise"면 writer로 돌아갑니다.
# 5. 그래프 조립
review_workflow = StateGraph(StudyState)

review_workflow.add_node("writer", writer_node)
review_workflow.add_node("critic", critic_node)

# 시작 → 작가 → 비평가
review_workflow.add_edge(START, "writer")
review_workflow.add_edge("writer", "critic")

# 비평가 → (조건) → 끝 또는 다시 작가
review_workflow.add_conditional_edges(
    "critic",
    should_continue,
    {
        "publish": END,       # 출판!
        "revise": "writer"    # 수정 요청 (Loop)
    }
)

review_app = review_workflow.compile()


# %% 2. 리뷰 및 반복 개선 패턴 - 실행
# 처음 메시지와 함께 attempt를 입력으로 줍니다. 이처럼 실행할 때 상태 스키마에 맞는 값을 함께 넘길 수 있습니다.
# 출력할 때는 메시지의 순서로 화자를 구분합니다. 0번은 주제이고, 그다음부터 작가와 비평가가 번갈아 나옵니다.
# 6. 실행
print("\n=== 글짓기 수업 시작 (주제: 가을 아침) ===")
inputs = {
    "messages": [HumanMessage(content="주제: '가을 아침'에 대한 짧은 시를 써주세요.")],
    "attempt": 0
}
result = review_app.invoke(inputs)

# 실행 결과 출력 (가독성을 위해 화자 구분)
for i, msg in enumerate(result["messages"]):
    if msg.type == "human":
        role = " 주제"
    elif i % 2 == 1:
        role = "작가"
    else:
        role = "비평가"

    print(f"\n[{role}]: {msg.content}")

# [책의 실행 결과]
# === 글짓기 수업 시작 (주제: 가을 아침) ===
#
# [주제]: 주제: '가을 아침'에 대한 짧은 시를 써주세요.
#
# [작가]:
# 창가에 서린 찬 이슬
# 붉게 물든 잎새 하나
# 바람에 실려 춤을 추네
# 아, 가을이 왔구나.
#
# [비평가]:
# 전반적으로 가을의 정취를 잘 표현했지만, 마지막 연의 "아, 가을이 왔구나"는 너무 직설적이고 평범합니다.
# 독자가 가을을 느낄 수 있도록 조금 더 감각적인 이미지나 은유를 사용하여 마무리를 수정해 보는 것이 좋겠습니다.
#
# [작가]:
# 창가에 서린 찬 이슬
# 붉게 물든 잎새 하나
# 바람에 실려 춤을 추네
# 하늘은 높고,
# 내 마음 한구석에
# 조용히 낙엽이 쌓인다.
#
# [비평가]:
# 수정된 마무리가 훨씬 좋습니다. '마음 한구석에 조용히 낙엽이 쌓인다'는 표현이 여운을 남기며
# 가을의 고독함을 잘 전달합니다.
# 완벽합니다.

# 참고: 비평가는 고칠 방향만 제시하고, 피드백을 어떻게 반영할지는 작가가 정합니다.
# 루프 패턴과 리뷰 및 반복 개선 패턴의 관계는 다음과 같습니다.
# [루프 패턴]         →  Action → Check(PASS/FAIL) → Loop
# [리뷰 및 반복 개선]  →  Generate → Critique(구체적 피드백) → Refine → Critique → ...
# 정답이 명확한 작업(수학 문제, 데이터 검증)은 루프 패턴으로 충분하고, 품질 향상이 필요한 작업(글쓰기, 코드 리뷰)에는 리뷰 및 반복 개선 패턴을 씁니다.
