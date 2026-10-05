"""
[6장 3절] [Swarm pattern] 왁자지껄 쉬는시간

이 절에서 배우는 것:
- 스웜 패턴: 중앙 관리자 없이 에이전트끼리 소통하며 스스로 다음 할 일을 정합니다.
- 핸드오프(Handoff): "다음은 네 차례"를 도구로 만들어, 도구를 호출하면 발언권이 다른 에이전트로 넘어갑니다.
- 다음 발언자(next_speaker)를 상태에 담고, 라우터가 그 값을 보고 다음 노드로 이동합니다.
- 명확한 종료 조건이 없으면 대화가 끝나지 않을 수 있으므로, 결론을 내는 역할(중재자)을 둡니다.

실행 방법:
    uv run python ch06_multi_agent_school/03_swarm_pattern.py
    중재자가 finish_discussion을 호출해야 끝납니다. 결론이 나지 않아 대화가 길어지면 Ctrl+C로 멈춥니다.
필요한 환경 변수: OPENAI_API_KEY (모델: gpt-4o)
표시: [보충] 실행을 위해 더한 코드, [수정] 책 코드의 오류를 고친 곳, [설명용 코드] 실행되지 않는 설명용 조각
"""

# .env 파일의 API 키를 환경 변수로 불러옵니다.
from dotenv import load_dotenv
load_dotenv()

# %% 2. 스웜의 핵심: 바통 터치 (Handoff) - LangGraph로 점심 메뉴 정하기 구현: 모듈과 상태 정의
# 스웜 패턴은 순차 패턴처럼 순서가 정해져 있지 않고, 병렬 패턴과 달리 결과를 취합하는 '조장'도 없습니다.
# 한식파, 양식파, 중재자 세 친구가 서로 턴을 넘기며 점심 메뉴를 정하는 스웜을 만듭니다.
# 루프 패턴에서 시도 횟수(attempt)를 상태에 넣었듯이, 스웜 패턴에서는 다음에 권한을 넘겨받을 에이전트를 상태에 둡니다.
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, List
from langchain_core.messages import BaseMessage, HumanMessage
import operator

# 1. 상태 정의
class SwarmState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    next_speaker: str # 다음에 말할 친구


# %% 2. 점심 메뉴 정하기 - 핸드오프 도구 정의
# 스웜 패턴에서는 다른 에이전트에 대한 호출을 도구로 정의합니다. 함수 본문은 안내 문구만 돌려줍니다.
# 일반적인 도구 호출과 달리, 핸드오프 도구를 호출하면 행동과 결정의 주체가 다른 에이전트로 넘어갑니다.
# 이 예제에서 실제로 발언권을 옮기는 일은 아래 agent_node와 라우터가 호출된 도구의 이름을 보고 처리합니다.
# 2. 도구 정의 (발언권 넘기기)
# 에이전트는 이 도구를 사용해 "다음은 누구 차례"인지 지정합니다.
@tool
def transfer_to_korean_expert():
    """한식 전문가에게 발언권을 넘깁니다."""
    return "한식 전문가에게 턴을 넘깁니다."

@tool
def transfer_to_western_expert():
    """양식 전문가에게 발언권을 넘깁니다."""
    return "양식 전문가에게 턴을 넘깁니다."

@tool
def transfer_to_moderator():
    """중재자에게 발언권을 넘깁니다."""
    return "중재자에게 턴을 넘깁니다."

@tool
def finish_discussion():
    """토론을 종료하고 결론을 냅니다."""
    return "토론 종료"


# %% 2. 점심 메뉴 정하기 - 에이전트 생성
# 서브 에이전트도 create_agent로 만들되, 핸드오프할 상대의 도구만 tools에 넣습니다.
# 한식파와 양식파는 서로와 중재자에게만 넘길 수 있고, 토론을 끝내는 finish_discussion은 중재자만 가집니다.
# 3. 에이전트 생성 (각자의 전문 분야와 성격 부여)
llm = ChatOpenAI(model="gpt-4o", temperature=0.7)

korean_agent = create_agent(
    model=llm,
    tools=[transfer_to_western_expert, transfer_to_moderator],
    system_prompt="당신은 한식을 사랑하는 학생입니다. 떡볶이나 김치찌개를 강력히 주장하세요."
)

western_agent = create_agent(
    model=llm,
    tools=[transfer_to_korean_expert, transfer_to_moderator],
    system_prompt="당신은 양식을 좋아하는 학생입니다. 피자나 햄버거가 최고라고 주장하세요."
)

moderator_agent = create_agent(
    model=llm,
    tools=[transfer_to_korean_expert, transfer_to_western_expert, finish_discussion],
    system_prompt="당신은 중재자입니다. 친구들의 의견을 듣고 합의점을 찾거나, 메뉴를 결정하여 토론을 끝내세요."
)


# %% 2. 점심 메뉴 정하기 - 노드 정의
# 핸드오프로 주도권을 옮기므로 노드 함수는 agent_node 하나로 충분합니다. 에이전트만 바꿔 세 노드에 씁니다.
# 호출한 도구 이름에 korean, western, finish가 들어 있으면 그에 맞춰 다음 발언자를 정하고,
# 그 밖의 경우(transfer_to_moderator 호출, 도구 호출 없음)에는 기본값인 중재자에게 돌아갑니다.
# 4. 노드 정의 (라우팅 로직 포함)
def agent_node(state, agent, name):
    result = agent.invoke(state)
    last_msg = result["messages"][-1]

    # 도구 호출(Handoff)을 분석해 다음 발언자 결정
    next_speaker = "moderator" # 기본값
    # [수정] create_agent는 도구 실행 후 최종 답변까지 마치므로 마지막 메시지에는 tool_calls가 없어 핸드오프가 무시됩니다.
    #        그래서 tool_calls가 있는 메시지 중 가장 마지막 것에서 도구 이름을 읽습니다. (책: 아래 주석 두 줄)
    # if result["messages"][-1].tool_calls:
    #     tool_name = result["messages"][-1].tool_calls[0]["name"]
    handoff_messages = [m for m in result["messages"] if getattr(m, "tool_calls", None)]
    if handoff_messages:
        tool_name = handoff_messages[-1].tool_calls[0]["name"]
        if "korean" in tool_name: next_speaker = "korean_expert"
        elif "western" in tool_name: next_speaker = "western_expert"
        elif "finish" in tool_name: next_speaker = "finish"

    # 에이전트의 최종 답변만 대화에 더하고, 정한 다음 발언자를 상태에 남깁니다.
    return {"messages": [last_msg], "next_speaker": next_speaker}


# %% 2. 점심 메뉴 정하기 - 그래프 조립
# 시작은 중재자가 열고, 세 노드 모두 route_speaker를 조건부 엣지로 달아 다음 노드를 고릅니다.
# route_speaker는 next_speaker 값을 노드 이름으로 그대로 돌려주고, "finish"일 때만 END로 보냅니다.
# 스웜은 노드 사이의 흐름이 코드에 드러나지 않아 추적이 어려우므로, 라우팅과 종료 조건을 신경 써서 설계해야 합니다.
# 5. 그래프 조립
workflow = StateGraph(SwarmState)

workflow.add_node("korean_expert", lambda state: agent_node(state, korean_agent, "korean"))
workflow.add_node("western_expert", lambda state: agent_node(state, western_agent, "western"))
workflow.add_node("moderator", lambda state: agent_node(state, moderator_agent, "moderator"))

# 시작은 중재자가 엽니다
workflow.add_edge(START, "moderator")

# 다음 발언자에 따라 동적으로 이동 (Router)
def route_speaker(state):
    speaker = state.get("next_speaker")
    if speaker == "finish": return END
    return speaker

workflow.add_conditional_edges("korean_expert", route_speaker)
workflow.add_conditional_edges("western_expert", route_speaker)
workflow.add_conditional_edges("moderator", route_speaker)

app = workflow.compile()


# %% 2. 점심 메뉴 정하기 - 실행
# 첫 메시지와 함께 next_speaker를 "moderator"로 넣어 실행하고, 내용이 있는 메시지만 출력합니다.
# 책의 '3. 너무 시끄러운 쉬는시간은 안돼!'에서 말하듯 에이전트끼리 말이 많아지면 모델 호출 횟수(비용)가 급격히 늘어납니다.
# 이 예제는 최대 대화 턴 수를 따로 제한하지 않으므로, 중재자가 finish_discussion을 호출해야 토론이 끝납니다.
# 6. 실행
print("=== 점심 메뉴 토론 시작 ===")
inputs = {
    "messages": [HumanMessage(content="오늘 점심 뭐 먹을까? 다들 의견 내봐.")],
    "next_speaker": "moderator"
}
result = app.invoke(inputs)

for msg in result["messages"]:
    if msg.content:
        print(f"\n {msg.content}")

# [책의 실행 결과] (실행 결과 예시)
# === 점심 메뉴 토론 시작 ===
#
# > 자, 다들 오늘 점심 뭐 먹고 싶어? 의견 들어보고 결정하자. (양식 전문가에게 턴을 넘깁니다)
# > 당연히 피자지! 학교 앞 새로 생긴 피자집 치즈가 예술이야. (한식 전문가에게 턴을 넘깁니다)
# > 무슨 소리야, 한국인은 밥심이지! 오늘 날씨도 쌀쌀한데 뜨끈한 김치찌개 먹으러 가자. (중재자에게 턴을 넘깁니다)
# > 음, 피자랑 김치찌개라... 너무 다른데? 그럼 절충해서 '김치 치즈 볶음밥' 어때? 분식집 가면 둘 다 만족할 것 같은데.
# > (한식/양식 전문가): 오, 그거 괜찮은데? 찬성!
# > 좋아, 그럼 오늘 점심은 학교 앞 분식집에서 김치 치즈 볶음밥으로 결정! 땅땅땅. (토론 종료)
