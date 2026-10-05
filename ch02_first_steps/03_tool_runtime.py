"""
[2장 3절] [ToolRuntime] 에이전트의 센스 기르기

이 절에서 배우는 것:
- 도구 함수에 runtime: ToolRuntime 인자 한 줄을 추가하면, 도구가 실행되는 순간 에이전트의
  상태(State), 맥락(Context), 저장소(Store)를 엿볼 수 있습니다.
- runtime.state로 행복도와 대화 기록을 읽어 상황에 맞게 반응하는 도구(check_mood, read_atmosphere)를 만듭니다.
- 도구가 Command(update={...})를 반환해 행동의 결과로 상태를 바꾸는 방법(play_toy)을 살펴봅니다.

실행 방법:
    uv run python ch02_first_steps/03_tool_runtime.py
필요한 환경 변수: 없음 (책처럼 도구를 정의만 하고 실행하지 않으므로 출력이 없습니다)
"""

# .env 파일의 API 키를 환경 변수로 불러옵니다.
from dotenv import load_dotenv
load_dotenv()


# %% 2. 도구에게 '감각' 달아주기: ToolRuntime
# ToolRuntime은 에이전트가 도구 상자를 여는 순간, 도구가 상태(State)와 맥락(Context)을 엿볼 수 있게 해 주는 연결 고리입니다.
# runtime 인자는 에이전트(LLM)에게 보이지 않습니다. 에이전트는 인자 없는 도구로 보고 호출하며,
# 도구가 실행되는 바로 그 순간(Runtime)에 LangGraph가 필요한 정보(State)를 도구에 쥐여 줍니다.
from langchain.tools import tool, ToolRuntime

# 에이전트의 상태(State)에 'happiness'(행복도)와 'messages'(대화기록)가 있다고 가정하겠습니다.

# @tool에 이름과 설명을 주지 않으면 함수 이름(check_mood)이 도구 이름, docstring이 도구 설명이 됩니다.
@tool
def check_mood(runtime: ToolRuntime) -> str:
    """현재 자신의 행복도와 대화량을 확인합니다. (외부 입력 아님)"""

    # runtime.state를 통해 도구가 직접 에이전트의 기억(State)을 엿봅니다!
    state = runtime.state

    happiness = state.get("happiness", 50)  # 기본 행복도 50
    messages = state.get("messages", [])

    return f"현재 행복도는 {happiness}이고, 엄마랑 {len(messages)}마디 대화를 나눴어요!"


# %% 3. 눈치 있는 아기 만들기 (Context Access)
# runtime.state의 messages(과거 대화 기록)를 뒤져 엄마가 화났는지 분위기를 파악하는 도구입니다.
# 사람이 보낸 메시지(HumanMessage)에 "혼난다", "그만해", "뚝" 중 하나라도 있으면 눈치를 챙겨 조용히 있고,
# 세 단어가 모두 없으면 더 놀아 달라는 반응을 돌려줍니다.
from langchain_core.messages import HumanMessage
from langchain.tools import tool, ToolRuntime

@tool
def read_atmosphere(runtime: ToolRuntime) -> str:
    """대화 기록을 보고 엄마가 화났는지 눈치를 봅니다."""

    # 1. 런타임(신경)을 통해 대화 기록(기억)을 가져옵니다.
    messages = runtime.state.get("messages", [])

    # 2. 최근 메시지 분석 (엄마가 화난 단어를 썼나?)
    angry_keywords = ["혼난다", "그만해", "뚝"]
    is_angry = False

    for msg in messages:
        if isinstance(msg, HumanMessage) and any(word in msg.content for word in angry_keywords):
            is_angry = True
            break

    # 3. 상황에 따른 반응 반환
    if is_angry:
        return "엄마 목소리가 무서워요... 조용히 있어야겠어요. (눈치 챙김)"
    else:
        return "엄마 기분이 좋아 보여요! 더 놀아달라고 할래요."


# %% 4. 행동의 결과로 상태 바꾸기: Command
# 도구가 실행된 뒤 상태(State)를 바꾸고 싶을 때는 Command를 반환합니다. 상태는 딕셔너리처럼 키와 값을 가지므로,
# update에 바꿀 키와 값을 적습니다. 이제 아기는 놀고 나면 "놀았다"고 말하는 데 그치지 않고 실제로 행복해집니다.
# 참고: 현재 langgraph의 Command에는 value 인자가 없어, 이 도구를 실제로 호출하면 TypeError가 납니다. (책에서는 정의만 합니다)
from langchain.tools import tool
from langgraph.types import Command

@tool
def play_toy() -> Command:
    """장난감을 가지고 놉니다. 기분이 좋아집니다."""

    # Command를 사용해 'happiness' 상태를 100으로 업데이트합니다.
    return Command(
        update={"happiness": 100},
        value="장난감을 가지고 놀아서 기분이 최고가 되었어요! (행복도 상승)"
    )
