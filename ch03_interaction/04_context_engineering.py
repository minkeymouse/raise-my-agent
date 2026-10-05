"""
[3장 4절] 컨텍스트 엔지니어링

이 절에서 배우는 것:
- 상태(State)는 에이전트의 데이터를 저장하고 업데이트하는 것, 컨텍스트는 실행에 필요한 정보를 주는 더 넓은 개념입니다.
- 컨텍스트는 쓰이는 단계에 따라 모델 컨텍스트(프롬프트, 답변 구조, 대화 히스토리, 도구),
  도구 컨텍스트(도구가 접근하는 상태, 스토어, 런타임), 라이프사이클 컨텍스트(대화 요약, 로그 등)로 나눌 수 있습니다.
- 3절 BabyMiddleware의 메서드를 다시 보며 각 코드가 어떤 컨텍스트 엔지니어링인지 구분합니다.
- 에이전트의 성패는 미들웨어로 상태, 런타임, 스토어 같은 데이터를 참고해 컨텍스트를 제어하는 데 달려 있습니다.

이 절의 코드는 3절 코드를 다시 보는 복습용이라, 실행하면 정의만 하고 아무것도 출력하지 않습니다.
에이전트에 넣어 실제로 동작시키는 전체 코드는 03_middleware_response.py에 있습니다.

실행 방법:
    uv run python ch03_interaction/04_context_engineering.py
필요한 환경 변수: 없음 (모델을 호출하지 않습니다)
표시: [보충] 실행을 위해 더한 코드, [수정] 책 코드의 오류를 고친 곳, [설명용 코드] 실행되지 않는 설명용 조각
"""

# .env 파일의 API 키를 환경 변수로 불러옵니다.
from dotenv import load_dotenv
load_dotenv()

# [보충] 아래 코드에 필요한 import (책에는 생략되어 있습니다)
import random
from typing import Literal, NotRequired
from langchain.messages import ToolMessage
from langchain.agents.middleware import AgentState, ModelRequest, ModelResponse
from pydantic import BaseModel, Field


# [보충] 아래 코드의 AgentState["BabyResponse"]가 가리키는 스키마입니다. (3장 3절의 코드)
class BabyResponse(BaseModel):
    mood: Literal["happy", "hungry", "poopoo", "sleepy"] = Field(..., description="현재 아기 에이전트의 기분")
    response: str = Field(..., description="아기 에이전트의 기분에 따른 짧은 의성어 답변")


# %% 2. 모델 컨텍스트 엔지니어링 - 상태와 메시지를 제어하는 before_model (3절 BabyMiddleware의 메서드)
# 모델 컨텍스트는 모델 호출 때 들어가는 프롬프트, 대화 메시지, 도구, 답변 구조입니다.
# before_model은 모델이 실행되기 전에 상태(기분, 행동)를 업데이트합니다. 이 상태도 모델에게 주는 맥락입니다.
# 직전 ToolMessage를 보고 if-else 규칙으로 다음 행동을 정하는 부분은 '메시지를 활용한 모델 컨텍스트 엔지니어링'입니다.
# 정확성을 담보하려면 모든 판단을 모델에 맡기기보다 이렇게 규칙으로 제어하는 편이 더 안전합니다.
def before_model(self, state: AgentState["BabyResponse"], runtime):
    messages = state.get("messages", [])
    if messages:
        last_message = messages[-1]
        if isinstance(last_message, ToolMessage):
            name = getattr(last_message, "name", "")
            content = getattr(last_message, "content", "")

            if name == "eat" and "맘마를 먹었어요" in content:
                return {"action": None, "mood": "happy"}
            elif name == "eat" and "맘마 없어" in content:
                return {"action": "cry", "mood": "hungry"}
            elif name in ("cry", "poo"):
                return {"action": None, "mood": "happy"}

    mood = state.get("mood", "happy")
    if state.get("action") is None and mood in ("poopoo", "sleepy", "hungry"):
        action_map = {"poopoo": "poo", "sleepy": "cry", "hungry": "eat"}
        return {"action": action_map[mood]}

    print(f"지금은 행동 준비 중, 기분은 {mood}에요!")
    return


# %% 2. 모델 컨텍스트 엔지니어링 - 프롬프트와 도구 선택을 제어하는 wrap_model_call (3절 BabyMiddleware의 메서드)
# 상황에 맞춰 시스템 프롬프트를 바꾸는 부분은 '프롬프트를 제어하는 컨텍스트 엔지니어링'이고,
# request.tools를 걸러 쓸 도구를 정하는 부분은 '도구 선택을 제어하는 모델 컨텍스트 엔지니어링'입니다.
def wrap_model_call(self, request: ModelRequest, handler) -> ModelResponse:
    mood = request.state.get("mood", "happy")
    action = request.state.get("action", None)

    base_prompt = f"지금 너의 기분은 {mood}야. 매우 짧은 한국어 아기 의성어만."

    if action in ("cry", "poo", "eat"):
        tool_name = action
        system_prompt = f"{base_prompt} 이번 턴에는 반드시 {tool_name} 도구를 호출해"
        request.response_format = None
    else:
        tool_name = None
        system_prompt = base_prompt
        request.tools = []

    # Set system prompt and filter tools
    request.system_prompt = system_prompt
    if tool_name:
        request.tools = [t for t in request.tools if getattr(t, "name", None) == tool_name]

    return handler(request)


# %% 3. 도구 컨텍스트 엔지니어링 - 도구 인자를 제어하는 wrap_tool_call (3절 BabyMiddleware의 메서드)
# 도구를 정의하거나 선택하고 직접 호출하는 과정에서 정보를 활용하는 것을 도구 컨텍스트 엔지니어링이라고 부릅니다.
# 여기서는 도구에 어떤 인자를 넣을지 미들웨어가 정합니다.
def wrap_tool_call(self, request, handler):
    tool_name = getattr(request.tool, "name", request.tool_call.get("name"))
    args = request.tool_call.setdefault("args", {})

    if tool_name == "cry":
        args["cry_count"] = random.randint(1, 10)
    elif tool_name == "poo":
        args["poo_count"] = random.randint(1, 10)
        args["poo_shape"] = random.choice(["큰응가", "작은응가"])

    return handler(request)


# %% 4. 라이프사이클 컨텍스트 엔지니어링 - 상태 스키마 정의
# 라이프사이클 컨텍스트 엔지니어링은 긴 메시지를 요약하거나, 실행 전반이 예상 범위를 벗어나지 않도록 제어하는 것입니다.
# 스키마로 상태를 정의해 기분을 네 값 중 하나로 묶어 두는 것도 여기에 속합니다.
class BabyState(AgentState["BabyResponse"]):
    mood: Literal["happy", "hungry", "poopoo", "sleepy"]
    action: NotRequired[Literal["cry", "poo", "eat"]]


# %% 4. 라이프사이클 컨텍스트 엔지니어링 - 도구와 스키마에 구체적인 설명 추가하기
# 도구와 스키마에 설명을 구체적으로 적어 상태/도구에 관한 정보를 주는 것도 라이프사이클 컨텍스트 엔지니어링입니다.
# [설명용 코드] 책은 eat 도구의 데코레이터 줄과 BabyResponse의 response 필드 한 줄만 보여 줍니다.
# @tool("eat", description="맘마를 먹습니다. 디렉토리에 있는 food.txt 파일을 읽고 내용을 지웁니다. 없으면 배고픔 상태를 기록합니다.")
#
# response: str = Field(..., description="아기 에이전트의 기분에 따른 짧은 의성어 답변")
