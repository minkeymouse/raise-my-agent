"""
[3장 3절] [Middleware] 에이전트 답변 제어하기

이 절에서 배우는 것:
- BabyResponse 스키마와 response_format=ToolStrategy(...)로 에이전트의 답변을 정해진 구조로 받습니다.
- AgentState를 상속한 BabyState로 기분(mood)과 행동(action)을 에이전트의 상태에 둡니다.
- 클래스 기반 커스텀 미들웨어(AgentMiddleware)의 before_agent, before_model, wrap_model_call, wrap_tool_call,
  after_model, after_agent 메서드로 실행 단계마다 상태와 도구 선택을 제어합니다.

실행 방법:
    uv run python ch03_interaction/03_middleware_response.py
필요한 환경 변수: OPENAI_API_KEY (리포지토리 루트의 .env 파일에 넣습니다)
준비물 (선택): 책의 hungry 결과처럼 eat 도구가 성공하려면 이 폴더에 food.txt를 만들어 둡니다. 예: echo "맘마" > ch03_interaction/food.txt
표시: [보충] 실행을 위해 더한 코드, [수정] 책 코드의 오류를 고친 곳, [설명용 코드] 실행되지 않는 설명용 조각

기분은 before_agent에서 무작위로 정해지므로 결과가 매번 다릅니다.
실행하면 이 폴더(ch03_interaction/)에 아기 에이전트의 기록 파일(cry.txt, poo.txt)이 생깁니다.
"""

# %% [보충] 기록 파일 위치 정하기
# [보충] 도구가 cry.txt, poo.txt, food.txt를 현재 작업 디렉터리에서 읽고 쓰므로,
#        어디서 실행하든 이 폴더(ch03_interaction/)를 쓰도록 작업 디렉터리를 옮깁니다.
import os
from pathlib import Path

if "__file__" in globals():  # 주피터 커널로 셀 단위 실행할 때는 __file__이 없으므로 건너뜁니다.
    os.chdir(Path(__file__).parent)


# %% [보충] 3장 실습 환경 설정 (00_baby_agent_setup.py와 같은 코드)
# 2장에서 만든 아기 에이전트의 도구(cry, poo, eat)와 입력 스키마를 한곳에 모았습니다. 앞에서 배운 내용을 정리해 봅시다.
# @tool은 함수를 도구로 만들고, args_schema에 넣은 Pydantic 스키마는 인자의 기본값, 설명, 최소/최대값(ge, le)을 정해 두는 '틀'입니다.
import os
import random
from typing import Literal, Optional
from dotenv import load_dotenv, find_dotenv
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.agents import create_agent
from langgraph.store.memory import InMemoryStore
# (참고) get_stream_writer는 이번 장의 '커스텀 스트리밍'에서 배울 내용이지만,
# 도구 정의를 위해 미리 import 합니다.
from langgraph.config import get_stream_writer

load_dotenv(find_dotenv())

# --- 1. 입력 스키마 정의 ---
class CryInput(BaseModel):
    # [수정] 필드 이름을 cry 함수의 인자(cry_count)와 맞췄습니다(다르면 cry를 부를 때마다 TypeError). (책: 아래 주석 줄)
    # how_many_times: Optional[int] = Field(None, description="울음 횟수(없으면 랜덤)", ge=1, le=10)
    cry_count: Optional[int] = Field(None, description="울음 횟수(없으면 랜덤)", ge=1, le=10)

class PooInput(BaseModel):
    poo_shape: Literal["큰응가", "작은응가"] = Field("큰응가", description="응가 형태")
    poo_count: Optional[int] = Field(None, description="응가 개수(없으면 랜덤)", ge=1, le=10)

# --- 2. 도구(Tool) 정의 ---
@tool("cry", args_schema=CryInput)
def cry(cry_count: Optional[int] = None) -> str:
    """아기처럼 웁니다. cry.txt 파일에 기록합니다."""
    n = cry_count if isinstance(cry_count, int) and cry_count > 0 else random.randint(1, 10)
    with open("cry.txt", "a") as f:
        f.write("응애" * n + "\n")
    return "울음 성공적으로 생성!"

@tool("poo", args_schema=PooInput)
def poo(poo_shape: Literal["큰응가", "작은응가"] = "큰응가", poo_count: Optional[int] = None) -> str:
    """응가를 봅니다. poo.txt 파일에 기록합니다."""
    c = poo_count if isinstance(poo_count, int) and poo_count > 0 else random.randint(1, 10)

    # 이 부분은 3장 실습에서 스트리밍을 확인하기 위한 코드입니다.
    # writer에 넘긴 값은 stream_mode="custom"으로 스트리밍할 때 '쪽지'로 받아 볼 수 있습니다(3장 1절).
    # 에이전트 실행 밖에서 도구를 직접 부르면 get_stream_writer()가 RuntimeError를 내므로 try/except로 감쌉니다.
    try:
        writer = get_stream_writer()
        writer("응가를 봅니다... 끙차!")
    except:
        pass

    with open("poo.txt", "a") as f:
        f.write(poo_shape * c + "\n")
    return f"{poo_shape} {c}개 응가 성공적으로 생성!"

@tool("eat")
def eat() -> str:
    """맘마를 먹습니다. food.txt를 비웁니다."""
    if os.path.exists("food.txt"):
        with open("food.txt", "r+") as f:
            content = f.read()
            if content == "": return "맘마 없어"
            f.seek(0)
            f.truncate()
            return "맘마를 먹었어요."
    return "맘마 없어"

print("아기 에이전트와 도구 준비 완료!")

# [보충] 책에는 my_model의 정의가 없어 2장에서 사용한 gpt-4o-mini 모델로 만듭니다.
my_model = init_chat_model("openai:gpt-4o-mini")

# [보충] 이 절의 코드에 필요한 import (책에는 생략되어 있습니다)
from typing import Any, NotRequired
from langchain.messages import ToolMessage
from langchain.agents.middleware import AgentMiddleware, AgentState, ModelRequest, ModelResponse
from langchain.agents.structured_output import ToolStrategy


# %% 1. 결과를 원하는 구조에 맞추기
# LLM의 출력은 형식이 중구난방이라, 원하는 구조(스키마)를 정해 두고 그 모양으로 답하게 합니다.
# BabyResponse로 구조화하면 에이전트가 {"mood": "happy", "response": "꺄아~"}처럼 정해진 모양으로 답합니다.
# 스키마는 pydantic, dataclass, json, TypedDict로 만들 수 있고, 아래 3번에서 create_agent의 response_format 인자로 넣습니다.
class BabyResponse(BaseModel):
    mood: Literal["happy", "hungry", "poopoo", "sleepy"] = Field(..., description="현재 아기 에이전트의 기분")
    response: str = Field(..., description="아기 에이전트의 기분에 따른 짧은 의성어 답변")


# %% 2. 미들웨어를 활용한 에이전트(응용) - 상태 정의
# 기분(mood)은 네 값 중 하나를 꼭 가지는 속성, 행동(action)은 NotRequired라 없어도 되는 속성입니다.
# 결과 구조화를 위해 AgentState 대신 AgentState["BabyResponse"]를 상속해 응답 타입을 명시했습니다.
class BabyState(AgentState["BabyResponse"]):
    mood: Literal["happy", "hungry", "poopoo", "sleepy"]
    action: NotRequired[Literal["cry", "poo", "eat"]]


# %% 2. 미들웨어를 활용한 에이전트(응용) - 클래스 기반 커스텀 미들웨어
# 클래스로 미들웨어를 만들면 before_agent 같은 실행 시점이 메서드 이름이 되어 코드가 직관적입니다.
# 책은 아래 클래스를 여러 코드 블록으로 나누어 보여 줍니다. 여기서는 블록들을 차례대로 하나의 클래스로 이었습니다.
class BabyMiddleware(AgentMiddleware[AgentState["BabyResponse"], Any]):
    # state_schema와 tools로 에이전트에 아기용 상태(mood, action)와 도구를 부여합니다.
    state_schema = BabyState
    tools = [cry, poo, eat]

    # before_agent: 워크플로우가 시작될 때 가장 먼저 한 번 호출됩니다. 기분을 무작위로 정해 mood를 업데이트합니다.
    def before_agent(self, state: AgentState["BabyResponse"], runtime) -> dict[str, Any] | None:
        initial_mood = random.choice(["happy", "hungry", "poopoo", "sleepy"])
        return {"mood": initial_mood}

    # before_model: 모델을 부를 때마다 먼저 호출됩니다. 직전 도구 결과(ToolMessage)를 보고 mood와 action을 갱신합니다.
    # 예: eat 결과가 "맘마 없어"면 기분은 hungry로 두고 action을 cry로 바꿉니다. 할 행동이 없으면 기분에 맞는 행동을 정합니다.
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

    # wrap_model_call: 모델 호출을 감쌉니다. request로 이번 호출의 프롬프트와 도구를 고친 뒤 handler(request)로 실제 모델을 부릅니다.
    # action이 있으면 그 도구 하나만 남기고 구조화 출력을 잠시 끄며(response_format=None), 없으면 도구를 비워 최종 답변을 내게 합니다.
    # 참고: 설치된 langchain에서는 request의 속성을 직접 바꾸면 DeprecationWarning이 출력되지만 동작에는 문제가 없습니다.
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

    # wrap_tool_call: 도구 호출을 감쌉니다. cry와 poo에는 무작위 인자를 직접 넣은 뒤 handler(request)로 도구를 실행합니다.
    # 모델이 인자를 제대로 만들지 못하거나 도구 호출 과정이 복잡할 때 쓸 수 있습니다.
    def wrap_tool_call(self, request, handler):
        tool_name = getattr(request.tool, "name", request.tool_call.get("name"))
        args = request.tool_call.setdefault("args", {})

        # Inject random arguments for tools that need them
        if tool_name == "cry":
            args["cry_count"] = random.randint(1, 10)
        elif tool_name == "poo":
            args["poo_count"] = random.randint(1, 10)
            args["poo_shape"] = random.choice(["큰응가", "작은응가"])

        return handler(request)

    # after_model, after_agent: 모델 호출이 끝난 뒤, 에이전트 워크플로우가 끝난 뒤에 호출됩니다.
    # 주로 로깅, 메트릭 수집, 상태 정리 같은 후처리를 맡습니다.
    def after_model(self, state: AgentState["BabyResponse"], runtime) -> dict[str, Any] | None:
        print(f"모델 호출 완료, 기분은 {state.get('mood', 'happy')}에요!")
        return None

    def after_agent(self, state: AgentState["BabyResponse"], runtime) -> dict[str, Any] | None:
        print("에이전트 호출 완료!")
        return None


# %% 3. 에이전트 실행 결과 비교
# 미들웨어가 상태와 도구 선택을 맡으므로 에이전트를 만드는 코드는 간단해집니다.
# response_format: 모델이 구조화 출력을 직접 지원하면 스키마를 바로 넣어도 되고, 아니면 ToolStrategy로 감쌉니다.
# 최종 답변은 BabyResponse 객체로 상태의 structured_response 키에 담깁니다. updates 모드에서는 미들웨어 단계도 Step으로 보입니다.
baby_agent = create_agent(
    model=my_model,
    tools=[cry, poo, eat],
    response_format=ToolStrategy(BabyResponse),
    store=InMemoryStore(),
    middleware=[BabyMiddleware()]
)

for chunk in baby_agent.stream(
    {"messages": [{"role": "user", "content": "아기야 안녕?"}]},
    stream_mode="updates",
):
    for step, data in chunk.items():
        print(f"Step: {step}")
        print(f"Data: {data}")

# 참고: 책의 실행 결과는 로컬 모델(gpt-oss:20b, Ollama)로 얻은 것이라 gpt-4o-mini로 실행하면 출력 형식이 조금 다를 수 있습니다.
#
# [책의 실행 결과] mood가 sleepy인 경우
# Step: BabyMiddleware.before_agent
# Data: {'mood': 'sleepy'}
# Step: BabyMiddleware.before_model
# Data: {'action': 'cry'}
# Step: model
# Data: {'messages': [AIMessage(content='', additional_kwargs={}, response_metadata={'model': 'gpt-oss:20b', 'created_at': '2025-10-25T12:07:57.143742516Z', 'done': True, 'done_reason': 'stop', 'total_duration': 1205435636, 'load_duration': 129938934, 'prompt_eval_count': 191, 'prompt_eval_duration': 31296570, 'eval_count': 255, 'eval_duration': 965671644, 'model_name': 'gpt-oss:20b', 'model_provider': 'ollama'}, id='lc_run--403b8366-8a60-42ca-80a0-3771f7160176-0', tool_calls=[{'name': 'cry', 'args': {'cry_count': 1}, 'id': '15494293-8a03-476c-bb8d-11c3a376d5cc', 'type': 'tool_call'}], usage_metadata={'input_tokens': 191, 'output_tokens': 255, 'total_tokens': 446})]}
# 모델 호출 완료, 기분은 sleepy에요!
# Step: BabyMiddleware.after_model
# Data: None
# Step: tools
# Data: {'messages': [ToolMessage(content='울음 성공적으로 생성!', name='cry', id='1b25691e-6829-4f4f-a597-e0a7bbf7ff1c', tool_call_id='15494293-8a03-476c-bb8d-11c3a376d5cc')]}
# Step: BabyMiddleware.before_model
# Data: {'action': None, 'mood': 'happy'}
# Step: model
# Data: {'messages': [AIMessage(content='피피', additional_kwargs={}, response_metadata={'model': 'gpt-oss:20b', 'created_at': '2025-10-25T12:07:57.741816424Z', 'done': True, 'done_reason': 'stop', 'total_duration': 595034645, 'load_duration': 123457447, 'prompt_eval_count': 222, 'prompt_eval_duration': 22510257, 'eval_count': 109, 'eval_duration': 411043334, 'model_name': 'gpt-oss:20b', 'model_provider': 'ollama'}, id='lc_run--d262e20e-c80c-4981-a6f5-4089cf492e4c-0', usage_metadata={'input_tokens': 222, 'output_tokens': 109, 'total_tokens': 331})]}
# 모델 호출 완료, 기분은 happy에요!
# Step: BabyMiddleware.after_model
# Data: None
# 에이전트 호출 완료!
# Step: BabyMiddleware.after_agent
# Data: None
#
# [책의 실행 결과] mood가 hungry인 경우 (food.txt에 내용이 있어 eat 도구가 성공한 경우입니다)
# Step: BabyMiddleware.before_agent
# Data: {'mood': 'hungry'}
# Step: BabyMiddleware.before_model
# Data: {'action': 'eat'}
# Step: model
# Data: {'messages': [AIMessage(content='', additional_kwargs={}, response_metadata={'model': 'gpt-oss:20b', 'created_at': '2025-10-25T12:09:41.90951679Z', 'done': True, 'done_reason': 'stop', 'total_duration': 1773772160, 'load_duration': 127678584, 'prompt_eval_count': 184, 'prompt_eval_duration': 22223409, 'eval_count': 400, 'eval_duration': 1517857781, 'model_name': 'gpt-oss:20b', 'model_provider': 'ollama'}, id='lc_run--af6b5437-f045-4cff-80fe-c6a4f58e3dce-0', tool_calls=[{'name': 'eat', 'args': {}, 'id': '71318c8e-c9a1-484f-a838-c6e9a2e57a96', 'type': 'tool_call'}], usage_metadata={'input_tokens': 184, 'output_tokens': 400, 'total_tokens': 584})]}
# 모델 호출 완료, 기분은 hungry에요!
# Step: BabyMiddleware.after_model
# Data: None
# Step: tools
# Data: {'messages': [ToolMessage(content='맘마를 먹었어요.', name='eat', id='e73ce2ea-e8cf-4863-a6e4-1ad8eb055aa4', tool_call_id='71318c8e-c9a1-484f-a838-c6e9a2e57a96')]}
# Step: BabyMiddleware.before_model
# Data: {'action': None, 'mood': 'happy'}
# Step: model
# Data: {'messages': [AIMessage(content='구구~', additional_kwargs={}, response_metadata={'model': 'gpt-oss:20b', 'created_at': '2025-10-25T12:09:42.779726025Z', 'done': True, 'done_reason': 'stop', 'total_duration': 866975078, 'load_duration': 130670238, 'prompt_eval_count': 219, 'prompt_eval_duration': 22290774, 'eval_count': 172, 'eval_duration': 656239352, 'model_name': 'gpt-oss:20b', 'model_provider': 'ollama'}, id='lc_run--52c8d62b-77c6-4682-b61e-b926c080a613-0', usage_metadata={'input_tokens': 219, 'output_tokens': 172, 'total_tokens': 391})]}
# 모델 호출 완료, 기분은 happy에요!
# Step: BabyMiddleware.after_model
# Data: None
# 에이전트 호출 완료!
# Step: BabyMiddleware.after_agent
# Data: None
