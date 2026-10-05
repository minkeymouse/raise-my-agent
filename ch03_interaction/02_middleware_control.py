"""
[3장 2절] [Middleware] 에이전트의 행동을 제어하기

이 절에서 배우는 것:
- 미들웨어는 에이전트 워크플로우의 중간중간에 끼어들어 행동을 제어하는 모듈입니다.
- 기본 미들웨어(ModelCallLimitMiddleware, ToolCallLimitMiddleware, ToolRetryMiddleware)로 모델/도구 호출 횟수와 재시도를 관리합니다.
- 런타임에 담긴 컨텍스트(아기의 기분)를 읽는 커스텀 미들웨어를 @dynamic_prompt로 만들어 프롬프트를 동적으로 바꿉니다.
- context_schema로 컨텍스트의 틀을 정하고, invoke(..., context=...)로 실제 값을 넘깁니다.

실행 방법:
    uv run python ch03_interaction/02_middleware_control.py
필요한 환경 변수: OPENAI_API_KEY (리포지토리 루트의 .env 파일에 넣습니다)
표시: [보충] 실행에 필요한 코드, [수정] 실행에 맞게 고친 코드, [설명용 코드] 실행되지 않는 설명용 조각

마지막 실행은 기분을 무작위로 정하므로 결과가 매번 다릅니다.
실행하면 이 폴더(ch03_interaction/)에 아기 에이전트의 기록 파일(cry.txt, poo.txt)이 생깁니다.
"""

# %% [보충] 기록 파일 위치 정하기
# [보충] 도구가 읽고 쓰는 cry.txt, poo.txt, food.txt가 이 폴더(ch03_interaction/)에 생기도록 작업 디렉터리를 옮깁니다.
import os
from pathlib import Path

if "__file__" in globals():  # 주피터 커널로 셀 단위 실행할 때는 __file__이 없으므로 건너뜁니다.
    os.chdir(Path(__file__).parent)


# %% [보충] 3장 실습 환경 설정 (00_baby_agent_setup.py와 같은 코드)
# 2장에서 만든 아기 에이전트의 도구(cry, poo, eat)와 입력 스키마를 하나로 합쳐 정의합니다. 앞에서 배운 내용을 정리해 봅시다.
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
    # [수정] 스키마의 필드 이름은 cry 함수의 인자 이름(cry_count)과 같아야 합니다. (책: 아래 주석 줄)
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

# [보충] 2장에서 사용한 gpt-4o-mini 모델
my_model = init_chat_model("openai:gpt-4o-mini")

# [보충] 3장 1절의 current_mood
# 현재 기분 랜덤 설정 (예: 'hungry')
current_mood = "hungry"


# %% 2. 기본 미들웨어
# 랭체인에 미리 정의된 기본 미들웨어는 인자만 알맞게 넣으면 바로 쓸 수 있습니다.
# run_limit은 한 번 실행할 때의 최대 호출 횟수, exit_behavior는 횟수를 넘었을 때의 행동입니다("end"는 루프를 끝냅니다).
# 아래는 차례로 모델 호출 최대 4번, 모든 도구 호출을 합쳐 최대 2번, tool_name으로 지정한 poo만 최대 1번,
# 실패한 도구 호출은 최대 3번 다시 시도(ToolRetryMiddleware)하도록 설정합니다.
from langchain.agents.middleware import ModelCallLimitMiddleware, ToolCallLimitMiddleware, ToolRetryMiddleware

model_call_limit_middleware = ModelCallLimitMiddleware(run_limit=4, exit_behavior="end")
global_tool_call_limit = ToolCallLimitMiddleware(run_limit=2, exit_behavior="end")
poo_tool_call_limit = ToolCallLimitMiddleware(tool_name="poo",run_limit=1, exit_behavior="end")
tool_retry_middleware = ToolRetryMiddleware(max_retries=3)

# %% 2. 기본 미들웨어 - 시스템 프롬프트 정의
# 기분(current_mood)에 따라 어떤 도구를 몇 번 부를지 프롬프트로 정합니다.
# 프롬프트가 길고 복잡할수록 오류 가능성이 높아지므로, 규칙은 미들웨어와 워크플로우로 지키게 하고 프롬프트는 짧게 쓰는 편이 좋습니다.
my_system_prompt = f"""
너는 아기 에이전트야. 너의 지금 기분은 {current_mood}야. 이 값에 따라 아래 상태 규칙을 엄격히 따르며 응답해.
- happy: 도구 호출 금지. 아기가 기분이 좋은 것 처럼 짧은 의성어로 즉시 대답해.
- hungry: 반드시 eat 도구를 1회 호출.
    • food.txt가 있어 내용을 읽으면(도구가 텍스트 반환) 내부 상태를 happy로 전환했다고 가정하고, 짧은 의성어로 마무리.
    • food.txt가 없으면(도구가 "맘마 없어" 등) hungry 유지 후 cry 도구를 1회 호출하고, 현재의 기분을 고려해서 짧은 의성어로 마무리.
- poopoo: 반드시 poo 도구를 1회 호출(poo_shape="큰응가" 또는 "작은응가"), 성공 후 cry 도구 1회 호출하고, 현재의 기분을 고려해서 짧은 의성어로 마무리.
- sleepy: cry 도구 1회 호출하고, 현재의 기분을 고려해서 짧은 의성어로 마무리.
추가 규칙:
- 도구 입력 값은 위에서 지정한 값으로 단순하게 사용. 숫자 인자는 생략 가능하며, 생략 시 내부에서 1~10 사이 랜덤으로 설정.
- 그 외에는 도구를 사용하지 말고, 항상 현재의 기분을 고려해서 짧은 의성어로 마무리해.
"""

# %% 2. 기본 미들웨어 - 미들웨어를 넣어 에이전트 만들기
# middleware 인자에 기본 미들웨어를 리스트로 넣습니다. 호출 횟수가 계속 기록되다가 최대 횟수를 넘으면 루프가 바로 끝납니다.
# 아래 결과는 이 에이전트를 stream으로 실행했을 때의 출력 일부로, poo 호출이 1회 제한을 넘어 루프가 끝난 모습입니다.
baby_agent = create_agent(
    model = my_model,
    tools = [cry, poo, eat],
    store=InMemoryStore(),
    middleware=[model_call_limit_middleware, global_tool_call_limit, poo_tool_call_limit, tool_retry_middleware],
    system_prompt=my_system_prompt,
)

# [책의 실행 결과]
# (생략)
#  chunk: {'ToolCallLimitMiddleware[poo].before_model': {'jump_to': 'end', 'messages': [AIMessage(content="'poo' tool call limits exceeded: run limit (1/1)", additional_kwargs={}, response_metadata={}, id='4b0123d6-7cb7-43f1-8770-596563673f5a')]}}


# %% 3. 커스텀 미들웨어 만들어 보기 - 컨텍스트 스키마
# 기본 미들웨어로 부족할 때는 커스텀 미들웨어를 만듭니다. 먼저 아기의 기분을 컨텍스트로 옮깁니다.
# Context는 실제 값이 아니라 스키마(틀)이며, 실제 값은 실행할 때 context={"mood": "happy"}처럼 넣어 줍니다.
from typing import TypedDict  # [보충] Context 정의에 필요한 import


class Context(TypedDict):
    mood: Literal["happy", "hungry", "poopoo", "sleepy"]


# %% 3. 커스텀 미들웨어 만들어 보기 - @dynamic_prompt
# 에이전트는 invoke나 stream으로 실행되는 순간 런타임을 가지며, 런타임에는 context, store, stream_writer 같은 정보가 담깁니다.
# @dynamic_prompt는 ModelRequest를 받아 시스템 프롬프트 문자열을 돌려주는 함수를 미들웨어로 바꿔 줍니다(실행 시점: wrap_model_call).
# 여기서는 런타임 컨텍스트의 mood를 읽어 기분에 맞는 프롬프트를 그때그때 만듭니다.
from langchain.agents.middleware import dynamic_prompt, ModelRequest

@dynamic_prompt
def mood_based_prompt(request: ModelRequest) -> str:
    """기분에 맞춰 시스템 프롬프트를 생성합니다.

    ModelRequest 객체는 다음 속성들을 포함합니다:
    - request.runtime: Runtime 객체 (context, store, stream_writer 등 접근 가능)
    - request.state: 현재 에이전트 상태
    - request.model: 사용할 모델
    - request.tools: 사용 가능한 도구 목록
    """
    # 런타임의 컨텍스트에서 mood 값을 가져옵니다
    # context는 실행 시 전달된 불변(immutable) 데이터입니다
    mood = request.runtime.context.get("mood", "happy")

    base_prompt = f"너는 아기 에이전트야. 너의 지금 기분은 {mood}야. 이 값에 따라 아래 상태 규칙을 엄격히 따르며 응답해."

    if mood == "happy":
        return f"{base_prompt} 도구 호출 금지. 아기가 기분이 좋은 것 처럼 짧은 의성어로 즉시 대답해."
    elif mood == "hungry":
        return f"{base_prompt} 반드시 eat 도구를 1회 호출. food.txt가 있어 내용을 읽으면(도구가 텍스트 반환) 내부 상태를 happy로 전환했다고 가정하고, 짧은 의성어로 마무리. food.txt가 없으면(도구가 '맘마 없어' 등) hungry 유지 후 cry 도구를 1회 호출하고, 현재의 기분을 고려해서 짧은 의성어로 마무리."
    elif mood == "poopoo":
        return f"{base_prompt} 반드시 poo 도구를 1회 호출(poo_shape='큰응가' 또는 '작은응가'), 성공 후 cry 도구 1회 호출하고, 현재의 기분을 고려해서 짧은 의성어로 마무리."
    elif mood == "sleepy":
        return f"{base_prompt} cry 도구 1회 호출하고, 현재의 기분을 고려해서 짧은 의성어로 마무리."

    return "프롬프트가 선택되지 않았습니다."


# %% 3. 커스텀 미들웨어 만들어 보기 - 미들웨어와 context_schema를 넣어 에이전트 만들기
# 커스텀 미들웨어를 기본 미들웨어와 함께 리스트에 넣고, context_schema 인자도 꼭 함께 넘깁니다.
baby_agent = create_agent(
    # ...
    # [보충] ...으로 생략된 인자는 앞의 에이전트와 같습니다.
    # 시스템 프롬프트는 mood_based_prompt 미들웨어가 만들므로 system_prompt 인자는 필요 없습니다.
    model = my_model,
    tools = [cry, poo, eat],
    store=InMemoryStore(),
    middleware=[
        mood_based_prompt,
        model_call_limit_middleware,
        global_tool_call_limit,
        poo_tool_call_limit,
        tool_retry_middleware
    ],
    context_schema=Context
)

# %% 3. 커스텀 미들웨어 만들어 보기 - 컨텍스트를 넣어 실행하기
# 기분을 무작위로 고르고 invoke의 context 인자로 넘깁니다. mood_based_prompt가 이 값을 읽어 프롬프트를 고릅니다.
# 아래 결과는 미들웨어가 작동해 poo 도구 호출 횟수가 제한을 넘으며 루프가 끝난 경우입니다.
current_mood = random.choice(["happy", "hungry", "poopoo", "sleepy"])

result = baby_agent.invoke(
    {"messages": [{"role": "user", "content": "아기야 안녕?"}]},
    context={"mood": current_mood}
)
print("결과:", result["messages"][-1].content)

# [책의 실행 결과]
# 결과: 'poo' tool call limits exceeded: run limit (1/1)
