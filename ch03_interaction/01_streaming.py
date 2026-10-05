"""
[3장 1절] [Streaming] 에이전트의 머릿속을 들여다보기

이 절에서 배우는 것:
- invoke는 결과만 알려 주고, stream은 에이전트가 생각하고 행동하는 모든 단계(Step)를 실시간으로 보여 줍니다.
- updates 모드로 model(판단)과 tools(도구 실행) 단계마다 무엇이 바뀌었는지 추적합니다.
- stream_mode 네 가지(updates, values, messages, custom)의 쓰임새를 비교합니다.
- custom 모드로 도구 안에서 get_stream_writer로 보낸 쪽지를 밖에서 받아 봅니다.

실행 방법:
    uv run python ch03_interaction/01_streaming.py
필요한 환경 변수: OPENAI_API_KEY (리포지토리 루트의 .env 파일에 넣습니다)
표시: [보충] 실행에 필요한 코드, [수정] 실행에 맞게 고친 코드, [설명용 코드] 실행되지 않는 설명용 조각

eat 도구가 "맘마 없어"를 돌려주게 하려면 이 폴더에 food.txt를 두지 않습니다.
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


# %% 1. 결과만 통보받기(Invoke) vs 과정을 지켜보기(Stream) - 닫힌 방문 너머의 소리: invoke의 한계
# invoke(단발성 호출)는 방문을 닫아 두고 대화하는 것과 같습니다. 최종 결과만 돌려주므로
# 밥을 먹고 대답했는지, 배고파서 그냥 울었는지, 중간에 에러는 없었는지 같은 과정이 보이지 않습니다.
# 현재 기분 랜덤 설정 (예: 'hungry')
current_mood = "hungry"

# [보충] 3장 2절의 시스템 프롬프트와 baby_agent (middleware 인자 제외)
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

baby_agent = create_agent(
    model = my_model,
    tools = [cry, poo, eat],
    store=InMemoryStore(),
    system_prompt=my_system_prompt,
)

# invoke로 실행
result = baby_agent.invoke(
    {"messages": [{"role": "user", "content": "아기야 안녕?"}]}
)

# 결과 확인
print(f"아기의 대답: {result['messages'][-1].content}")

# [책의 실행 결과]
# 아기의 대답: 으으... 맘마!


# %% 2. stream으로 과정 지켜보기
# stream 메서드는 에이전트가 생각하고 행동하는 모든 단계(Step)를 실시간으로 보내 줍니다.
# updates 모드의 chunk는 {단계 이름: 그 단계에서 바뀐 상태} 모양이라 model(판단)과 tools(도구 실행) 단계가 차례로 찍힙니다.
# invoke 때는 보이지 않던 [밥 먹기 시도 -> 실패 -> 울기] 과정이 드러나, 어느 단계에서 판단을 잘못했는지 찾는 단서가 됩니다.
print(f"현재 기분: {current_mood}")

# 스트리밍 시작 (updates 모드: 변경된 사항만 보고받기)
for chunk in baby_agent.stream(
    {"messages": [{"role": "user", "content": "아기야 안녕?"}]},
    stream_mode="updates",
):
    # 단계별로 어떤 일이 일어났는지 출력
    for step, data in chunk.items():
        print(f"===== Step: {step} =====")
        # 복잡한 로그 중 사람이 읽을 수 있는 메시지 부분만 예쁘게 출력
        last_msg = data['messages'][-1]
        print(f"내용: {last_msg.content}")

        # 만약 도구를 사용했다면?
        if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
             print(f"도구 호출: {last_msg.tool_calls[0]['name']}")

# [책의 실행 결과: 베이비 모니터 화면] (괄호 안은 설명입니다)
# 현재 기분: hungry
#
# ===== Step: model =====
# 도구 호출: eat (배가 고프니 '먹기' 도구를 선택함)
#
# ===== Step: tools =====
# 내용: 맘마 없어 (도구 실행 결과: 밥이 없음)
#
# ===== Step: model =====
# 도구 호출: cry (밥이 없으니 '울기' 도구를 선택함)
#
# ===== Step: tools =====
# 내용: 울음 성공적으로 생성! (도구 실행 결과: 엥~ 하고 움)
#
# ===== Step: model =====
# 내용: 으앙! 배고파! (최종 답변)


# %% 3. 스트리밍 모드(stream_mode) 골라 쓰기 - (심화) 내 마음대로 로그 남기기: Custom Streaming
# stream_mode는 상황에 맞게 고릅니다: updates(각 단계에서 변한 것만, 흐름 파악용), values(누적된 전체 상태),
# messages(LLM이 치는 토큰을 하나씩, 채팅 화면용), custom(도구 안에서 개발자가 보낸 쪽지).
# custom 모드에서는 도구 안에서 get_stream_writer()로 얻은 writer에 넘긴 값을 그대로 받습니다.
# 쪽지는 poo 도구 안에서 보내므로 에이전트가 poo 도구를 호출할 때만 도착합니다.
# [설명용 코드] poo 도구에 몰래 심어 둔 커스텀 스트리밍 부분입니다. (전체 코드는 맨 위 실습 환경 설정에 있습니다)
# @tool("poo", args_schema=PooInput)
# def poo(...):
#
#     (...)
#     try:
#         writer = get_stream_writer() # <- 커스텀 스트리밍 정의!
#         writer("응가를 봅니다... 끙차!")
#     except:
#         pass
#     (...)

# 결과를 받아보는 코드
# [보충] ...으로 생략된 입력은 2번과 같은 메시지입니다. (책: 아래 주석 줄)
# for chunk in baby_agent.stream(..., stream_mode="custom"):
for chunk in baby_agent.stream({"messages": [{"role": "user", "content": "아기야 안녕?"}]}, stream_mode="custom"):
    print(f" 쪽지 도착: {chunk}")

# [책의 실행 결과]
# * 쪽지 도착: 응가를 봅니다... 끙차!
# * 쪽지 도착: 응가 성공적으로 생성!
