"""
[3장 5절] [HITL] 사람의 개입과 안전 장치

이 절에서 배우는 것:
- Human-in-the-loop(HITL): 되돌리기 어렵거나 미심쩍은 행동 앞에서 에이전트를 멈추고 사람에게 허락을 받습니다.
- HumanInTheLoopMiddleware의 interrupt_on으로 멈출 도구를 정하고, 멈춘 상태를 기억하도록 체크포인터를 함께 씁니다.
- Command(resume=...)로 부모의 세 가지 결정(approve, edit, reject)을 전달해 실행을 이어 갑니다.

실행 방법:
    uv run python ch03_interaction/05_human_in_the_loop.py
필요한 환경 변수: OPENAI_API_KEY (리포지토리 루트의 .env 파일에 넣습니다)
표시: [보충] 실행에 필요한 코드, [수정] 실행에 맞게 고친 코드, [설명용 코드] 실행되지 않는 설명용 조각

invoke 결과를 출력하지 않으므로 화면에는 도구 준비 완료 메시지만 나옵니다.
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


# %% 1. 안전 장치 설치하기: HITL 미들웨어
# HumanInTheLoopMiddleware는 특정 도구를 쓰기 전에 에이전트를 '일시 정지'시키고 부모의 결정을 기다립니다.
# interrupt_on에는 도구 이름별로 멈출지와 허용할 결정을 적습니다. False로 두거나 적지 않은 도구는 멈추지 않고 바로 실행됩니다.
# 에이전트가 멈췄다가 다시 시작하려면 멈춘 시점의 상태를 기억해야 하므로 체크포인터를 반드시 함께 씁니다.
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.middleware import HumanInTheLoopMiddleware

# 에이전트 생성
baby_agent = create_agent(
    model=my_model,
    tools=[cry, poo, eat],
    store=InMemoryStore(),
    middleware=[
        # 특정 도구(eat)를 호출할 때만 멈추도록 설정
        HumanInTheLoopMiddleware(
            interrupt_on={
                "eat": {"allowed_decisions": ["approve", "reject", "edit"]},
                "cry": False # 울음은 자유롭게 허용!
            }
        )
    ],
    checkpointer=InMemorySaver(), # 일시 정지를 위한 기억 장치
)


# %% 2. 부모의 세 가지 선택: Approve, Edit, Reject - 1. Approve (허락하기)
# 에이전트가 eat 앞에서 멈추면(Interrupt), 부모는 Command(resume=...)에 결정을 담아 실행을 이어 갑니다.
# config의 thread_id로 어느 대화를 이어 갈지 알려 줍니다. approve는 도구 호출을 그대로 허가합니다.
from langgraph.types import Command  # [보충] Command import

# [보충] 대화방(config)을 정하고 말을 걸어 eat 앞에서 멈춘 상태를 만듭니다.
# resume은 멈춘 대화를 이어 가는 것이므로 Edit, Reject도 각자 새 대화방에서 같은 방법으로 먼저 멈춥니다.
config = {"configurable": {"thread_id": "approve"}}
baby_agent.invoke({"messages": [{"role": "user", "content": "아기야, 맘마 먹을 시간이야!"}]}, config=config)

# 승인 메시지를 담아 다시 실행(resume)
baby_agent.invoke(
    Command(resume={"decisions": [{"type": "approve"}]}),
    config=config
)


# %% 2. 부모의 세 가지 선택: Approve, Edit, Reject - 2. Edit (교정하기)
# edit는 edited_action으로 부모가 도구 이름과 인자를 직접 고쳐 실행시킵니다.
# 실제 eat 도구에는 음식을 고르는 기능이 없으므로, edit 결정이 이런 식으로 쓰인다는 것만 이해하면 됩니다.
config = {"configurable": {"thread_id": "edit"}}  # [보충]
baby_agent.invoke({"messages": [{"role": "user", "content": "아기야, 맘마 먹을 시간이야!"}]}, config=config)  # [보충]

baby_agent.invoke(
    Command(resume={
        "decisions": [{
            "type": "edit",
            "edited_action": {
                "name": "eat", 
                "args": {"food_type": "유기농 맘마"} # 부모가 인자를 수정!
            }
        }]
    }),
    config=config
)


# %% 2. 부모의 세 가지 선택: Approve, Edit, Reject - 3. Reject (거절하기)
# reject는 도구 호출을 거절합니다. 도구는 실행되지 않고 message에 적은 이유가 에이전트에게 전달되며,
# 에이전트는 이 피드백을 듣고 다음 행동을 정합니다.
config = {"configurable": {"thread_id": "reject"}}  # [보충]
baby_agent.invoke({"messages": [{"role": "user", "content": "아기야, 맘마 먹을 시간이야!"}]}, config=config)  # [보충]

baby_agent.invoke(
    Command(resume={
        "decisions": [{
            "type": "reject",
            "message": "안 돼, 지금은 밥 먹을 시간이 아니야." # 거절 사유 전달
        }]
    }),
    config=config
)
