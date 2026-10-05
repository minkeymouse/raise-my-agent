"""
[3장 도입] 실습 환경 설정: 아기 에이전트 소환

3장에서는 아기 에이전트의 머릿속을 들여다보고(1절 스트리밍), 행동과 답변을 제어하고(2, 3절 미들웨어),
맥락을 다듬고(4절 컨텍스트 엔지니어링), 위험한 행동 앞에서 부모가 개입하는(5절 HITL) 방법을 배웁니다.

이 절에서 배우는 것:
- 2장에서 만든 아기 에이전트의 도구(cry, poo, eat)와 입력 스키마를 하나로 모아 정의합니다.
- poo 도구에는 1절에서 배울 커스텀 스트리밍(get_stream_writer)이 미리 들어 있습니다.

01, 02, 03, 05 파일의 맨 앞에도 같은 코드가 있으므로 이 파일을 먼저 실행하지 않아도 됩니다.

실행 방법 (3장의 명령은 모두 리포지토리 루트에서 실행합니다):
    uv run python ch03_interaction/00_baby_agent_setup.py
필요한 환경 변수: 없음 (도구만 정의하고 모델은 만들지 않습니다)
준비물 (3장 공통): 01, 02, 03, 05 파일은 OPENAI_API_KEY가 필요합니다.
    https://platform.openai.com/api-keys 에서 발급한 키를 리포지토리 루트의 .env 파일에 적어 둡니다.
준비물 (선택): 아기에게 맘마를 주려면 이 폴더에 food.txt를 만들어 둡니다. 예: echo "맘마" > ch03_interaction/food.txt
    eat 도구는 food.txt에 내용이 있으면 비우면서 "맘마를 먹었어요."를, 없거나 비어 있으면 "맘마 없어"를 돌려줍니다.
표시: [보충] 실행에 필요한 코드, [수정] 실행에 맞게 고친 코드, [설명용 코드] 실행되지 않는 설명용 조각
"""

# %% 실습 환경 설정: 아기 에이전트 소환
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
