"""
[2장 2절] [Tool Schema] 에이전트에게 사용 설명서 쥐어주기

이 절에서 배우는 것:
- 도구의 설명은 에이전트가 무시할 수도 있지만, 도구 스키마는 입력값을 강제하는 '제약'입니다.
  스키마가 없으면 울음 횟수 자리에 '많이' 같은 문자열이 들어가 도구 호출이 실패할 수 있습니다.
- Pydantic의 BaseModel과 Field로 입력 규칙(CryInput, PooInput)을 정의하고, @tool의 args_schema로 장착합니다.
- create_agent()에 도구 리스트를 넘기면 모든 도구의 스키마가 자동으로 통합됩니다.

실행 방법:
    uv run python ch02_first_steps/02_tool_schema.py
필요한 환경 변수: OPENAI_API_KEY (리포지토리 루트의 .env)
에이전트가 도구를 호출하면 이 폴더(ch02_first_steps/)에 cry.txt나 poo.txt가 생깁니다.

표시: [보충] 실행을 위해 더한 코드, [수정] 책 코드의 오류를 고친 곳, [설명용 코드] 실행되지 않는 설명용 조각
"""

# .env 파일의 API 키를 환경 변수로 불러옵니다.
from dotenv import load_dotenv
load_dotenv()


# %% [보충] 실습 준비
# [보충] 아래 도구들이 쓰는 random 모듈입니다. (책에서는 1절에서 임포트했습니다.)
import random

# [보충] 리포 루트에서 실행해도 cry.txt, poo.txt가 이 장 폴더(ch02_first_steps/)에 생기도록 작업 디렉토리를 옮깁니다.
import os
from pathlib import Path

if "__file__" in globals():  # 주피터 커널로 셀 단위 실행할 때는 __file__이 없으므로 건너뜁니다.
    os.chdir(Path(__file__).parent)


# %% 2. Pydantic으로 규칙 정의하기
# 스키마는 도구에 전달할 데이터의 '틀'로, Pydantic의 BaseModel을 상속해 만듭니다.
# Field로 기본값(default), LLM에게 보내는 설명(description), 최소값과 최대값(ge, le)을 지정합니다.
# 에이전트가 범위를 벗어난 값(예: 100번 울기)이나 Literal 보기에 없는 값(예: 별모양 응가)을 넣으면 Pydantic이 막아 줍니다.
from pydantic import BaseModel, Field
from typing import Literal, Optional

# 1. 울기 규칙 정의
class CryInput(BaseModel):
    # Field(...) 안에 설명을 적으면 LLM이 이를 읽고 판단합니다.
    cry_count: Optional[int] = Field(
        default=None,
        description="울음 횟수입니다. 1에서 10 사이의 정수만 가능합니다.",
        ge=1, le=10  # ge: greater or equal, le: less or equal
    )

# 2. 응가 규칙 정의
class PooInput(BaseModel):
    # Literal을 사용하면 LLM이 선택할 수 있는 보기를 제한할 수 있습니다.
    poo_shape: Literal["큰응가", "작은응가"] = Field(
        default="큰응가",
        description="응가 형태입니다. '큰응가' 혹은 '작은응가' 중 하나여야 합니다."
    )
    poo_count: Optional[int] = Field(
        default=None,
        description="응가 개수입니다. (없으면 랜덤)",
        ge=1, le=10
    )


# %% 3. 도구에 스키마 장착하기
# @tool 데코레이터의 args_schema 파라미터로 앞에서 만든 규칙(스키마)을 도구에 연결합니다.
# 입력값 검증을 Pydantic이 맡으므로, 1절에서 isinstance로 값을 확인하던 코드가 빠져 가독성이 좋아졌습니다.
# LLM에게 보이는 인자 규칙은 함수의 타입 힌트가 아니라 args_schema를 따릅니다.
# 그래서 poo의 poo_shape가 str로 적혀 있어도 PooInput의 두 보기('큰응가', '작은응가')만 허용됩니다.
from langchain.tools import tool

# args_schema에 우리가 만든 규칙(CryInput)을 넣어줍니다.
@tool("cry", args_schema=CryInput, description="아기처럼 웁니다. 디렉토리에 cry.txt 파일에 기록을 추가합니다.")
def cry(cry_count: Optional[int] = None) -> str:
    """우는 행동을 수행합니다."""
    # 이제 입력값 검증은 Pydantic이 앞에서 다 처리해줬으므로, 로직에만 집중하면 됩니다!
    n = cry_count if cry_count is not None else random.randint(1, 10)

    with open("cry.txt", "a") as f:
        f.write("응애" * n + "\n")
    return "울음 성공적으로 생성!"

# 응가 툴에도 규칙 장착!
@tool("poo", args_schema=PooInput, description="응가를 봅니다. 디렉토리에 poo.txt 파일에 기록을 추가합니다.")
def poo(poo_shape: str = "큰응가", poo_count: Optional[int] = None) -> str:
    """응가 행동을 수행합니다."""
    c = poo_count if poo_count is not None else random.randint(1, 10)

    with open("poo.txt", "a") as f:
        f.write(poo_shape * c + "\n")
    return f"{poo_shape} {c}개 응가 성공적으로 생성!"


# %% 4. 입력이 없는 도구 (eat)
# eat처럼 실행만 하면 되는 도구는 별도의 Pydantic 스키마 없이 함수만 정의해도 충분합니다.
# 이때 스키마는 {}(빈 객체)가 되며, 에이전트는 "이건 그냥 실행하면 되는구나"라고 판단합니다.
# 참고: 책은 본문을 "1절과 동일"로 생략하고 "냠냠"만 반환합니다. food.txt를 비우는 전체 본문은 01_tools.py에 있습니다.
@tool("eat", description="맘마를 먹습니다. food.txt를 비웁니다.")
def eat() -> str:
    # ... (1절과 동일) ...
    return "냠냠"


# %% 5. 아기 에이전트에게 툴 스키마 장착하기
# create_agent()에는 도구를 리스트에 담아 넘기기만 하면 됩니다. 예전처럼 bind_tools()나 initialize_agent()를 부를 필요가 없습니다.
# create_agent()는 각 도구의 스키마와 설명을 모아 LLM이 이해하는 JSON Schema로 합치고,
# "5번만 울어볼래?" 같은 자연어 요청을 cry_count=5 같은 함수 인자로 바꿔 실행한 뒤 결과를 대화 맥락으로 가져옵니다.
# 스키마가 있어도 함수 이름, 설명, 로직을 잘 설계하는 일은 여전히 개발자의 몫입니다.
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

# 1. 모델 준비
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 2. 툴 리스트 준비 (우리가 만든 도구들)
tools = [cry, poo, eat]

# 3. 에이전트 생성 (자동 스키마 통합)
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=(
        "당신은 갓 태어난 아기 에이전트입니다. "
        "배가 고프거나 불만이 있으면 도구를 사용해 표현하세요."
    )
)

# 4. 실행 테스트: "5번 울어줘!"
response = agent.invoke({
    "messages": [{"role": "user", "content": "너 기분이 안 좋아 보이네? 5번만 울어볼래?"}]
})

# 결과의 messages 리스트에서 마지막 메시지가 아기 에이전트의 최종 답변입니다.
print(response["messages"][-1].content)
