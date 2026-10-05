"""
[2장 1절] [Tool] 에이전트에게 팔다리를 달아주기

2장에서는 아기 에이전트에게 팔다리(Tool)와 사용 설명서(Tool Schema)를 달아 주고,
센스(ToolRuntime)와 기억(체크포인트)을 차례로 길러 줍니다. 1절은 그 첫 단계입니다.

이 절에서 배우는 것:
- LLM은 생각하는 뇌일 뿐 세상과 단절되어 있어, 파일을 쓰거나 API를 호출하는 행동에는 도구(Tool)가 필요합니다.
- 도구의 정체는 함수입니다. 함수 위에 @tool 데코레이터를 붙이면 LangGraph가 인식하는 도구가 됩니다.
- 파일에 흔적을 남기는 세 가지 도구(울기 cry, 응가하기 poo, 밥먹기 eat)를 만들고 invoke로 직접 테스트합니다.

실행 방법 (2장의 명령은 모두 리포지토리 루트에서 실행합니다):
    uv run python ch02_first_steps/01_tools.py
필요한 환경 변수: 없음 (LLM을 호출하지 않고 도구만 직접 실행합니다)
준비물 (2장 공통): 02_tool_schema.py와 04_memory_checkpoint.py는 OPENAI_API_KEY가 필요합니다.
    https://platform.openai.com/api-keys 에서 발급한 키를 리포지토리 루트의 .env 파일에 적어 둡니다.

실행하면 이 폴더(ch02_first_steps/)에 cry.txt가 생깁니다. food.txt가 없으면 밥먹기 테스트는
'맘마 없어'를 출력합니다.

표시: [보충] 실행에 필요한 코드, [수정] 실행에 맞게 고친 코드, [설명용 코드] 실행되지 않는 설명용 조각
"""

# %% 2. 환경 설정 및 라이브러리 임포트
# 툴을 만드는 데 필요한 라이브러리를 불러오고, load_dotenv로 .env 파일의 환경 변수를 읽어 옵니다.
# 에이전트가 필요로 하는 변수가 빠진 채로 실행하다 오류가 나는 경우가 많으므로 환경 변수 관리는 중요합니다.
# (BaseModel과 Field는 다음 절에서 도구의 스키마를 정의할 때 사용합니다.)
import os
import random
from typing import Literal, Optional
from dotenv import load_dotenv, find_dotenv

# LangChain/LangGraph 관련 임포트
from langchain.tools import tool
from pydantic import BaseModel, Field

# 환경 변수 로드 (.env)
load_dotenv(find_dotenv())

# [보충] 도구가 cry.txt, poo.txt, food.txt를 이 장 폴더(ch02_first_steps/)에서 읽고 쓰도록 작업 디렉토리를 옮깁니다.
from pathlib import Path

if "__file__" in globals():  # 주피터 커널로 셀 단위 실행할 때는 __file__이 없으므로 건너뜁니다.
    os.chdir(Path(__file__).parent)


# %% 3. 툴 정의하기: 1) 울기 (cry) & 2) 응가하기 (poo)
# @tool("이름", description="설명"): LLM은 이 이름과 설명을 읽고 언제 이 도구를 쓸지 결정합니다.
# 타입 힌트(Optional[int], Literal["큰응가", ...])는 LLM이 인자의 타입을 추론하는 데 도움을 줍니다.
# cry는 횟수가 없으면 1~10 사이에서 임의로 정하고, poo는 '큰응가'와 '작은응가' 중 하나만 고르게 합니다.
# 이처럼 도구를 어떻게 정의하느냐에 따라 에이전트의 결정권과 행동 양식이 달라집니다.
@tool("cry", description="아기처럼 웁니다. 디렉토리에 cry.txt 파일에 기록을 추가합니다.")
def cry(cry_count: Optional[int] = None) -> str:
    """우는 행동을 수행하고 파일에 기록합니다."""
    # 횟수가 지정되지 않으면 랜덤하게 설정 (1~10회)
    n = cry_count if isinstance(cry_count, int) and cry_count > 0 else random.randint(1, 10)

    # cry.txt 파일에 '응애' 기록 추가
    with open("cry.txt", "a") as f:
        f.write("응애" * n + "\n")

    return "울음 성공적으로 생성!"

@tool("poo", description="응가를 봅니다. 디렉토리에 poo.txt 파일에 기록을 추가합니다.")
def poo(poo_shape: Literal["큰응가", "작은응가"] = "큰응가", poo_count: Optional[int] = None) -> str:
    """응가 행동을 수행하고 파일에 기록합니다."""
    # 횟수가 지정되지 않으면 랜덤하게 설정
    c = poo_count if isinstance(poo_count, int) and poo_count > 0 else random.randint(1, 10)

    # poo.txt 파일에 응가 기록 추가
    with open("poo.txt", "a") as f:
        f.write(poo_shape * c + "\n")

    return f"{poo_shape} {c}개 응가 성공적으로 생성!"


# %% 3. 툴 정의하기: 3) 밥먹기 (eat)
# 입력 인자가 없는 도구입니다. food.txt에 내용이 있으면 읽고 비워서(섭취) 배고픔을 해소합니다.
# 파일이 없거나 비어 있으면 맘마가 준비되지 않은 상황으로 보고 "맘마 없어"를 돌려줍니다.
@tool("eat", description="맘마를 먹습니다. 디렉토리에 있는 food.txt 파일을 읽고 내용을 지웁니다. 없으면 배고픔 상태를 기록합니다.")
def eat() -> str:
    """음식을 먹는 행동을 수행합니다."""
    if os.path.exists("food.txt"):
        with open("food.txt", "r+") as f:
            content = f.read()
            # 파일 내용을 비웁니다 (먹었으니까요!)
            f.seek(0)
            f.truncate()

            if content == "":
                return "맘마 없어" # 파일은 있는데 빈 파일일 경우
            return "맘마를 먹었어요."
    else:
        return "맘마 없어" # 파일 자체가 없을 경우


# %% 4. 정의한 도구 테스트하기
# 도구도 채팅 모델처럼 invoke 메소드로 실행할 수 있습니다. 인자는 {"인자 이름": 값} 딕셔너리로 넘깁니다.
# 에이전트에게 쥐여 주기 전에 도구가 의도대로 동작하는지 확인해 두면, 나중에 오류가 났을 때 원인을 크게 좁힐 수 있습니다.
# 1. 울기 테스트 (3번 울기)
print(cry.invoke({"cry_count": 3}))

# 2. 밥먹기 테스트 (파일이 없으므로 '맘마 없어' 예상)
print(eat.invoke({}))

# 확인: 실제 디렉토리에 'cry.txt'가 생겼는지 확인해보세요!
