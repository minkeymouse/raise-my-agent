"""
[2장 4절] [단기 메모리 & 체크포인트] 에이전트가 대화의 맥락을 기억하도록 하기

이 절에서 배우는 것:
- 채팅 모델은 요청 하나를 처리하면 이전 대화를 잊는 무상태(Stateless)입니다. LangGraph는 체크포인트로 이를 해결합니다.
- create_agent()에 MemorySaver를 checkpointer로 넣으면, 게임의 세이브 포인트처럼 에이전트의 상태(메시지 등)가
  자동으로 저장됩니다.
- thread_id는 대화방의 고유 번호입니다. "mom"과 "dad"로 대화방을 나누어 기억이 대화방에 귀속되는 모습을 확인합니다.

실행 방법:
    uv run python ch02_first_steps/04_memory_checkpoint.py
필요한 환경 변수: OPENAI_API_KEY (리포지토리 루트의 .env)

표시: [보충] 실행을 위해 더한 코드, [수정] 책 코드의 오류를 고친 곳, [설명용 코드] 실행되지 않는 설명용 조각
"""

# .env 파일의 API 키를 환경 변수로 불러옵니다.
from dotenv import load_dotenv
load_dotenv()


# %% [보충] 2절에서 만든 모델과 도구 준비
# [보충] 책에서 "(그래프/에이전트 정의 과정 생략)"으로 넘어간 llm과 tools입니다.
#        1절의 임포트와 2절의 스키마, 도구, 모델 코드를 책 그대로 가져왔습니다.
import os
import random

# [보충] 도구가 만드는 cry.txt, poo.txt가 이 장 폴더(ch02_first_steps/)에 생기도록 작업 디렉토리를 옮깁니다.
from pathlib import Path

if "__file__" in globals():  # 주피터 커널로 셀 단위 실행할 때는 __file__이 없으므로 건너뜁니다.
    os.chdir(Path(__file__).parent)

# [보충] 2-2절 "2. Pydantic으로 규칙 정의하기"
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

# [보충] 2-2절 "3. 도구에 스키마 장착하기"
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

# [보충] 2-2절 "4. 입력이 없는 도구 (eat)"
@tool("eat", description="맘마를 먹습니다. food.txt를 비웁니다.")
def eat() -> str:
    # ... (1절과 동일) ...
    return "냠냠"

# [보충] 2-2절 "5. 아기 에이전트에게 툴 스키마 장착하기"
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

# 1. 모델 준비
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 2. 툴 리스트 준비 (우리가 만든 도구들)
tools = [cry, poo, eat]


# %% 3. 가장 쉬운 기억법: MemorySaver
# 체크포인트는 에이전트가 행동을 마칠 때마다 현재 상태(메시지, 변수 등)를 스냅샷처럼 저장소에 기록한 것입니다.
# MemorySaver는 개발 단계에서 가장 간편한 체크포인터로, 기억을 램(RAM)에 저장하므로 프로그램을 끄면 기억도 사라집니다.
# create_agent()의 checkpointer 인자로 넘겨 주기만 하면 에이전트에게 자동 저장 기능이 생깁니다.
# v0에서는 RunnableWithMessageHistory 같은 래퍼를 씌워야 했지만, v1에서는 그래프 자체가 상태를 가지므로 이것으로 충분합니다.
from langgraph.checkpoint.memory import MemorySaver

# 1. 기억 저장소(세이브 파일 관리자) 준비
memory = MemorySaver()

# ... (그래프/에이전트 정의 과정 생략) ...

# 2. 에이전트(그래프)를 만들 때 저장소 주기
agent = create_agent(
    model=llm,
    tools=tools,
    checkpointer=memory, # 체크포인터를 인자로 넣어줍니다!
)


# %% 4. 너 누구랑 얘기하고 있니? : 스레드(Thread)로 사용자 구별하기 - 엄마와의 대화
# 실행할 때 config의 "configurable"에 thread_id를 넣어 대화방을 정합니다. 체크포인트는 thread_id별로 따로 저장되고,
# thread_id가 같아야 그 대화방의 상태와 기록을 복원할 수 있습니다. 이 대화는 "mom" 대화방에 저장됩니다.
# 참고: 아래 'AI:' 주석은 책에 실린 예시 답변입니다. 이 코드에는 엄마가 밥을 주는 단계가 없어 실제 답변은 다를 수 있습니다.
# 엄마와의 대화방
config_mom = {
    "configurable": {
        "thread_id": "mom"
    }
}

# 엄마가 질문
response1 = agent.invoke(
    {"messages": [("user", "나는 엄마 에이전트야. 밥은 먹었니?")]},
    config=config_mom
)

print(response1["messages"][-1].content)
# AI: "으앙~ 구구~! 배고파~"


# %% 4. 너 누구랑 얘기하고 있니? - 아빠와의 대화
# thread_id가 "dad"인 새 대화방입니다. 엄마와 나눈 기억은 꺼내지 않고 대화를 처음부터 다시 시작합니다.
# 아빠와의 대화방
config_dad = {
    "configurable": {
        "thread_id": "dad"
    }
}

response2 = agent.invoke(
    {"messages": [("user", "나는 아빠 에이전트야. 밥은 먹었니?")]},
    config=config_dad
)

print(response2["messages"][-1].content)
# AI: "아직 밥을 먹지 않았습니다."


# %% 4. 너 누구랑 얘기하고 있니? - 다시 엄마와의 대화
# 같은 config_mom으로 부르면 체크포인터가 "mom" 대화방의 이전 메시지를 자동으로 복원합니다.
# 이번에는 "밥은 먹었니?"만 보내지만, 에이전트는 앞서 엄마와 나눈 대화까지 이어서 보고 답합니다.
# 이처럼 기억은 대화방(thread_id)에 귀속됩니다.
# 다시 엄마와의 대화
response3 = agent.invoke(
    {"messages": [("user", "밥은 먹었니?")]},
    config=config_mom
)

print(response3["messages"][-1].content)
# AI: "밥 먹었다고 했잖아요, 엄마!"


# %% 5. 꺼지지 않는 기억 (Persistence)
# MemorySaver의 기억은 코드가 끝나면 사라지므로, 기억을 유지하려면 메모리 대신 데이터베이스나 파일에 저장합니다.
# 직접 만든 체크포인터는 저장하고 불러오는 로직을 손수 구현해야 해서, 보통은 PostgresSaver, SqliteSaver 같은 DB용 저장소를 씁니다.
# [설명용 코드] 로컬 파일에 JSON으로 저장하는 원리를 보여 주는 예시입니다.
#               get_latest_checkpoint의 본문이 생략되어 그대로는 문법 오류가 나므로 주석으로 남겼습니다.
#
# import json, os
# from langgraph.checkpoint.base import BaseCheckpointSaver
#
# class FileCheckpointSaver(BaseCheckpointSaver):
#     """파일에 대화 내용을 저장하는 저장소"""
#     def put_checkpoint(self, config, checkpoint):
#         # thread_id를 파일명으로 사용하여 저장
#         thread_id = config["configurable"]["thread_id"]
#         with open(f"{thread_id}.json", "w") as f:
#             json.dump(checkpoint, f)
#
#     def get_latest_checkpoint(self, config):
#         # 파일에서 불러오기
#         # ... (구현 생략) ...
