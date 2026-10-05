"""
[1장 2절] 채팅 모델

이 절에서 배우는 것:
    - 채팅 모델은 LLM을 랭체인에서 쓸 수 있게 모듈화한 객체로, 에이전트 실행의 주체(두뇌)입니다.
      다음 단어를 잇는 LLM(Completion)과 달리 대화의 흐름(Messages)과 자신의 역할(Role)에 맞춰 대답합니다.
    - init_chat_model에 '공급자:모델명' 문자열만 넘겨 모델을 불러오고, 문자열만 바꿔 모델을 교체합니다.
    - 표준 호출 메서드 invoke로 말을 걸고, 대답은 AIMessage의 content에서 꺼냅니다.
    - SystemMessage와 HumanMessage를 리스트로 넘겨 아기 에이전트에게 성격(Persona)을 부여합니다.

실행 방법:
    uv run python ch01_agent_birth/02_chat_model.py
필요한 환경 변수: OPENAI_API_KEY (주석의 Claude 모델로 바꿔 볼 때는 ANTHROPIC_API_KEY)
"""

# .env 파일의 API 키를 환경 변수로 불러옵니다.
from dotenv import load_dotenv
load_dotenv()

# %% 2. 모델은 어떻게 호출하나요? - 모델 불러오기: init_chat_model
# init_chat_model: '공급자:모델명' 문자열만으로 채팅 모델을 불러옵니다.
# 공급자(OpenAI, Anthropic 등)가 달라도 코드는 같고, 문자열만 바꾸면 배우(모델)를 교체할 수 있습니다.
# API 키는 공급자별 환경 변수(OPENAI_API_KEY, ANTHROPIC_API_KEY 등)를 그대로 사용합니다.
from langchain.chat_models import init_chat_model

# 1. 아기 에이전트의 두뇌를 깨웁니다. (OpenAI의 GPT-4o 사용)
# 미리 환경변수에 OPENAI_API_KEY가 설정되어 있어야 합니다.
model = init_chat_model("openai:gpt-4o")

# 만약 Claude로 뇌를 바꾸고 싶다면? 아래 한 줄이면 충분합니다!
# model = init_chat_model("anthropic:claude-3-5-sonnet-latest")
# 참고: Claude로 바꿀 때는 Anthropic 문서의 모델 목록에서 현재 모델명을 확인해 넣습니다(예: anthropic:claude-sonnet-4-6).

# %% 2. 모델은 어떻게 호출하나요? - 표준 호출 메서드: invoke
# 표준 호출 메서드는 invoke(단일 응답, AIMessage), stream(실시간 스트리밍, AIMessageChunk),
# batch(병렬 처리, List[AIMessage]) 세 가지입니다. 문자열 하나만 넘기면 HumanMessage에 담아 보냅니다.
# 이 model은 아직 도구가 연결되지 않은 채팅 모델이라 텍스트 응답만 돌려줍니다.
# 도구 호출 루프가 포함된 에이전트 실행은 2장의 create_agent부터 다룹니다.

# invoke()를 사용해 아기에게 말을 겁니다.
response = model.invoke("안녕 아가야? 오늘 기분이 어때?")

# 결과 확인 (AIMessage 객체 안의 content 속성에 대답이 들어있습니다)
print(f"아기 에이전트: {response.content}")

# [책의 실행 결과]
# 아기 에이전트: 응애! 엄마 안녕? 난 기분이 아주 좋아! (방긋)

# %% 3. 에이전트의 언어: 메시지(Message)
# 메시지는 '누가 말했는지(Role)'를 구분하는 구조화된 데이터 단위입니다.
# SystemMessage(지시)는 성격과 규칙을, HumanMessage(사용자)는 질문이나 명령을 담고, 대답은 AIMessage로 옵니다.
# 맥락을 전달할 때는 메시지가 오간 순서도 중요하므로 리스트로 넘깁니다.
# 이렇게 SystemMessage로 행동 양식을 정하는 것이 프롬프트 엔지니어링의 기초입니다.
from langchain_core.messages import SystemMessage, HumanMessage

# 1. 메시지 리스트 생성 (대화의 맥락 구성)
messages = [
    # 시스템 메시지로 아기의 성격을 부여합니다.
    SystemMessage(content="너는 갓 태어난 아기 에이전트야. 말끝마다 '응애'를 붙여야 해."),

    # 사용자가 말을 겁니다.
    HumanMessage(content="우리 아기, 배고프지 않니?"),
]

# 2. 모델에게 메시지 리스트 전달 (생각하고 답하기)
response = model.invoke(messages)

# 3. 대답 출력
print(response.content)

# [책의 실행 결과]
# 아니요 엄마, 아직 배 안 고파요 응애!
