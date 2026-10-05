"""
[1장 3절] [Messages] 에이전트와의 대화법

이 절에서 배우는 것:
    - 같은 문자열도 누가(Role) 말했는지에 따라 의미가 달라지므로, 랭체인은 대화의 단위를 메시지 객체로 정의합니다.
    - 네 가지 메시지 타입: SystemMessage(세계관), HumanMessage(자극), AIMessage(생각과 답변), ToolMessage(경험)
    - AIMessage의 tool_calls(행동 신호)와, 콘텐츠 블록으로 텍스트와 이미지를 함께 담는 멀티모달 메시지를 살펴봅니다.
    - 메시지를 순서대로 쌓은 리스트로 대화 흐름을 전달하고, pretty_print()로 보기 좋게 출력합니다.

실행 방법:
    uv run python ch01_agent_birth/03_messages.py
필요한 환경 변수: OPENAI_API_KEY

표시: [보충] 실행을 위해 더한 코드, [수정] 책 코드의 오류를 고친 곳, [설명용 코드] 실행되지 않는 설명용 조각
"""

# .env 파일의 API 키를 환경 변수로 불러옵니다.
from dotenv import load_dotenv
load_dotenv()

# %% 2. 대화의 참여자들: 메시지 타입 (Message Types)
# 한 번의 대화에서 메시지가 쌓이는 순서입니다: 지침 -> 사용자 입력 -> AI의 도구 호출 -> 도구 결과 -> 최종 답변.
# AIMessage는 말(content)과 함께, 도구를 쓰겠다는 행동 신호를 tool_calls 속성에 정형화된 데이터로 담습니다.
# ToolMessage는 사용자도 AI도 아닌 외부 세계의 사실(도구 실행 결과)입니다. 실제로는 도구를 호출한 뒤
# 런타임이 채워 넣으며, AIMessage의 tool_calls와 서로 대응해야 모델이 이어서 추론할 수 있습니다.
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

# 에이전트의 대화 기억(Memory)은 이렇게 쌓여갑니다.
messages = [
    SystemMessage(content="너는 팩트만 말하는 비서다."),   # 지침(세계관)
    HumanMessage(content="지금 환율 얼마야?"),            # 자극(사용자 입력)
    AIMessage(content="잠시만요, 검색해볼게요...",          # 생각/행동(AI의 1차 반응)
              # [수정] 책 본문의 설명대로 도구 호출에 args와 id를 더했습니다(없으면 TypeError). (책: 아래 주석 줄)
              #   tool_calls=[{"name": "search_rate"}]),
              tool_calls=[{"name": "search_rate", "args": {}, "id": "call_1"}]),
    # [수정] 위 도구 호출과 같은 tool_call_id를 더했습니다(없으면 KeyError). (책: 아래 주석 줄)
    # ToolMessage(content="1달러 = 1430원"),                 # 경험(외부 도구 실행 결과)
    ToolMessage(content="1달러 = 1430원", tool_call_id="call_1"),  # 경험(외부 도구 실행 결과)
    AIMessage(content="현재 환율은 1430원입니다.")          # 최종 답변
]

# %% 3. 메시지의 해부: 속성(Attributes)과 구조 - content와 content_blocks
# 콘텐츠 블록(Content Blocks): content에 문자열 대신 {"type": ...} 딕셔너리 리스트를 넣어 텍스트, 이미지 등을 나눕니다.
# 블록 단위로 나누면 모델이 입력 종류를 구분하기 쉽고, 오류가 난 조각을 추적하기도 좋습니다.
# 이미지는 url로 첨부할 수도, 파일을 직접 첨부할 수도 있습니다. 비전(이미지)을 지원하는 모델이 필요하고,
# 공급자마다 블록 형식이 조금씩 다를 수 있습니다. 이 셀은 메시지를 만들기만 하고 모델에 보내지는 않습니다.
from langchain_core.messages import HumanMessage

# 텍스트와 이미지를 동시에 '보는' 메시지
msg = HumanMessage(
    content=[
        {"type": "text", "text": "이 사진 속에 고양이가 몇 마리야?"},
        {"type": "image_url", "image_url": {"url": "https://example.com/cat.jpg"}}
    ]
)

# %% 4. 메시지 모아보기: 대화의 흐름 (List of Messages)
# 메시지를 순서대로 쌓은 리스트를 넘기면, 모델은 리스트 전체를 앞에서부터 읽고 이전 대화의 맥락을 파악합니다.
# 순서를 바꾸면 같은 문장이라도 해석 맥락이 달라질 수 있습니다. 랭그래프에서는 이 리스트가 상태(State)의 중심이 되며,
# 너무 많이 쌓이면 자원을 초과할 수 있어 요약, 잘라 내기 같은 컨텍스트 엔지니어링을 책의 뒷부분에서 다룹니다.

# [보충] 1-2절에서 만든 채팅 모델 (이 파일만으로 실행되도록 다시 만듭니다)
from langchain.chat_models import init_chat_model
model = init_chat_model("openai:gpt-4o")

# 대화의 흐름: 시스템 지침 -> 사용자 질문 -> AI 대답 -> 다시 사용자 질문
history = [
    SystemMessage(content="당신은 수학 선생님입니다."),
    HumanMessage(content="피타고라스 정리가 뭐야?"),
    AIMessage(content="직각삼각형에서 빗변의 제곱은..."),
    HumanMessage(content="그럼 예시를 하나 들어줘.")
]

# 모델은 리스트 전체를 보고 '아, 앞에서 피타고라스 얘기를 했었지'라고 기억합니다.
response = model.invoke(history)

# %% 5. 내 아이의 속마음 들여다보기: 출력과 확인
# print(message)로는 내용(content)과 메타데이터가 뒤섞여 보기 어렵습니다.
# pretty_print()는 메시지 타입과 본문을 구분해 출력하므로, tool_calls가 붙은 AIMessage나
# 블록이 여러 개인 멀티모달 메시지를 볼 때 특히 유용합니다. 토큰 단위 스트리밍은 3장에서 다룹니다.

# model은 앞 절에서 만든 채팅 모델이라고 가정합니다.
response = model.invoke([HumanMessage(content="서울 날씨 한 문장으로 알려줘.")])
response.pretty_print()

# [책의 실행 결과] (예시, 모델에 따라 문장은 달라질 수 있음)
# ================================== AI Message ==================================
# 서울의 날씨는 맑음입니다.
# 참고: 실제 pretty_print()는 제목 줄을 'AI Message'가 아니라 'Ai Message'로 출력하고, 그 아래에 빈 줄을 넣습니다.
