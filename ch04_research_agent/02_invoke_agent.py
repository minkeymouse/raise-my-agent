"""
[4장 2절] 자료조사 에이전트란? - 4. 에이전트 만들기 (invoke로 실행하기)

이 절에서 배우는 것:
- 완성한 에이전트(graph)는 invoke나 stream 메서드로 실행하며, 이때 Context를 함께 전달할 수 있습니다.
- BabyContext.create()로 현재 날짜를 담은 Context를 만들어 context 인자로 넘깁니다.
- config의 thread_id는 체크포인터가 대화를 저장하고 이어 가는 단위입니다.

실행 결과는 result에 담기만 하고 화면에 출력하지 않습니다.
실행 과정을 보려면 LangSmith 트레이싱을 켜거나(run_research_agent.py 머리말 참고) run_research_agent.py를 사용합니다.

실행 방법:
    uv run python ch04_research_agent/02_invoke_agent.py
필요한 환경 변수 (리포지토리 루트의 .env): OPENAI_API_KEY, NAVER_CLIENT_ID, NAVER_CLIENT_SECRET

표시: [보충] 실행에 필요한 코드, [수정] 실행에 맞게 고친 코드, [설명용 코드] 실행되지 않는 설명용 조각
"""

# .env 파일의 API 키를 환경 변수로 불러옵니다.
from dotenv import load_dotenv
load_dotenv()

# [보충] 4장 2절에서 만든 에이전트와 컨텍스트 스키마(agent/baby_research.py)를 불러옵니다.
from agent.baby_research import graph, BabyContext

# %% 4. 에이전트 만들기 - Context와 함께 실행하기
# 현재 시간으로 Context를 만들고 메시지와 함께 invoke에 넘깁니다.
# 동적 프롬프트(baby_research_prompt)가 이 context의 current_date를 읽어 시스템 프롬프트에 넣습니다.
# thread_id가 같으면 체크포인터에 저장된 앞의 대화에 이어서 실행됩니다.
context = BabyContext.create()
result = graph.invoke(
    {"messages": [{"role": "user", "content": "요즘 유행하는 티니핑에 대해 알려줘"}]},
    config={"configurable": {"thread_id": "1"}},
    context=context
)
