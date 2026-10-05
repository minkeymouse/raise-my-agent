"""
[10장 3절] 보고서 작성하기 - 4. 체크포인터 연결 (선택)

이 절에서 배우는 것:
- 체크포인터(MemorySaver)를 추가하면 중간에 실행이 중단되어도 이어서 작업할 수 있습니다.
- config의 thread_id("research-001")로 세션을 관리합니다. 책은 같은 thread_id로 다시 실행하면 이전 조사 결과가 유지되어
  "어제 조사한 내용에 이어서 추가 분석해줘" 같은 요청이 가능해진다고 설명합니다.

실행 방법 (리포지토리 루트에서):
    uv run python ch10_intern_agent/03_checkpointer.py
노드들이 출력하는 진행 메시지만 나오고, 보고서는 result["report"]에 담길 뿐 파일로 저장하지 않습니다.
필요한 환경 변수: OPENAI_API_KEY, NAVER_CLIENT_ID, NAVER_CLIENT_SECRET
(키 발급 방법과 Studio 실행 방법은 agent/deep_research.py의 머리말에 있습니다.)

표시: [보충] 실행을 위해 더한 코드, [수정] 책 코드의 오류를 고친 곳, [설명용 코드] 실행되지 않는 설명용 조각
"""

# .env 파일의 API 키를 환경 변수로 불러옵니다.
from dotenv import load_dotenv
load_dotenv()

# %% [보충] 3절에서 만든 그래프와 초기 State 준비
# [보충] 3절 '2. 그래프 조립'의 build_research_graph를 agent/deep_research.py에서 가져옵니다.
from agent.deep_research import build_research_graph

# [보충] 책의 코드가 쓰는 initial_state는 run_deep_research() 안에만 있어 그대로 실행하면 NameError가 납니다.
#        같은 값을 여기에 정의합니다. (user_input은 책의 실행 예시 문장)
user_input = "AI 반도체 시장 동향에 대해 심층 보고서를 작성해줘"
initial_state = {
    "user_input": user_input,
    "topic": "",
    "depth": "",
    "target_audience": "",
    "questions": [],
    "search_results": [],
    "news_results": [],
    "categories": [],
    "insights": [],
    "gaps": [],
    "report": "",
    "iteration": 0,
    "max_iterations": 2,
}

# %% [10-3] 4. 체크포인터 연결 (선택)
# MemorySaver는 그래프의 State를 메모리에 저장하는 체크포인터이고, config의 thread_id가 세션을 구분하는 단위입니다.
# 참고: 책의 코드는 app_with_memory = app이라 체크포인터가 실제로 연결되지 않아 State가 저장되지 않습니다.
#       실제로 저장하려면 책의 주석대로 그래프를 컴파일할 때 compile(checkpointer=memory)로 넘겨야 합니다.
from langgraph.checkpoint.memory import MemorySaver

# 체크포인터 추가
memory = MemorySaver()
app = build_research_graph()
app_with_memory = app  # compile 시 checkpointer=memory 추가 가능

# thread_id로 세션 관리
config = {"configurable": {"thread_id": "research-001"}}
result = app_with_memory.invoke(initial_state, config=config)
