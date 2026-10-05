"""
[10장 3절] 보고서 작성하기 - 4. 체크포인터 연결 (선택)

이 절에서 배우는 것:
- 체크포인터(MemorySaver)를 추가하면 중간에 실행이 중단되어도 이어서 작업할 수 있습니다.
- config의 thread_id("research-001")로 세션을 관리합니다. 같은 thread_id로 다시 실행하면 이전 조사 결과가 유지되어
  "어제 조사한 내용에 이어서 추가 분석해줘" 같은 요청이 가능해집니다.

실행 방법 (리포지토리 루트에서):
    uv run python ch10_intern_agent/03_checkpointer.py
실행하면 노드들의 진행 메시지가 출력되고, 보고서는 파일로 저장되지 않고 result["report"]에 담깁니다.
필요한 환경 변수: OPENAI_API_KEY, NAVER_CLIENT_ID, NAVER_CLIENT_SECRET
(키 발급 방법과 Studio 실행 방법은 agent/deep_research.py의 머리말에 있습니다.)

표시: [보충] 실행에 필요한 코드, [수정] 실행에 맞게 고친 코드, [설명용 코드] 실행되지 않는 설명용 조각
"""

# .env 파일의 API 키를 환경 변수로 불러옵니다.
from dotenv import load_dotenv
load_dotenv()

# %% [보충] 3절에서 만든 그래프와 초기 State 준비
# [보충] agent/deep_research.py의 build_research_graph (3절 '2. 그래프 조립')
from agent.deep_research import build_research_graph

# [보충] run_deep_research()와 같은 초기 State (요청 문장은 3절의 실행 예시)
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
# 체크포인터는 그래프를 컴파일할 때 compile(checkpointer=memory)처럼 넘겨 연결합니다.
from langgraph.checkpoint.memory import MemorySaver

# 체크포인터 추가
memory = MemorySaver()
app = build_research_graph()
app_with_memory = app  # compile 시 checkpointer=memory 추가 가능

# thread_id로 세션 관리
config = {"configurable": {"thread_id": "research-001"}}
result = app_with_memory.invoke(initial_state, config=config)
