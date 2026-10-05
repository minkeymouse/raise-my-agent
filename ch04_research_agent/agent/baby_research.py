"""
[4장 2절] 자료조사 에이전트란? - 메인 에이전트 파일 (agent/baby_research.py)

이 절에서 배우는 것:
- 실행 중에 바뀌는 데이터는 State(BabyState)로, 실행을 시작할 때 넘기는 고정 데이터는 Context(BabyContext)로 나눕니다.
- 네이버 검색 도구 세 개와 @tool로 만든 보고서 저장 도구를 한 에이전트에 연결합니다.
- 기본 미들웨어(작업 계획, 호출 제한, 대화 요약, 사람의 승인)와 @dynamic_prompt 커스텀 미들웨어를 조합합니다.
- create_agent로 모델, 도구, 스키마, 체크포인터, 미들웨어를 하나의 그래프(graph)로 묶습니다.

이 파일은 직접 실행하지 않습니다. run_research_agent.py, 02_invoke_agent.py, LangGraph Studio가 불러와 씁니다.
langgraph.json은 LangGraph CLI가 에이전트를 찾도록 경로를 지정하는 설정 파일입니다. 이 폴더의 langgraph.json은
이 파일의 studio_graph를 baby_research라는 이름으로 등록하고, 루트의 .env("../.env")를 읽게 합니다.

실행 방법:
    uv run python ch04_research_agent/run_research_agent.py   (파일을 통한 실행, 4장 1절)
    cd ch04_research_agent && uv run langgraph dev             (LangGraph Studio, 4장 1절)
langgraph dev는 langgraph.json이 있는 ch04_research_agent 폴더 안에서 실행합니다. 서버가 시작되면 세 주소가 나옵니다.
    API (http://127.0.0.1:2024): HTTP 요청으로 에이전트를 실행하는 API 엔드포인트
    Studio UI: 브라우저에서 그래프 구조를 보고, 대화하고, Human-in-the-Loop 승인도 처리하는 웹 인터페이스
    API Docs (http://127.0.0.1:2024/docs): 사용 가능한 엔드포인트와 파라미터를 보여 주는 문서
Studio UI 주소를 열고 그래프 목록에서 baby_research를 선택합니다. 서버는 Ctrl+C로 멈춥니다.
Safari에서 Studio가 로컬 서버에 연결되지 않으면 Chrome을 쓰거나 uv run langgraph dev --tunnel로 실행합니다.

필요한 환경 변수 (리포지토리 루트의 .env):
    OPENAI_API_KEY, NAVER_CLIENT_ID, NAVER_CLIENT_SECRET (선택: LANGSMITH_API_KEY)
준비물:
    - NAVER Developers(https://developers.naver.com/main/)에서 계정을 만들고 애플리케이션을 등록해(사용 API: 검색)
      클라이언트 아이디와 시크릿을 발급받습니다.
    - Studio는 LangSmith(https://smith.langchain.com)의 웹 인터페이스이므로 LangSmith 계정이 필요합니다.
    - 책처럼 패키지를 직접 설치한다면 langgraph-cli 대신 "langgraph-cli[inmem]"을 설치해야 langgraph dev가 실행됩니다.
      이 리포지토리는 루트에서 uv sync를 한 번 실행하면 모두 설치됩니다.

표시: [보충] 실행을 위해 더한 코드, [수정] 책 코드의 오류를 고친 곳, [설명용 코드] 실행되지 않는 설명용 조각
"""

# .env 파일의 API 키를 환경 변수로 불러옵니다.
# (네이버 검색 도구와 채팅 모델을 이 파일을 불러오는 순간 만들기 때문에, 키를 먼저 불러와야 합니다.)
from dotenv import load_dotenv
load_dotenv()

# %% 1. 스키마 정의 - 상태 스키마
# State는 실행 중에 바뀌는 동적 데이터입니다. AgentState를 상속하면 messages 키가 이미 들어 있으므로
# 검색 횟수를 추적할 search_count만 더합니다.
# NotRequired: 선택 필드라는 뜻입니다. 처음 실행할 때 값을 주지 않아도 에이전트가 정상적으로 동작합니다.
from langchain.agents import AgentState
from typing_extensions import NotRequired

class BabyState(AgentState):
    """아기 리서치 에이전트의 상태."""
    # messages는 AgentState에서 상속됨
    search_count: NotRequired[int]  # 검색 횟수 추적 등 추가 가능

# %% 1. 스키마 정의 - 컨텍스트 스키마
# Context는 실행을 시작할 때 전달하는 불변 데이터입니다(사용자 ID, 데이터베이스 연결, 현재 시간 등).
# 현재 날짜를 Context로 넘기면 에이전트가 '최근'이 언제인지 분명하게 알 수 있습니다.
# default_factory는 Context를 만들 때 기본값을 동적으로 정합니다. 매번 실행해야 하는 간단한 함수는
# 도구로 만드는 대신 create()처럼 클래스 메서드로 정의하는 편이 낫습니다.
from dataclasses import dataclass, field
from datetime import datetime
from zoneinfo import ZoneInfo

def _get_current_date_str() -> str:
    """현재 날짜/시간 문자열 생성 (KST)."""
    kst = ZoneInfo('Asia/Seoul')
    current_time = datetime.now(kst)
    return current_time.strftime("%Y-%m-%d %H:%M:%S KST")

@dataclass
class BabyContext:
    """아기 리서치 에이전트의 컨텍스트."""
    current_date: str = field(default_factory=_get_current_date_str)

    @classmethod
    def create(cls) -> "BabyContext":
        """현재 시간으로 컨텍스트 생성."""
        return cls(current_date=_get_current_date_str())

# %% 2. 도구 정의 - 검색 도구
# 자료조사 에이전트의 핵심은 여러 검색 도구입니다. 블로그, 뉴스, 일반 웹 검색은 서로 다른 소스에서
# 정보를 가져오므로 에이전트가 상황에 맞는 도구를 골라 씁니다.
# NaverSearchAPIWrapper는 환경 변수 NAVER_CLIENT_ID, NAVER_CLIENT_SECRET으로 네이버 검색 API를 호출합니다.
# 모델에게 보이는 도구 이름은 차례로 naver_blog_search, naver_news_search, naver_search_results_json입니다.
from langchain_naver_community.tool import (
    NaverBlogSearch,
    NaverNewsSearch,
    NaverSearchResults,
)
from langchain_naver_community.utils import NaverSearchAPIWrapper

# Naver 검색 API 래퍼 초기화 (모든 도구에서 공유)
naver_api_wrapper = NaverSearchAPIWrapper()

# 모든 Naver 검색 도구 초기화
naver_tools = [
    NaverBlogSearch(api_wrapper=naver_api_wrapper),      # 블로그 포스트
    NaverNewsSearch(api_wrapper=naver_api_wrapper),      # 뉴스 기사
    NaverSearchResults(api_wrapper=naver_api_wrapper),   # 일반 웹 검색
]

# %% 2. 도구 정의 - 파일 생성 도구
# @tool 데코레이터는 평범한 파이썬 함수를 에이전트가 쓸 수 있는 도구로 바꿉니다.
# 도구의 docstring은 모델이 도구를 이해하는 설명서입니다. 목적, 인자, 반환값을 분명히 적을수록 정확하게 씁니다.
# Path(__file__).parent.parent는 agent/의 상위 폴더이므로, 보고서는 ch04_research_agent/output/report.md에 저장됩니다.
from langchain.tools import tool
from pathlib import Path

@tool
def create_report_markdown(content: str) -> str:
    """리서치 결과를 report.md 파일로 저장합니다.

    사용자가 보고서 생성을 요청하거나, 리서치 결과를 정리한 후 
    마크다운 형식의 보고서로 저장하고 싶을 때 사용하세요.

    Args:
        content: 저장할 마크다운 형식의 보고서 내용
                  (제목 포함, 마크다운 문법 사용)

    Returns:
        저장 완료 메시지
    """
    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(exist_ok=True)

    file_path = output_dir / "report.md"
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

    return f"보고서가 {file_path}에 저장되었습니다."

# 모든 도구를 하나의 리스트로 결합
all_tools = naver_tools + [create_report_markdown]

# %% 3. 미들웨어 정의 - 기본 미들웨어
# TodoListMiddleware: 작업을 시작하기 전에 계획을 세우고 진행 과정을 추적합니다(write_todos 도구를 더합니다).
# SummarizationMiddleware: 대화 기록이 너무 길어지면 자동으로 요약해 컨텍스트를 관리합니다.
# ModelCallLimitMiddleware / ToolCallLimitMiddleware: 호출 횟수를 제한해 비용을 관리하고 무한 루프를 막습니다.
# HumanInTheLoopMiddleware: 파일 생성 같은 민감한 작업 전에 사용자 승인을 요청합니다(4. 에이전트 만들기에서 사용).
from langchain.agents.middleware import (
    TodoListMiddleware,
    HumanInTheLoopMiddleware,
    ModelCallLimitMiddleware,
    ToolCallLimitMiddleware,
    SummarizationMiddleware,
)

# 참고: 이 middleware 리스트는 인자를 주는 방법을 보여 주는 예시이고, 에이전트에는 4.의 middleware=[...]가 들어갑니다.
middleware = [
    TodoListMiddleware(),
    # thread_limit은 한 대화(thread_id) 전체, run_limit은 한 번의 실행에서 허용하는 호출 횟수입니다.
    # exit_behavior="end": 한도를 넘으면 오류를 내지 않고, 한도 초과를 알리는 메시지를 남긴 뒤 실행을 끝냅니다.
    ModelCallLimitMiddleware(
        thread_limit=10,
        run_limit=5,
        exit_behavior="end",
    ),
    # tool_name으로 지정한 도구 하나의 호출 횟수만 제한합니다.
    ToolCallLimitMiddleware(
        tool_name="naver_blog_search",
        thread_limit=10,
        run_limit=5,
    ),
    # 대화가 4000토큰에 이르면 gpt-4o-mini로 앞부분을 요약하고, 최근 메시지 20개는 그대로 남깁니다.
    # 요약을 시작할 토큰 수나 도구 호출 한도는 에이전트를 직접 써 보면서 알맞은 값을 찾아 갑니다.
    SummarizationMiddleware(
        model="gpt-4o-mini",
        trigger=("tokens", 4000),
        keep=("messages", 20),
    ),
]

# %% 3. 미들웨어 정의 - 커스텀 미들웨어
# @dynamic_prompt: 모델을 호출할 때마다 시스템 프롬프트를 새로 만드는 커스텀 미들웨어입니다.
# request.runtime.context로 실행할 때 넘긴 Context(BabyContext)를 읽어 현재 날짜를 프롬프트에 넣습니다.
# 여기서는 날짜와 시간만 바뀌지만, 더 많은 정보를 넣도록 응용할 수 있습니다.
from langchain.agents.middleware import dynamic_prompt, ModelRequest

@dynamic_prompt
def baby_research_prompt(request: ModelRequest) -> str:
    """Context에서 현재 날짜를 읽어서 프롬프트에 포함."""
    current_date = request.runtime.context.current_date

    return f"""너는 아기 리서치 에이전트야. 아기의 말투를 써서 꼭 의성어를 포함해서 대답해야 해.

현재 날짜와 시간: {current_date}

너의 역할:
1. 사용자가 하는 질문을 이해해.
2. 최근, 현재, 최신 정보에 대한 질문을 받으면 위의 현재 날짜를 참고해서 
   '최근'이 언제인지 파악하고, 검색 쿼리에 현재 연도나 날짜 컨텍스트를 포함해야 해.
3. 적절한 Naver 검색 도구를 사용하여 관련 정보를 수집해:
   - 블로그 검색: 의견, 경험, 상세한 기사에 사용
   - 뉴스 검색: 최근 뉴스와 현재 이벤트에 사용
   - 일반 검색: 포괄적인 웹 결과에 사용
4. 최근 정보를 검색할 때는 검색 쿼리에 현재 연도나 날짜 컨텍스트를 포함해.
5. 발견한 내용을 명확하고 포괄적인 답변으로 종합해.
6. 주장할 때는 출처를 인용하고 정보가 얼마나 최신인지 표시해.
7. 사용자가 보고서를 요청하거나, 리서치 결과를 정리해서 파일로 저장해야 할 때는
   create_report_markdown 도구를 사용하여 report.md 파일을 생성해.
   보고서에는 제목, 주요 내용, 출처를 마크다운 형식으로 작성해.

초기 검색으로 충분한 정보를 얻지 못했다면,가장 적절한 검색 도구를 사용하여 개선된 쿼리로 추가 검색을 수행해."""

# %% 4. 에이전트 만들기
# create_agent는 모델, 도구, 미들웨어, State와 Context 스키마를 받아 완성된 에이전트(그래프)를 만듭니다.
# 체크포인터(MemorySaver)를 넣어야 thread_id별로 대화를 저장하고 이어서 대화할 수 있습니다.
# 미들웨어는 3.에서 본 기본 미들웨어에 동적 프롬프트와 사람의 승인, 검색 도구별 호출 제한을 더한 구성입니다.
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver

model = ChatOpenAI(model="gpt-4o", temperature=0)

# 체크포인터: thread_id별로 대화 상태 저장 → 이어서 대화 가능
checkpointer = MemorySaver()

graph = create_agent(
    model=model,
    tools=all_tools,  # 검색 도구 + 파일 생성 도구
    state_schema=BabyState,
    context_schema=BabyContext,
    checkpointer=checkpointer,
    middleware=[
        baby_research_prompt,  # 동적 프롬프트
        TodoListMiddleware(),
        # interrupt_on: 도구 이름마다 호출 전에 승인을 받을지 정합니다. False는 승인 없이 자동 실행합니다.
        HumanInTheLoopMiddleware(
            interrupt_on={
                "naver_blog_search": False,
                "naver_news_search": False,
                "naver_search_results": False,
                "create_report_markdown": False,
            },
        ),
        ModelCallLimitMiddleware(
            thread_limit=10,
            run_limit=5,
            exit_behavior="end",
        ),
        ToolCallLimitMiddleware(
            tool_name="naver_blog_search",
            thread_limit=10,
            run_limit=5,
        ),
        ToolCallLimitMiddleware(
            tool_name="naver_news_search",
            thread_limit=10,
            run_limit=5,
        ),
        # 참고: 일반 웹 검색 도구의 실제 이름은 naver_search_results_json이라, 이 설정은 오류 없이 실행되지만
        # 일반 웹 검색의 호출 횟수는 제한하지 못합니다. 제한하려면 이름을 "naver_search_results_json"으로 씁니다.
        ToolCallLimitMiddleware(
            tool_name="naver_search_results",
            thread_limit=10,
            run_limit=5,
        ),
        SummarizationMiddleware(
            model="gpt-4o-mini",
            trigger=("tokens", 4000),
            keep=("messages", 20),
        ),
    ],
)

# %% [4-3] 2. 보고서 생성하기 - 승인이 필요한 도구 지정하기
# 지금 설정(모두 False)에서는 모든 도구가 자동으로 실행되지만, 민감하거나 비용이 큰 작업은 승인을 받게 할 수 있습니다.
# 아래처럼 설정하면 해당 도구를 호출하기 전에 실행이 멈추고, 승인(approve), 수정(edit), 거절(reject)을 고를 수 있습니다.
# 승인 요청은 LangGraph Studio에서 처리합니다(run_research_agent.py에는 승인을 처리하는 코드가 없습니다).
# [설명용 코드] 위 create_agent 안의 HumanInTheLoopMiddleware를 이 설정으로 바꿔 씁니다.
# HumanInTheLoopMiddleware(
#     interrupt_on={
#         "naver_blog_search": True,  # 승인 요구
#         "naver_news_search": True,  # 승인 요구
#         "naver_search_results": False,  # 자동 실행
#         "create_report_markdown": True,  # 승인 요구
#     },
# )

# %% [보충] LangGraph Studio용 그래프
# langgraph dev는 체크포인터가 들어 있는 graph를 불러오지 못하므로("... includes a custom checkpointer" 오류),
# 체크포인터만 뺀 그래프를 langgraph.json에 baby_research로 등록합니다. Studio에서는 서버가 대화 상태를 저장합니다.
studio_graph = graph.copy(update={"checkpointer": None})
