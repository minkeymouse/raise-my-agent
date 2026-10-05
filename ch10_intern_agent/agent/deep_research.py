"""
[10장 1~3절] 목표설정하기 / 자료 조사하기 / 보고서 작성하기 - 심층 리서치 에이전트 (agent/deep_research.py)

1~3절에서 배우는 것:
- 1장~9장에서 배운 State, 동적 프롬프트, 조건부 엣지와 루프 패턴을 모아 LangGraph 기반 심층 리서치 에이전트를 만듭니다.
- 워크플로우는 목표설정(goal_setting_node) → 조사(investigation_node) → 갭 체크(check_gaps) → 보고서 생성(generation_node)이고,
  조사 결과에 부족한 영역(갭)이 남아 있으면 조사 노드로 되돌아가는 조건부 루프가 핵심입니다.
- 1절은 State와 목표설정 노드, 2절은 검색 함수와 조사 노드, 갭 체크, 3절은 생성 노드와 그래프 조립, 실행을 다룹니다.
  세 절의 코드는 모두 이 파일 하나에 이어서 작성합니다. 3절 '4. 체크포인터 연결'은 03_checkpointer.py에 있습니다.

실행 방법 (리포지토리 루트에서):
    uv run python ch10_intern_agent/agent/deep_research.py     (3절 '1) 코드에서 직접 실행')
    cd ch10_intern_agent && uv run langgraph dev                (3절 '2) LangGraph Studio에서 실행')
직접 실행하면 보고서가 명령을 실행한 폴더의 research_report.md에 저장됩니다.
Studio: 터미널에 나오는 Studio UI 주소를 브라우저에서 열고 그래프 목록에서 deep_research를 고릅니다.
    입력은 user_input만 채웁니다(예: AI 반도체 시장 동향에 대해 심층 보고서를 작성해줘). 나머지 필드는 노드들이 채웁니다.
    그래프 구조와 조건부 루프, 노드마다 채워지는 State, 조사 노드의 반복 횟수, 최종 report 필드를 확인할 수 있습니다.
    Studio에서는 보고서가 파일로 저장되지 않습니다. 서버는 Ctrl+C로 멈춥니다.
    Safari에서 Studio가 로컬 서버에 연결되지 않으면 Chrome을 쓰거나 uv run langgraph dev --tunnel로 실행합니다.
    [수정] langgraph dev가 시작되려면 langgraph.json에 "dependencies": ["."]가 필요합니다. (책: "dependencies" 항목 없음)

필요한 환경 변수 (리포지토리 루트의 .env):
    OPENAI_API_KEY, NAVER_CLIENT_ID, NAVER_CLIENT_SECRET (선택: LANGSMITH_API_KEY)
준비물:
    - NAVER Developers(https://developers.naver.com/main/)에서 애플리케이션을 등록해(사용 API: 검색)
      클라이언트 아이디와 시크릿을 발급받습니다. 4장에서 발급받은 키를 그대로 써도 됩니다.
    - Studio는 LangSmith(https://smith.langchain.com) 웹사이트에서 열리므로 LangSmith 계정이 필요합니다.
    - 필요한 패키지는 리포지토리 루트에서 uv sync를 실행해 설치합니다. 패키지를 직접 설치할 때는
      langgraph dev를 실행할 수 있도록 "langgraph-cli[inmem]"을 설치합니다.

표시: [보충] 실행에 필요한 코드, [수정] 실행에 맞게 고친 코드, [설명용 코드] 실행되지 않는 설명용 조각
"""

# ==============================================================================
# [10장 1절] 목표설정하기
# 좋은 검색 도구가 있어도 무엇을 조사할지 불분명하면 에이전트는 방향을 잃습니다.
# 전체 워크플로우의 데이터 구조(State)를 정의하고, 사용자 요청을 분석해 리서치 목표와
# 조사 질문을 세우는 첫 번째 노드(goal_setting_node)를 만듭니다.
# ==============================================================================

# %% [10-1] 1. ResearchState 정의: 워크플로우의 데이터 뼈대
# LangGraph에서 가장 먼저 할 일은 State 정의입니다. State는 그래프의 모든 노드가 읽고 쓰는 공유 데이터 구조로,
# 1절(목표설정) → 2절(조사) → 3절(생성)을 거치며 점점 채워집니다. 필드마다 어느 절에서 채우는지 주석으로 적어 둡니다.
# iteration과 max_iterations는 "최대 N번까지만 재조사"라는 루프 탈출 조건에 쓰입니다.
# agent/deep_research.py

import os
import json
import requests
from typing import TypedDict, List, Optional, Annotated
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

# find_dotenv()는 이 파일의 폴더에서 상위 폴더로 올라가며 .env를 찾으므로, 리포지토리 루트의 .env를 읽습니다.
load_dotenv(find_dotenv())

# ── State 정의 ─────────────────────────────────────────────────

class ResearchState(TypedDict):
    # 사용자 입력
    user_input: str

    # 1절: 목표설정에서 채움
    topic: str
    depth: str                      # "개요", "상세", "심층"
    target_audience: str
    questions: List[str]            # 조사 질문 목록

    # 2절: 조사에서 채움
    search_results: List[dict]      # 웹 검색 결과
    news_results: List[dict]        # 뉴스 검색 결과
    categories: List[dict]          # 분류된 정보
    insights: List[str]             # 도출된 통찰
    gaps: List[str]                 # 정보 부족 영역

    # 3절: 생성에서 채움
    report: str                     # 최종 보고서 (마크다운)

    # 루프 제어 (7장 루프 패턴)
    iteration: int                  # 현재 조사 반복 횟수
    max_iterations: int             # 최대 반복 횟수


# %% [10-1] 2. LLM 준비
# 모든 노드에서 공유할 LLM을 한 번만 만들어 둡니다.
# 목표설정(goal_setting_node), 정보 분석(analyze_and_categorize), 보고서 생성(generation_node)이 모두 이 llm을 호출합니다.
# ── LLM 초기화 ─────────────────────────────────────────────────

llm = ChatOpenAI(model="gpt-4o", temperature=0.3)


# %% [10-1] 3. goal_setting_node 구현 - 1) 요청 분석 + 질문 생성을 한 번에 처리
# 사용자의 자연어 요청을 받아 State에 topic, depth, target_audience, questions를 채우는 첫 번째 노드입니다.
# 동적 프롬프트를 활용합니다. "AI 반도체 시장 동향"만 입력해도 LLM이 맥락을 파악해 "시장 규모는?", "주요 기업은?" 같은
# 구체적인 검색 질문을 만들고, 이 질문들이 2절 조사 노드에서 검색 쿼리로 그대로 쓰입니다.
# ── 1절: 목표설정 노드 ─────────────────────────────────────────

def goal_setting_node(state: ResearchState) -> dict:
    """
    사용자 요청을 분석하여 리서치 목표와 조사 질문을 설정하는 노드.

    [입력] state["user_input"]
    [출력] topic, depth, target_audience, questions, iteration, max_iterations
    """

    user_input = state["user_input"]

    # LLM에게 요청 분석 + 질문 생성을 함께 요청
    # (f-문자열이므로 프롬프트 안 JSON 예시의 중괄호는 {{ }}로 두 번 씁니다.)
    analysis_prompt = f"""
    사용자가 다음과 같은 리서치를 요청했습니다:

    "{user_input}"

    아래 JSON 형식으로 분석 결과를 반환하세요. 반드시 JSON만 반환하세요.

    {{
        "topic": "리서치 주제 (핵심 키워드 중심으로)",
        "depth": "개요/상세/심층 중 하나",
        "target_audience": "대상 독자",
        "questions": [
            "조사할 핵심 질문 1",
            "조사할 핵심 질문 2",
            "조사할 핵심 질문 3",
            "조사할 핵심 질문 4",
            "조사할 핵심 질문 5"
        ]
    }}

    questions는 반드시 5개를 생성하세요.
    검색 엔진에 바로 입력할 수 있는 구체적인 질문으로 작성하세요.
    """

    response = llm.invoke(analysis_prompt)

    # JSON 파싱
    # LLM이 ```json ... ``` 코드블록으로 감싸 답하는 경우가 있어, 걷어 낸 뒤 dict로 바꿉니다.
    content = response.content.strip()
    # 마크다운 코드블록 제거
    if content.startswith("```"):
        content = content.split("\n", 1)[1].rsplit("```", 1)[0]

    result = json.loads(content)

    print(f"\n 목표설정 완료")
    print(f"   주제: {result['topic']}")
    print(f"   깊이: {result['depth']}")
    print(f"   질문 {len(result['questions'])}개 생성됨")

    # 노드 함수의 입출력 규칙(1절 '2) 노드의 입출력 규칙'): 인자는 state 하나, 반환은 바꿀 필드만 담은 dict입니다.
    # 반환하지 않은 필드(search_results, report 등)는 그대로 유지됩니다.
    return {
        "topic": result["topic"],
        "depth": result["depth"],
        "target_audience": result["target_audience"],
        "questions": result["questions"],
        "iteration": 0,
        "max_iterations": 2,  # 최대 2번 재조사
    }


# ==============================================================================
# [10장 2절] 자료 조사하기
# 1절에서 State에 채운 questions로 웹과 뉴스를 검색하고, LLM으로 분류·분석해 통찰과
# 정보 부족 영역(갭)을 찾습니다. 검색 함수 → 정보 분석 함수 → 조사 노드 → 갭 체크 순서로 만듭니다.
# ==============================================================================

# %% [10-2] 1. 검색 도구 정의 - 1) 웹 검색 도구
# 네이버 서치 API로 웹 검색을 하는 도구입니다. 전반적인 정보 수집을 맡고, 정렬은 유사도순입니다.
# 검색 결과는 title, url, snippet, source_type을 담은 dict 목록으로 정리해 돌려줍니다.
# 웹 검색과 뉴스 검색 도구는 investigation_node 안에서 직접 호출됩니다.
# ── 검색 도구 정의 (2장 Tool 활용) ─────────────────────────────

def search_web(query: str, num_results: int = 5) -> List[dict]:
    """네이버 서치 API로 웹 검색을 수행하고 결과를 반환"""

    # 네이버 API는 요청 헤더의 클라이언트 아이디와 시크릿(.env의 NAVER_CLIENT_ID, NAVER_CLIENT_SECRET)으로 인증합니다.
    headers = {
        "X-Naver-Client-Id": os.getenv("NAVER_CLIENT_ID"),
        "X-Naver-Client-Secret": os.getenv("NAVER_CLIENT_SECRET"),
    }
    # display는 가져올 결과 수, sort="sim"은 유사도순 정렬입니다.
    params = {
        "query": query,
        "display": num_results,
        "sort": "sim",
    }

    response = requests.get(
        "https://openapi.naver.com/v1/search/webkr.json",
        headers=headers,
        params=params,
    )

    # 응답 코드가 200(성공)일 때만 결과를 담으므로, 키가 잘못되면 오류 없이 빈 목록이 돌아갑니다.
    results = []
    if response.status_code == 200:
        for item in response.json().get("items", []):
            results.append({
                "title": item.get("title", ""),
                "url": item.get("link", ""),
                "snippet": item.get("description", ""),
                "source_type": "web",
            })

    return results


# %% [10-2] 1. 검색 도구 정의 - 2) 뉴스 검색 도구
# 뉴스는 최신성을 담당합니다. 웹 검색과 구조가 비슷하지만 엔드포인트가 news.json이고 정렬 기준이 date(최신순)입니다.
# 결과에는 기사 원문 주소(originallink)와 발행일(pubDate)도 담습니다.
def search_news(query: str, num_results: int = 5) -> List[dict]:
    """네이버 뉴스 검색 API로 최근 뉴스를 수집"""

    headers = {
        "X-Naver-Client-Id": os.getenv("NAVER_CLIENT_ID"),
        "X-Naver-Client-Secret": os.getenv("NAVER_CLIENT_SECRET"),
    }
    params = {
        "query": query,
        "display": num_results,
        "sort": "date",
    }

    response = requests.get(
        "https://openapi.naver.com/v1/search/news.json",
        headers=headers,
        params=params,
    )

    results = []
    if response.status_code == 200:
        for item in response.json().get("items", []):
            results.append({
                "title": item.get("title", ""),
                "url": item.get("originallink", ""),
                "snippet": item.get("description", ""),
                "date": item.get("pubDate", ""),
                "source_type": "news",
            })

    return results


# %% [10-2] 2. 정보 분석 함수 - 1) 정보 분류 + 통찰 도출
# 검색 결과를 그냥 쌓아 두지 않고, LLM으로 카테고리별로 분류하고 인사이트를 도출합니다.
# LLM의 출력을 정해진 JSON 형식(categories, insights, gaps)으로 제약해 다음 단계에서 안정적으로 처리할 수 있게 합니다.
# gaps(아직 답을 찾지 못한 질문)는 갭 체크에서 조건부 엣지의 판단 기준이 됩니다.
# ── 정보 분석 (LLM 활용) ───────────────────────────────────────

def analyze_and_categorize(
    search_results: List[dict],
    news_results: List[dict],
    topic: str,
    questions: List[str]
) -> dict:
    """수집된 정보를 LLM으로 분류하고 통찰을 도출"""

    # 검색 결과를 텍스트로 정리
    # 웹 결과는 앞에서부터 15건, 뉴스 결과는 10건까지만 프롬프트에 넣습니다.
    web_text = "\n".join(
        f"- [{r['title']}] {r['snippet']}"
        for r in search_results[:15]
    )
    news_text = "\n".join(
        f"- [{r['title']}] {r['snippet']}"
        for r in news_results[:10]
    )
    questions_text = "\n".join(f"- {q}" for q in questions)

    analysis_prompt = f"""
    주제: {topic}

    [조사 질문]
    {questions_text}

    [웹 검색 결과]
    {web_text}

    [뉴스 검색 결과]
    {news_text}

    위 정보를 분석하여 아래 JSON 형식으로 반환하세요. 반드시 JSON만 반환하세요.

    {{
        "categories": [
            {{"category": "카테고리명", "key_facts": ["핵심 사실 1", "핵심 사실 2"]}},
            ...
        ],
        "insights": ["인사이트 1", "인사이트 2", ...],
        "gaps": ["아직 답을 찾지 못한 질문 1", ...]
    }}

    규칙:
    - categories는 최소 3개 이상 만드세요.
    - insights는 단순 사실 나열이 아니라, 정보를 종합한 통찰이어야 합니다.
    - gaps에는 위 조사 질문 중 검색 결과에서 충분히 답을 찾지 못한 것을 넣으세요.
    - 검색 결과에 정보가 없는 질문은 반드시 gaps에 포함하세요.
    """

    response = llm.invoke(analysis_prompt)

    # goal_setting_node와 같은 방법으로 코드블록을 걷어 내고 JSON을 dict로 바꿉니다.
    content = response.content.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[1].rsplit("```", 1)[0]

    return json.loads(content)


# %% [10-2] 3. investigation_node 구현
# 1절에서 채운 questions를 검색 쿼리로 써서 정보를 모으고, 분석 결과를 State에 기록하는 핵심 노드입니다.
# 다중 소스 검색(1단계) → 기존 결과와 병합(2단계) → LLM 분석과 통찰 도출(3단계) 순서로 진행합니다.
# ── 2절: 조사 노드 ─────────────────────────────────────────────

def investigation_node(state: ResearchState) -> dict:
    """
    검색 도구로 정보를 수집하고, LLM으로 분석하여 통찰을 도출하는 노드.

    [입력] state["questions"], state["topic"], state["iteration"]
    [출력] search_results, news_results, categories, insights, gaps, iteration
    """

    questions = state["questions"]
    topic = state["topic"]
    iteration = state.get("iteration", 0)

    print(f"\n 조사 시작 (반복 {iteration + 1}회차)")

    # ── 1단계: 다중 소스 검색 ──────────────────────────────────

    all_web_results = []
    all_news_results = []

    # 갭이 있으면 갭 질문만, 없으면 전체 질문 검색
    # 재조사(iteration > 0)에서는 부족한 영역만 다시 검색해, 충분한 영역은 건너뛰는 '의미 있는 반복'을 만듭니다.
    gaps = state.get("gaps", [])
    search_queries = gaps if gaps and iteration > 0 else questions

    for query in search_queries:
        print(f"    웹 검색: {query}")
        web_results = search_web(query)
        all_web_results.extend(web_results)

        print(f"    뉴스 검색: {query}")
        news_results = search_news(query)
        all_news_results.extend(news_results)

    print(f"   → 웹 {len(all_web_results)}건, 뉴스 {len(all_news_results)}건 수집")

    # ── 2단계: 기존 결과와 병합 (반복 시) ──────────────────────
    # 재조사할 때 이전 결과를 버리지 않고 이어 붙입니다. 이전 작업 결과를 기억하고 이어서 진행하는 체크포인터의 원리와 같습니다.

    existing_web = state.get("search_results", [])
    existing_news = state.get("news_results", [])

    merged_web = existing_web + all_web_results
    merged_news = existing_news + all_news_results

    # ── 3단계: LLM 기반 분석 + 통찰 도출 ──────────────────────
    # 분석은 원래의 조사 질문 전체(questions)를 기준으로 하므로, 남은 갭도 매번 새로 판단됩니다.

    print("    정보 분석 중...")
    analysis = analyze_and_categorize(
        merged_web, merged_news, topic, questions
    )

    categories = analysis.get("categories", [])
    insights = analysis.get("insights", [])
    new_gaps = analysis.get("gaps", [])

    print(f"   → 카테고리 {len(categories)}개, 인사이트 {len(insights)}개, 갭 {len(new_gaps)}개")

    # 조사를 한 번 마칠 때마다 iteration을 1 늘립니다. check_gaps가 이 값으로 재조사 여부를 정합니다.
    return {
        "search_results": merged_web,
        "news_results": merged_news,
        "categories": categories,
        "insights": insights,
        "gaps": new_gaps,
        "iteration": iteration + 1,
    }


# %% [10-2] 4. 갭 체크: 조건부 엣지
# 조사 노드 다음에 실행되는 조건부 엣지의 분기 함수입니다. 루프 패턴에서 본 check_pass(), should_continue()와 같은 구조로,
# 반환값이 문자열("retry" 또는 "proceed")이고 이 문자열이 조건부 엣지의 분기 키가 됩니다.
# goal_setting_node가 max_iterations를 2로 정하므로 조사 노드는 처음 조사 1번과 재조사 1번, 최대 2번 실행됩니다.
# ── 갭 체크: 조건부 엣지 (7장 루프 패턴) ───────────────────────

def check_gaps(state: ResearchState) -> str:
    """
    조사 결과의 갭을 확인하여, 재조사할지 생성으로 넘어갈지 결정.

    - 갭이 있고 반복 여유가 있으면 → "retry" (조사 노드로 루프)
    - 갭이 없거나 최대 반복 도달 → "proceed" (생성 노드로 진행)
    """

    gaps = state.get("gaps", [])
    iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", 2)

    if gaps and iteration < max_iterations:
        print(f"\n 갭 {len(gaps)}개 발견 → 재조사 (남은 횟수: {max_iterations - iteration})")
        return "retry"
    else:
        if not gaps:
            print("\n 갭 없음 → 보고서 생성으로 진행")
        else:
            print(f"\n 최대 반복 도달 ({max_iterations}회) → 보고서 생성으로 진행")
        return "proceed"


# ==============================================================================
# [10장 3절] 보고서 작성하기
# 2절에서 채운 categories, insights, gaps로 보고서를 생성하는 마지막 노드를 만들고,
# 1~3절의 모든 노드를 LangGraph 그래프로 조립해 실행합니다.
# ==============================================================================

# %% [10-3] 1. generation_node 구현
# 2절에서 수집·분석한 정보를 바탕으로 마크다운 형식의 리서치 보고서를 만드는 마지막 노드입니다.
# 동적 프롬프트 원리대로 topic, target_audience, categories, insights, gaps를 모두 프롬프트에 넣어 LLM이 맥락을 충분히 파악하게 합니다.
# 섹션별로 LLM을 따로 부르지 않고 한 번에 전체를 생성합니다. LLM이 전체 구조를 보며 쓰므로 섹션 간 흐름이 자연스럽고,
# API 호출 횟수도 줄어듭니다.
# ── 3절: 생성 노드 ─────────────────────────────────────────────

def generation_node(state: ResearchState) -> dict:
    """
    수집된 정보를 바탕으로 구조화된 리서치 보고서를 생성하는 노드.

    [입력] state["topic"], state["categories"], state["insights"],
           state["gaps"], state["target_audience"]
    [출력] report (마크다운 문자열)
    """

    topic = state["topic"]
    depth = state.get("depth", "상세")
    target_audience = state.get("target_audience", "일반인")
    categories = state.get("categories", [])
    insights = state.get("insights", [])
    gaps = state.get("gaps", [])
    questions = state.get("questions", [])
    search_results = state.get("search_results", [])
    news_results = state.get("news_results", [])

    print("\n[생성] 보고서 생성 시작")

    # ── 카테고리 정보를 텍스트로 정리 ──────────────────────────

    categories_text = ""
    for cat in categories:
        facts = cat.get("key_facts", [])
        facts_str = "\n".join(f"  - {f}" for f in facts)
        categories_text += f"\n[{cat.get('category', '기타')}]\n{facts_str}\n"

    insights_text = "\n".join(f"- {i}" for i in insights)
    gaps_text = "\n".join(f"- {g}" for g in gaps) if gaps else "- 없음 (모든 질문에 대해 충분한 정보 수집)"

    # 참고 소스 목록 (중복 제거)
    # set으로 같은 소스를 한 번만 남기고, 정렬한 뒤 앞의 15개에 번호를 붙입니다.
    sources = set()
    for r in search_results[:20]:
        sources.add(f"{r.get('title', '')} - {r.get('url', '')}")
    for r in news_results[:10]:
        sources.add(f"{r.get('title', '')} - {r.get('url', '')}")
    sources_text = "\n".join(f"{i+1}. {s}" for i, s in enumerate(sorted(sources)[:15]))

    # ── LLM에게 보고서 전체를 한 번에 생성 요청 ───────────────
    # [보고서 작성 가이드]로 목차(요약 → 카테고리별 섹션 → 주요 인사이트 → 결론 및 권고사항 → 참고문헌)를 정해 줍니다.

    report_prompt = f"""
    아래 정보를 바탕으로 '{topic}'에 대한 {depth} 리서치 보고서를 마크다운 형식으로 작성하세요.

    [대상 독자] {target_audience}

    [분류된 정보]
    {categories_text}

    [도출된 인사이트]
    {insights_text}

    [추가 조사 필요 영역]
    {gaps_text}

    [참고 소스]
    {sources_text}

    [보고서 작성 가이드]
    다음 구조를 반드시 따르세요:

    # {{제목}}

    ## 요약 (Executive Summary)
    - 3~5문장으로 핵심 발견사항을 요약

    ## 1. {{첫 번째 카테고리 제목}}
    - 해당 카테고리의 핵심 사실과 분석

    ## 2. {{두 번째 카테고리 제목}}
    - ...

    (분류된 정보의 카테고리 수만큼 섹션 생성)

    ## 주요 인사이트
    - 단순 사실 나열이 아닌, 정보를 종합한 통찰

    ## 결론 및 권고사항
    - 핵심 결론
    - 추가 조사가 필요한 영역 (gaps 반영)
    - 실행 가능한 제안

    ## 참고문헌
    - 출처 목록

    [작성 규칙]
    - {target_audience}이 이해할 수 있는 수준으로 작성
    - 객관적이고 균형 잡힌 관점 유지
    - 주장에는 반드시 근거(수집된 정보)를 제시
    - 전문 용어는 간단한 설명을 덧붙이기
    """

    response = llm.invoke(report_prompt)
    report = response.content

    print(f"   -> 보고서 생성 완료 ({len(report)}자)")

    return {
        "report": report,
    }


# %% [10-3] 2. 그래프 조립: 모든 노드를 하나로 엮기
# 1절의 goal_setting_node, 2절의 investigation_node와 check_gaps, 3절의 generation_node를 하나의 그래프로 엮습니다.
# add_node()로 노드를 등록하고 add_edge()와 add_conditional_edges()로 연결한 뒤, compile()로 실행 가능한 앱을 만듭니다.
#   START → goal_setting → investigation → [check_gaps] → generation → END
#                               ↑                │
#                               └── retry ───────┘
# ── 그래프 조립 ────────────────────────────────────────────────

def build_research_graph():
    """심층 리서치 에이전트의 그래프를 조립하고 컴파일"""

    # 1절의 ResearchState를 모든 노드가 공유하는 State로 씁니다.
    workflow = StateGraph(ResearchState)

    # ── 노드 등록 ──────────────────────────────────────────────
    workflow.add_node("goal_setting", goal_setting_node)      # 1절
    workflow.add_node("investigation", investigation_node)    # 2절
    workflow.add_node("generation", generation_node)          # 3절

    # ── 엣지 연결 ──────────────────────────────────────────────

    # START → 목표설정 → 조사 (순차)
    workflow.add_edge(START, "goal_setting")
    workflow.add_edge("goal_setting", "investigation")

    # 조사 → 갭 체크 (조건부 엣지: 7장 루프 패턴)
    # add_conditional_edges: check_gaps가 돌려준 문자열을 아래 dict에서 찾아 다음 노드를 고릅니다.
    workflow.add_conditional_edges(
        "investigation",
        check_gaps,
        {
            "retry": "investigation",    # 갭 있음 → 재조사
            "proceed": "generation",     # 충분함 → 보고서 생성
        }
    )

    # 생성 → END
    workflow.add_edge("generation", END)

    # ── 컴파일 ─────────────────────────────────────────────────
    app = workflow.compile()

    return app


# %% [10-3] 3. 실행하기 - 1) 코드에서 직접 실행
# 조립한 그래프를 app에 담고, run_deep_research()가 초기 State를 만들어 invoke로 실행합니다.
# 파일 맨 위 수준의 app은 langgraph.json의 "./agent/deep_research.py:app"이 가리키는 그래프이기도 해서,
# 3절 '2) LangGraph Studio에서 실행'의 Studio도 이 app을 불러옵니다.
# ── 실행 ───────────────────────────────────────────────────────

# 그래프 빌드
app = build_research_graph()

def run_deep_research(user_input: str) -> str:
    """심층 리서치 에이전트 실행"""

    print("=" * 60)
    print(" 심층 리서치 시작")
    print(f" 요청: {user_input}")
    print("=" * 60)

    # 초기 State 설정
    # user_input만 채우고 나머지 필드는 빈 값으로 시작합니다. 노드들이 차례로 값을 채워 넣습니다.
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

    # 그래프 실행
    # invoke는 END까지 실행한 뒤의 최종 State를 돌려줍니다. 보고서는 report 필드에 들어 있습니다.
    result = app.invoke(initial_state)

    print("\n" + "=" * 60)
    print(" 심층 리서치 완료!")
    print("=" * 60)
    return result["report"]


# 실행 예시
# (파일을 직접 실행할 때만 동작하고, Studio가 이 파일을 불러올 때는 실행되지 않습니다. 다른 주제는 요청 문장을 바꿉니다.)
if __name__ == "__main__":
    report = run_deep_research(
        "AI 반도체 시장 동향에 대해 심층 보고서를 작성해줘"
    )

    # 보고서를 파일로 저장
    # 상대 경로이므로 명령을 실행한 폴더에 research_report.md가 생깁니다.
    # [수정] 한글 Windows(cp949)에서도 보고서를 저장할 수 있도록 encoding="utf-8"을 지정합니다. (책: 아래 주석 줄)
    # with open("research_report.md", "w") as f:
    with open("research_report.md", "w", encoding="utf-8") as f:
        f.write(report)

    print("\n보고서가 research_report.md에 저장되었습니다.")

# [책의 실행 결과] 1회차 조사에서 갭 2개가 발견되어 재조사하고, 2회차에서 갭이 해소되어 보고서 생성으로 넘어갑니다.
# ============================================================
#  심층 리서치 시작
#  요청: AI 반도체 시장 동향에 대해 심층 보고서를 작성해줘
# ============================================================
#
# [목표설정] 완료
#    주제: AI 반도체 시장 동향
#    깊이: 심층
#    질문 5개 생성됨
#
# [조사] 시작 (반복 1회차)
#    [웹] AI 반도체 시장 규모 2025
#    [뉴스] AI 반도체 시장 규모 2025
#    [웹] AI 칩 주요 기업 경쟁 구도
#    [뉴스] AI 칩 주요 기업 경쟁 구도
#    ...
#    -> 웹 25건, 뉴스 20건 수집
#    [분석] 정보 분석 중...
#    -> 카테고리 4개, 인사이트 3개, 갭 2개
#
# [갭 체크] 갭 2개 발견 -> 재조사 (남은 횟수: 1)
#
# [조사] 시작 (반복 2회차)
#    [웹] AI 반도체 공급망 리스크
#    [뉴스] AI 반도체 공급망 리스크
#    ...
#    -> 웹 35건, 뉴스 28건 수집 (누적)
#    [분석] 정보 분석 중...
#    -> 카테고리 5개, 인사이트 5개, 갭 0개
#
# [갭 체크] 갭 없음 -> 보고서 생성으로 진행
#
# [생성] 보고서 생성 시작
#    -> 보고서 생성 완료 (3,847자)
#
# ============================================================
#  심층 리서치 완료!
# ============================================================
# 보고서가 research_report.md에 저장되었습니다.


# %% [10-3] 5. 전체 코드 구조 정리
# [설명용 코드] agent/deep_research.py 파일 하나에 들어가는 전체 구조를 요약한 코드입니다. ...은 본문 생략을 뜻합니다.
# # agent/deep_research.py 전체 구조
#
# # ── 임포트 & 환경 설정 ─────────────────────────────────────────
# import os, json, requests
# from typing import TypedDict, List, Optional
# from dotenv import load_dotenv, find_dotenv
# from langchain_openai import ChatOpenAI
# from langgraph.graph import StateGraph, START, END
#
# load_dotenv(find_dotenv())
# llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
#
# # ── State 정의 (1절) ───────────────────────────────────────────
# class ResearchState(TypedDict):
#     user_input: str
#     topic: str
#     depth: str
#     target_audience: str
#     questions: List[str]
#     search_results: List[dict]
#     news_results: List[dict]
#     categories: List[dict]
#     insights: List[str]
#     gaps: List[str]
#     report: str
#     iteration: int
#     max_iterations: int
#
# # ── 검색 도구 (2절) ────────────────────────────────────────────
# def search_web(query, num_results=5): ...
# def search_news(query, num_results=5): ...
# def analyze_and_categorize(search_results, news_results, topic, questions): ...
#
# # ── 노드 함수 (1~3절) ─────────────────────────────────────────
# def goal_setting_node(state): ...      # 1절: 목표설정
# def investigation_node(state): ...     # 2절: 조사
# def check_gaps(state): ...             # 2절: 갭 체크 (조건부 엣지)
# def generation_node(state): ...        # 3절: 보고서 생성
#
# # ── 그래프 조립 (3절) ──────────────────────────────────────────
# workflow = StateGraph(ResearchState)
# workflow.add_node("goal_setting", goal_setting_node)
# workflow.add_node("investigation", investigation_node)
# workflow.add_node("generation", generation_node)
#
# workflow.add_edge(START, "goal_setting")
# workflow.add_edge("goal_setting", "investigation")
# workflow.add_conditional_edges(
#     "investigation", check_gaps,
#     {"retry": "investigation", "proceed": "generation"}
# )
# workflow.add_edge("generation", END)
#
# app = workflow.compile()
