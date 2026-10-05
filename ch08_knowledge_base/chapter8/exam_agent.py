"""
[8장 3절] 에이전트 시험 공부 하기 - 기출 문제 풀이 (chapter8 exam)

data/exam/기출_날개.txt에서 지문과 40~43번 문항 블록을 나눈 뒤, 문항별로
(1) 키워드 추출 -> (2) 《날개》 메모 네임스페이스에서 의미 검색 -> (3) Neo4j에서 인물 관계·해석 노드 조회
-> (4) 구조화 출력으로 선지 번호·해설을 생성합니다. 시험 문제는 '정답(판단) + 근거(텍스트)'를 함께 요구하므로
그래프에서 관계의 뼈대를 잡고 스토어에서 근거 문장을 보강합니다. 지문은 잘라 쓰지 않고 전체를 그대로 넣습니다.

실행 방법 (ch08_knowledge_base 폴더에서, ingest를 먼저 실행해 두어야 합니다):
    uv run python -m chapter8 exam
    uv run python -m chapter8 exam --question 40
필요한 환경 변수: OPENAI_API_KEY, NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD (선택: NEO4J_DATABASE, CHAPTER8_LLM_MODEL)
표시: [보충] 실행을 위해 더한 코드, [수정] 책 코드의 오류를 고친 곳, [설명용 코드] 실행되지 않는 설명용 조각

기출 문항 파일 준비 (출제 기관에 권리가 있어 리포에 넣지 않았습니다. ingest는 이 파일 없이 실행됩니다):
1. 《날개》가 지문으로 나온 수능 국어 문학 기출 문항을 한국교육과정평가원 누리집(https://www.suneung.re.kr/) 등에서 구합니다.
2. 지문 본문을 먼저 적고, 그 아래에 문항을 "40. ", "41. ", "42. ", "43. "처럼 줄 맨 앞에 번호, 마침표, 공백을 두어 차례로 적습니다.
   각 문항에는 발문과 선지(①~⑤)를 함께 적습니다. 첫 "40. " 앞까지가 지문으로 쓰입니다.
3. UTF-8로 ch08_knowledge_base/data/exam/기출_날개.txt에 저장합니다. 이 폴더는 .gitignore에 들어 있어 커밋되지 않습니다.
OpenAI 키 없이도 파일이 지문과 문항으로 나뉘는지 확인할 수 있습니다(ch08_knowledge_base 폴더에서):
    uv run python -c "from chapter8.exam_agent import _split_exam_nalgae; from chapter8.paths import EXAM_NALGAE; p, q = _split_exam_nalgae(EXAM_NALGAE.read_text(encoding='utf-8')); print('passage_chars', len(p)); print('question_numbers', [x.number for x in q])"
[책의 실행 결과]
    passage_chars 2064
    question_numbers [40, 41, 42, 43]
"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from chapter8.env_loader import load_project_env
from chapter8.memo_store import USER_ID_DEFAULT, load_memo_cache, search_memos_for_work
from chapter8.neo4j_store import (
    build_neo4j_graph,
    fetch_character_relations,
    fetch_interpretations,
)
from chapter8.paths import EXAM_NALGAE, MEMO_CACHE_PATH
from chapter8.schemas import ExamKeywords, ExamSolution


@dataclass
class ParsedQuestion:
    number: int
    passage: str
    body: str


def _split_exam_nalgae(text: str) -> tuple[str, list[ParsedQuestion]]:
    """기출_날개.txt에서 지문 본문과 문항 블록을 분리."""
    # 줄 맨 앞의 "40. "을 찾아 그 앞까지를 지문으로 쓰고, 나머지는 "40. "~"43. "으로 시작하는 줄마다 문항 블록으로 나눕니다.
    m = re.search(r"^40\.\s", text, re.MULTILINE)
    if not m:
        raise ValueError("파일에서 '40.' 문항 시작을 찾지 못했습니다.")
    passage = text[: m.start()].strip()
    rest = text[m.start() :]
    parts = re.split(r"(?=^4[0-3]\.\s)", rest, flags=re.MULTILINE)
    questions: list[ParsedQuestion] = []
    for block in parts:
        block = block.strip()
        if not block:
            continue
        hm = re.match(r"^(4[0-3])\.\s*", block)
        if not hm:
            continue
        num = int(hm.group(1))
        body = block[hm.end() :].strip()
        questions.append(ParsedQuestion(number=num, passage=passage, body=body))
    return passage, questions


# Neo4j에서 《날개》의 인물 관계(RELATION)와 해석(Interpretation)을 꺼내, 관계의 뼈대가 되는 근거 자료로 만듭니다.
def _format_graph_context_nalgae(graph) -> str:
    rels = fetch_character_relations(graph, "날개")
    interps = fetch_interpretations(graph, "날개")
    lines: list[str] = ["[Neo4j: 《날개》 인물 관계]", ""]
    for r in rels:
        lines.append(
            f"- {r.get('source')} —{r.get('relation')}→ {r.get('target')} "
            f"(근거: {r.get('evidence')})"
        )
    lines.extend(["", "[Neo4j: 《날개》 해석 노드]", ""])
    for i in interps:
        lines.append(f"- {i.get('subject')}: {i.get('meaning')} (근거: {i.get('evidence')})")
    return "\n".join(lines) if rels or interps else "(그래프에서 《날개》 관계/해석 조회 결과 없음)"


# 키워드로 《날개》 메모 서랍을 의미 검색해, 보기 문장과 비슷한 근거 메모를 모읍니다.
def _vector_context(store, query: str, *, user_id: str) -> str:
    items = search_memos_for_work(store, query, "날개", user_id=user_id, limit=10)
    if not items:
        return "(벡터 메모 검색 결과 없음 — 먼저 `uv run python -m chapter8 ingest` 실행)"
    lines = ["[벡터 메모: 《날개》]", ""]
    for it in items:
        v = it.value
        lines.append(f"- [{v.get('memo_type')}] {v.get('content')}")
    return "\n".join(lines)


def solve_question(
    pq: ParsedQuestion,
    *,
    store,
    graph,
    user_id: str,
    llm: ChatOpenAI,
) -> ExamSolution:
    # (1) 키워드 추출: ExamKeywords 스키마로 검색 키워드와 문제의 초점을 구조화해 받습니다.
    kw_llm = llm.with_structured_output(ExamKeywords)
    keywords = kw_llm.invoke(
        [
            HumanMessage(
                content=(
                    "다음은 문학 기출 문항입니다. 지식베이스 검색에 쓸 키워드 5~12개를 뽑고, "
                    "문제 초점을 한 문장으로 적으세요.\n\n" + pq.body
                )
            )
        ]
    )
    q = ", ".join(keywords.keywords)
    vec_ctx = _vector_context(store, q, user_id=user_id)  # (2) 벡터 메모 검색
    graph_ctx = _format_graph_context_nalgae(graph)  # (3) Neo4j 관계·해석 조회

    # 두 근거 자료를 시스템 메시지에 함께 넣고, 이 자료만으로 추론해 답을 하나 고르게 합니다.
    sys = SystemMessage(
        content=f"""당신은 수능/평가원 스타일 한국어 문학 객관식 문제를 푸는 조력자입니다.
반드시 아래 '근거 자료'만을 인용해 추론하고, 선지 번호 ①~⑤는 각각 1~5에 대응합니다.
답은 하나만 고릅니다.

### 근거 자료 — 그래프(관계·해석)
{graph_ctx}

### 근거 자료 — 벡터 메모
{vec_ctx}
"""
    )
    # (4) ExamSolution 스키마로 선지 번호(answer_number)와 해설(reasoning)을 받습니다.
    sol_llm = llm.with_structured_output(ExamSolution)
    return sol_llm.invoke(
        [
            sys,
            HumanMessage(
                content=(
                    f"다음은 작품 《날개》의 지문입니다.\n\n{pq.passage}\n\n"
                    f"### 문항 {pq.number}\n{pq.body}\n\n"
                    "정답 번호(1~5)와 근거를 제시하세요."
                )
            ),
        ]
    )


def run_exam(*, only: int | None = None, user_id: str = USER_ID_DEFAULT) -> None:
    load_project_env()
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY가 필요합니다.")

    # [보충] 기출 문항 파일은 저작권 문제로 리포에 포함하지 않았습니다. 파일이 없으면 준비 방법을 안내합니다.
    if not EXAM_NALGAE.is_file():
        raise SystemExit(
            f"기출 문항 파일이 없습니다: {EXAM_NALGAE}\n"
            "《날개》 지문과 40~43번 문항을 직접 준비해 이 경로에 저장하세요. 준비 방법과 형식은 chapter8/exam_agent.py 머리말을 참고하세요."
        )
    raw = EXAM_NALGAE.read_text(encoding="utf-8")
    passage, questions = _split_exam_nalgae(raw)
    if only is not None:
        questions = [q for q in questions if q.number == only]
        if not questions:
            raise SystemExit(f"문항 {only}을(를) 찾을 수 없습니다.")

    # ingest가 저장한 메모 캐시를 다시 임베딩해 벡터 스토어를 만들고, Neo4j에 접속합니다.
    store = load_memo_cache(MEMO_CACHE_PATH)
    graph = build_neo4j_graph()
    model = os.environ.get("CHAPTER8_LLM_MODEL", "gpt-4o-mini")
    llm = ChatOpenAI(model=model, temperature=0)

    print(f"지문 길이: {len(passage)}자, 풀이 문항: {[q.number for q in questions]}")
    for pq in questions:
        sol = solve_question(pq, store=store, graph=graph, user_id=user_id, llm=llm)
        circled = "①②③④⑤"[sol.answer_number - 1] if 1 <= sol.answer_number <= 5 else "?"  # 1~5를 ①~⑤로
        print(f"\n=== 문항 {pq.number} ===")
        print(f"선택: {circled} ({sol.answer_number})")
        print(f"해설: {sol.reasoning}")


def main() -> None:
    p = argparse.ArgumentParser(description="8장 3절: 기출 《날개》 문항 풀이")
    p.add_argument("--question", type=int, default=None, help="40~43 중 특정 번호만")
    p.add_argument("--user-id", default=USER_ID_DEFAULT)
    args = p.parse_args()
    run_exam(only=args.question, user_id=args.user_id)


if __name__ == "__main__":
    main()
