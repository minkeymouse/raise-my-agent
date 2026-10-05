"""
[8장 3절] 에이전트 시험 공부 하기 - 지식베이스 적재 (chapter8 ingest)

data/날개.txt, data/동백꽃.txt, data/운수좋은날.txt 본문(앞부분 일정 글자)을 LLM으로
LiteratureExtraction에 구조화한 뒤, Neo4j에 노드·관계로 저장하고, 해석·관계·인용을
벡터 검색 가능한 메모로 InMemoryStore에 넣은 다음 chapter8/.memo_cache/literature_memos.json에 저장합니다.
2절에서 배운 구조화 추출(with_structured_output), 그래프 저장(MERGE), 스토어 메모를 세 작품에 차례로 적용하는 단계입니다.

실행 방법 (ch08_knowledge_base 폴더에서):
    uv run python -m chapter8 ingest
필요한 환경 변수: OPENAI_API_KEY, NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD (선택: NEO4J_DATABASE, CHAPTER8_LLM_MODEL)
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from chapter8.env_loader import load_project_env
from chapter8.memo_store import (
    USER_ID_DEFAULT,
    build_vector_store,
    put_extraction_memos,
    save_memo_cache,
)
from chapter8.neo4j_store import build_neo4j_graph, save_extraction_to_graph
from chapter8.paths import LITERATURE_FILES, MEMO_CACHE_PATH
from chapter8.schemas import LiteratureExtraction

# 파일명 → (작품명, 작가) 고정 매핑 (예시 데이터와 맞춤)
WORK_BY_FILENAME: dict[str, tuple[str, str]] = {
    "날개.txt": ("날개", "이상"),
    "동백꽃.txt": ("동백꽃", "김유정"),
    "운수좋은날.txt": ("운수 좋은 날", "현진건"),
}

# 본문 길이: 토큰 한도를 고려해 본문을 앞에서부터 이 글자 수까지만 씁니다.
MAX_CHARS = 14_000


def _read_corpus(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _truncate(text: str, max_chars: int = MAX_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[이하 생략 — 원본은 로컬 data 파일 전체입니다.]"


def _build_llm() -> ChatOpenAI:
    model = os.environ.get("CHAPTER8_LLM_MODEL", "gpt-4o-mini")
    return ChatOpenAI(model=model, temperature=0)


def extract_from_text(
    text: str,
    work_title: str,
    author: str,
    structured_llm,
) -> LiteratureExtraction:
    # 작품명과 작가를 프롬프트에 고정하고, 발췌분에 드러난 인물·관계·사건·주제·인용을 LiteratureExtraction 형식으로 받습니다.
    prompt = f"""다음은 한국 근·현대 문학 작품 《{work_title}》(작가: {author})의 본문 일부입니다.
작품명과 작가는 위와 반드시 동일하게 채우고, 발췌분에 실제로 드러나는 내용 위주로
인물·관계·사건·주제·핵심 인용을 추출하세요. 불확실하면 관계 유형은 unknown을 쓰세요.

본문:
{text}
"""
    return structured_llm.invoke([HumanMessage(content=prompt)])


def run_ingest(*, user_id: str = USER_ID_DEFAULT) -> None:
    load_project_env()
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY가 필요합니다 (.env.local 권장).")

    llm = _build_llm()
    structured = llm.with_structured_output(LiteratureExtraction)  # 2절의 관계 중심 노트 스키마
    graph = build_neo4j_graph()
    store = build_vector_store()

    for path in LITERATURE_FILES:
        if not path.exists():
            raise FileNotFoundError(path)
        meta = WORK_BY_FILENAME.get(path.name)
        if not meta:
            raise KeyError(f"작품 메타 없음: {path.name}")
        title, author = meta
        body = _truncate(_read_corpus(path))
        extracted = extract_from_text(body, title, author, structured)
        # 작품명과 작가는 파일 기준 값으로 덮어써, 그래프와 메모의 작품명이 어긋나지 않게 합니다.
        extracted = extracted.model_copy(update={"work_title": title, "author": author})
        # 같은 추출 결과를 Neo4j(관계의 뼈대)와 스토어(근거 메모) 두 곳에 저장합니다.
        save_extraction_to_graph(extracted, graph)
        put_extraction_memos(store, extracted, user_id=user_id)
        print(f"적재 완료: 《{extracted.work_title}》 ({extracted.author})")

    # InMemoryStore는 프로그램이 끝나면 비워지므로, exam에서 다시 쓸 수 있게 메모 값을 파일로 남깁니다.
    save_memo_cache(MEMO_CACHE_PATH, store, user_id)
    print(f"메모 캐시 저장: {MEMO_CACHE_PATH}")


def main() -> None:
    p = argparse.ArgumentParser(description="8장: 문학 코퍼스 → Neo4j + 메모 캐시")
    p.add_argument(
        "--user-id",
        default=USER_ID_DEFAULT,
        help="스토어 네임스페이스에 쓸 사용자 id",
    )
    args = p.parse_args()
    run_ingest(user_id=args.user_id)


if __name__ == "__main__":
    main()
