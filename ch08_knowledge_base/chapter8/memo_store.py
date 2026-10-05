"""
[8장 3절] 에이전트 시험 공부 하기 - 벡터 메모 스토어

InMemoryStore에 임베딩 인덱스(fields=["content"])를 붙여 작품 프로필과 메모를 넣고,
query=로 의미 기반 검색을 합니다. 프로필과 메모 모두 검색에 쓰일 content 문자열을 채웁니다.
InMemoryStore는 프로그램이 끝나면 비워지므로 메모 값을 chapter8/.memo_cache/literature_memos.json에
저장해 두었다가 exam 실행 때 다시 불러옵니다.

필요한 환경 변수: OPENAI_API_KEY (임베딩 모델 text-embedding-3-small)
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, Iterable

from langchain.embeddings import init_embeddings
from langgraph.store.memory import InMemoryStore

from chapter8.schemas import LiteratureExtraction

USER_ID_DEFAULT = "student_local"


def build_vector_store() -> InMemoryStore:
    embeddings = init_embeddings("openai:text-embedding-3-small")
    return InMemoryStore(
        index={
            "embed": embeddings,
            "dims": 1536,
            "fields": ["content"],  # value 중 content 필드만 임베딩합니다(지정하지 않으면 value 전체)
        }
    )


# 2절의 서랍 구조에 user_id를 더했습니다. 메모는 ("literature", "memos", user_id, 작품명), 프로필은 ("literature", "profiles", user_id)입니다.
def memos_namespace(user_id: str, work_title: str) -> tuple[str, ...]:
    return ("literature", "memos", user_id, work_title)


def profiles_namespace(user_id: str) -> tuple[str, ...]:
    return ("literature", "profiles", user_id)


def put_extraction_memos(
    store: InMemoryStore,
    extracted: LiteratureExtraction,
    *,
    user_id: str = USER_ID_DEFAULT,
) -> None:
    """프로필 1건 + 해석/인용/관계 요약 메모를 스토어에 적재."""
    # 작품 프로필: 작품명을 key로 덮어씁니다("한 작품 = 한 항목").
    ns_prof = profiles_namespace(user_id)
    theme_txt = ", ".join(extracted.themes)
    store.put(
        ns_prof,
        extracted.work_title,
        {
            "work_title": extracted.work_title,
            "author": extracted.author,
            "summary": extracted.summary,
            "themes": extracted.themes,
            "period": "",
            "content": f"{extracted.author} 《{extracted.work_title}》. {extracted.summary} 주제: {theme_txt}",
        },
    )

    # 해석·관계·핵심 문장 메모: 메모마다 uuid key로 계속 쌓습니다("한 작품 = 여러 항목").
    ns_m = memos_namespace(user_id, extracted.work_title)

    for interp in extracted.interpretations:
        store.put(
            ns_m,
            str(uuid.uuid4()),
            {
                "work_title": extracted.work_title,
                "memo_type": "exam",
                "content": f"[해석:{interp.subject}] {interp.meaning}",
                "tags": ["interpretation", interp.subject],
            },
        )

    for rel in extracted.relations:
        line = f"[관계] {rel.source} —{rel.relation}→ {rel.target}"
        if rel.evidence_quote:
            line += f" (근거: {rel.evidence_quote})"
        store.put(
            ns_m,
            str(uuid.uuid4()),
            {
                "work_title": extracted.work_title,
                "memo_type": "exam",
                "content": line,
                "tags": ["relation", rel.relation],
            },
        )

    for q in extracted.key_quotes:
        store.put(
            ns_m,
            str(uuid.uuid4()),
            {
                "work_title": extracted.work_title,
                "memo_type": "quote",
                "content": q,
                "tags": ["quote"],
            },
        )


def search_memos_for_work(
    store: InMemoryStore,
    query: str,
    work_title: str,
    *,
    user_id: str = USER_ID_DEFAULT,
    limit: int = 8,
) -> list:
    # 네임스페이스를 작품까지 좁힌 뒤 벡터 검색합니다(2절의 '필터 + 벡터 검색 조합').
    ns = memos_namespace(user_id, work_title)
    return list(store.search(ns, query=query, limit=limit))


def search_memos_all_works(
    store: InMemoryStore,
    query: str,
    work_titles: Iterable[str],
    *,
    user_id: str = USER_ID_DEFAULT,
    limit_per_work: int = 4,
) -> list:
    out: list = []
    for t in work_titles:
        out.extend(search_memos_for_work(store, query, t, user_id=user_id, limit=limit_per_work))
    out.sort(key=lambda x: getattr(x, "score", 0) or 0, reverse=True)
    return out


def save_memo_cache(path: Path, store: InMemoryStore, user_id: str) -> None:
    """임베딩은 저장하지 않고 value만 저장; 로드 시 다시 임베딩."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    # profiles
    ns_p = profiles_namespace(user_id)
    for item in store.search(ns_p, limit=200):
        rows.append(
            {
                "namespace": list(ns_p),
                "key": item.key,
                "value": item.value,
            }
        )
    # memos: 모든 작품 네임스페이스를 알 수 없으므로 profiles 키(작품명) 기준으로 수집
    prof = store.search(ns_p, limit=200)
    titles = list(
        dict.fromkeys(
            item.value.get("work_title") or item.key
            for item in prof
            if isinstance(item.value.get("work_title") or item.key, str)
        )
    )
    for title in titles:
        ns_m = memos_namespace(user_id, title)
        for item in store.search(ns_m, limit=500):
            rows.append(
                {
                    "namespace": list(ns_m),
                    "key": item.key,
                    "value": item.value,
                }
            )
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def load_memo_cache(path: Path) -> InMemoryStore:
    store = build_vector_store()
    if not path.exists():
        return store
    raw = json.loads(path.read_text(encoding="utf-8"))
    for row in raw:
        ns = tuple(row["namespace"])
        store.put(ns, row["key"], row["value"])
    return store
