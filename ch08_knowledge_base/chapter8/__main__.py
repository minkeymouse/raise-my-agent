"""
[8장 3절] 에이전트 시험 공부 하기 - 3. 시험공부 에이전트 구축 (실행 진입점)

이 절에서 배우는 것:
- 저장소(벡터 메모)와 Neo4j(관계·해석)를 함께 쓰는 시험공부 에이전트를 chapter8 패키지로 완성합니다.
- ingest: data/의 세 소설 본문을 LLM으로 LiteratureExtraction에 구조화해 Neo4j와 벡터 메모에 적재합니다(chapter8/ingest.py).
- exam: 기출 문항마다 키워드 추출 -> 《날개》 메모 의미 검색 -> Neo4j 인물 관계·해석 조회 -> 구조화 출력으로
  선지 번호와 해설을 만듭니다(chapter8/exam_agent.py). 내부 논리는 'RAG + 구조화 추론' 흐름입니다.

실행 방법 (책은 프로젝트 루트에서 실행하지만, 이 리포에서는 chapter8/이 ch08_knowledge_base/ 안에 있으므로 먼저 이 폴더로 이동합니다):
    cd ch08_knowledge_base
    uv run python -m chapter8 ingest   # 소설 텍스트를 DB에 적재
    uv run python -m chapter8 exam              # 40~43 전부
    uv run python -m chapter8 exam --question 40
    (리포지토리 루트에서는: uv run --directory ch08_knowledge_base python -m chapter8 ingest)
    exam은 ingest가 만든 Neo4j 그래프와 메모 캐시(chapter8/.memo_cache/literature_memos.json)를 쓰므로 ingest를 먼저 실행합니다.
필요한 환경 변수: OPENAI_API_KEY, NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD (선택: NEO4J_DATABASE, CHAPTER8_LLM_MODEL, 기본 gpt-4o-mini)
    리포지토리 루트의 .env.local을 먼저 읽고 .env로 보완합니다. Neo4j 준비는 01_store_basics.py 머리말을 참고합니다.
준비물: exam은 data/exam/기출_날개.txt가 필요합니다. 저작권 문제로 리포에 넣지 않았으며, 준비 방법은 chapter8/exam_agent.py 머리말에 있습니다.

모델이 정답을 맞힌다는 보장은 없습니다. 이 에이전트의 목적은 지식베이스로 근거를 찾고 추론하는 과정을 보여 주는 것입니다.
"""

import argparse

from chapter8.exam_agent import run_exam
from chapter8.ingest import run_ingest


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m chapter8")
    sub = parser.add_subparsers(dest="cmd", required=True)  # ingest, exam 두 하위 명령

    p_ing = sub.add_parser("ingest", help="data/*.txt → Neo4j + 메모 캐시")
    # --user-id는 스토어 네임스페이스 ("literature", "memos", user_id, 작품명)에 들어가는 사용자 id입니다.
    p_ing.add_argument("--user-id", default="student_local")
    p_ing.set_defaults(fn=lambda a: run_ingest(user_id=a.user_id))

    p_ex = sub.add_parser("exam", help="기출 《날개》 풀이 (ingest 선행 권장)")
    p_ex.add_argument("--question", type=int, default=None)
    p_ex.add_argument("--user-id", default="student_local")
    p_ex.set_defaults(fn=lambda a: run_exam(only=a.question, user_id=a.user_id))

    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()

# [책의 실행 결과] (gpt-4o-mini, ingest 후 exam 실행. 직접 실행하면 다른 선지·해설이 나올 수 있습니다.)
# 적재 완료: 《날개》 (이상)
# 적재 완료: 《동백꽃》 (김유정)
# 적재 완료: 《운수 좋은 날》 (현진건)
# 메모 캐시 저장: .../chapter8/.memo_cache/literature_memos.json
#
# 지문 길이: 2064자, 풀이 문항: [40, 41, 42, 43]
#
# === 문항 40 ===
# 선택: ① (1)
# 해설: 주인공의 독백적인 어조가 두드러지며, 그의 의식 상태가 현실과 단절된 느낌을 주고 있다. 또한, 회상의 기법을 통해 과거의 경험을 회고하며 현재의 갈등과 화해를 지향하는 모습이 나타난다.
#
# === 문항 41 ===
# 선택: ② (2)
# 해설: 주인공은 미쓰꼬시 옥상에서 자신의 존재와 삶에 대한 회고를 하며 내면적 성찰을 시도하고, 회탁의 거리를 바라보며 그곳의 부자유와 체념을 인식한다. 그러나 '이전과는 다른 삶의 태도를 갖게 한다'는 설명은 주인공이 여전히 갈등과 혼란 속에 있음을 보여주므로 적절하지 않다.
#
# === 문항 42 ===
# 선택: ④ (4)
# 해설: 주인공은 정오의 사이렌 소리를 듣고 자신의 존재와 날개에 대한 갈망을 느끼며, 자아의 문제를 인식하고 있지만, 사회의 문제로 시선을 전환하는 내용은 나타나지 않는다. 오히려 주인공은 자신의 고독과 아내와의 관계에서 느끼는 갈등을 탐구하고 있다.
#
# === 문항 43 ===
# 선택: ② (2)
# 해설: 주인공이 아내 몰래 집에서 나온 것은 현대 문명에 대한 반발이 아니라, 오히려 아내와의 갈등 속에서 느끼는 고독과 소외감을 나타내며, 이는 현대 문명에 대한 대결 의지가 아닌 고립된 삶을 상징한다.
