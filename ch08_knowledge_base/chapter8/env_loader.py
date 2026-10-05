"""
[8장 3절] 에이전트 시험 공부 하기 - 환경 변수 불러오기

리포지토리 루트의 .env.local(우선)과 .env를 로드합니다. Neo4j·OpenAI 키는 .env.local에 두고 저장소에 커밋하지 않습니다.
표시: [보충] 실행에 필요한 코드, [수정] 실행에 맞게 고친 코드, [설명용 코드] 실행되지 않는 설명용 조각
"""

from pathlib import Path

from dotenv import load_dotenv

# [보충] chapter8/이 ch08_knowledge_base/ 안에 있으므로 세 단계 위의 리포지토리 루트를 가리킵니다.
#        (원본: _ROOT = Path(__file__).resolve().parent.parent)
_ROOT = Path(__file__).resolve().parent.parent.parent


def load_project_env() -> None:
    """`.env.local`을 먼저 읽고, 이어서 `.env`로 보완합니다."""
    load_dotenv(_ROOT / ".env.local", override=True)
    load_dotenv(_ROOT / ".env", override=False)
