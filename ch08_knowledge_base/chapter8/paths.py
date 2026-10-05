"""
[8장 3절] 에이전트 시험 공부 하기 - 데이터 경로

모든 경로를 이 패키지의 위치를 기준으로 계산하므로, 어느 폴더에서 실행해도
ch08_knowledge_base/data/ 아래의 소설 본문과 기출 문제 파일을 찾습니다.
(기출 문제 파일은 리포에 포함되어 있지 않으므로 독자가 직접 data/exam/기출_날개.txt로 준비합니다.
준비 방법은 chapter8/exam_agent.py 머리말에 있습니다.)
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
LITERATURE_FILES = [
    DATA_DIR / "날개.txt",
    DATA_DIR / "동백꽃.txt",
    DATA_DIR / "운수좋은날.txt",
]
EXAM_NALGAE = DATA_DIR / "exam" / "기출_날개.txt"
MEMO_CACHE_PATH = Path(__file__).resolve().parent / ".memo_cache" / "literature_memos.json"
