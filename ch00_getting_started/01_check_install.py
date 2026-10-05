"""
[0장 1절] 개발 환경 설정

이 절에서 배우는 것:
    - 파이썬(3.11 이상)과 uv로 프로젝트 전용 가상환경을 만들고 필수 패키지를 설치합니다.
    - LangChain/LangGraph v1은 공급자별 API 키만 있으면 대부분의 예제를 실행할 수 있어, .env 파일에 OpenAI 키를 넣어 둡니다.
    - 마지막으로 langgraph의 버전을 출력해 설치가 잘 되었는지 확인합니다.

준비 순서:
    1. 파이썬 확인: python --version (3.11 이상 권장. 낮거나 없으면 3번의 uv sync가 알맞은 버전을 내려받습니다)
    2. uv 설치: https://docs.astral.sh/uv/getting-started/installation/
    3. 리포지토리 루트에서 uv sync
       가상환경(.venv) 생성과 패키지 설치가 이 한 줄로 끝납니다. pyproject.toml과 uv.lock에
       모든 장의 패키지가 들어 있으므로, 각 장에 나오는 uv add나 pip install은 따로 실행하지 않아도 됩니다.
    4. 가상환경 활성화는 선택입니다. uv run이 .venv를 자동으로 사용합니다.
    5. cp .env.example .env (Windows 명령 프롬프트에서는 copy) 후 OPENAI_API_KEY 값을 본인의 키로 바꿉니다.
       키는 https://platform.openai.com/api-keys 에서 발급합니다. 다른 키는 해당 장에 들어갈 때 채웁니다.
       각 파일 머리말의 '필요한 환경 변수'를 확인하세요.
    6. 이 파일을 실행해 에러 없이 버전이 출력되면 준비가 끝난 것입니다.

실행 방법:
    uv run python ch00_getting_started/01_check_install.py
    모든 예제는 리포지토리 루트에서 uv run python <장 폴더>/<파일> 형태로 실행합니다.
    '# %%' 줄은 책의 소제목입니다. VS Code(Python, Jupyter 확장) 등 편집기에서 리포지토리의 .venv를 인터프리터로
    선택하면 위에서부터 차례로 셀 단위로 실행할 수도 있습니다.
필요한 환경 변수: 없음

표시: [보충] 실행에 필요한 코드, [수정] 실행에 맞게 고친 코드, [설명용 코드] 실행되지 않는 설명용 조각
"""

# .env 파일의 API 키를 환경 변수로 불러옵니다.
from dotenv import load_dotenv
load_dotenv()

# %% 설치 확인
# langgraph를 불러와 버전을 출력해 봅니다. 에러 없이 버전이 나오면 준비가 끝난 것입니다.
# ModuleNotFoundError가 나면 uv sync를 실행했는지, uv run(또는 활성화된 가상환경)으로 실행했는지 확인하세요.
import langgraph

# [수정] langgraph의 버전은 importlib.metadata의 version()으로 읽습니다. (책: 아래 주석 줄)
# print("LangGraph 버전:", langgraph.__version__)
from importlib.metadata import version

print("LangGraph 버전:", version("langgraph"))
