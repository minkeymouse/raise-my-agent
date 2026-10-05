r"""
[9장 3절] [Build] 우리 집 비법 공유하기: 아이의 재능을 세상의 표준으로

이 절에서 배우는 것:
- 2절과 반대로, FastMCP로 우리만의 MCP Server를 만들어(방과후 수업 개설) 어떤 AI 앱에서든 우리 도구를 쓰게 합니다.
- Server의 세 가지 구성요소를 모두 만듭니다. Tool(@mcp.tool(), 실행하는 함수), Resource(@mcp.resource(),
  URI로 읽는 읽기 전용 데이터), Prompt(@mcp.prompt(), LLM이 대화를 시작할 때 참고하는 템플릿)입니다.
- mcp.run()의 인자 하나로 stdio(로컬)와 HTTP(원격) 실행 방식을 고릅니다.
- 이 서버에 연결해 보는 코드는 같은 폴더의 study_client.py에, Claude Desktop 연결(4절)은 아래에 있습니다.

실행 방법:
    uv run python ch09_mcp/03_build/study_client.py
    (study_client.py가 이 서버를 stdio 방식으로 직접 띄웁니다. 이 파일만 단독으로 실행하면
     클라이언트의 연결을 기다리며, Ctrl+C로 종료합니다.)
필요한 환경 변수: 없음 (LLM을 호출하지 않습니다)
create_note가 만든 노트는 이 파일 옆의 notes/ 폴더(ch09_mcp/03_build/notes/)에 저장됩니다.

Claude Desktop에 연결하기 (4절 [Integration] 더 큰 네트워크로: Claude 라는 친구 사귀기)
이 서버를 Claude Desktop(Host)의 설정 파일에 등록하면, 코드를 실행하지 않고 대화만으로 우리 도구가 호출됩니다.
Claude Desktop은 앱을 실행할 때마다 설정 파일에 적힌 서버를 자동으로 시작합니다.
1. https://claude.ai/download 에서 Claude Desktop을 설치해 실행하고, Settings -> Developer -> Edit Config로
   설정 파일 claude_desktop_config.json을 엽니다.
   macOS: ~/Library/Application Support/Claude/claude_desktop_config.json
   Windows: %APPDATA%\Claude\claude_desktop_config.json
2. ch09_mcp/04_integration/claude_desktop_config.example.json(4절의 설정 예시)의 내용을 붙여 넣고, 두 값을
   이 컴퓨터의 절대 경로로 고칩니다(상대 경로는 Claude Desktop이 인식하지 못할 수 있습니다).
   - args: 이 파일의 절대 경로. 예: /Users/username/raise-my-agent/ch09_mcp/03_build/study_server.py
   - command: fastmcp가 설치된 파이썬이어야 하므로 "python" 대신 리포지토리 루트 .venv의 파이썬 절대 경로를 적습니다.
     (macOS: .../raise-my-agent/.venv/bin/python, Windows: ...\raise-my-agent\.venv\Scripts\python.exe)
   Windows 경로는 JSON 안에서 "C:\\Users\\username\\...\\study_server.py"처럼 역슬래시를 두 번씩 적습니다.
   여러 서버를 함께 등록하려면 "mcpServers" 안에 항목을 더합니다. 4절의 예시(Node.js의 npx가 필요합니다):
       "filesystem": {
         "command": "npx",
         "args": ["-y", "@modelcontextprotocol/server-filesystem", "/Users/username/Desktop"]
       }
3. 저장한 뒤 Claude Desktop을 완전히 종료했다가(macOS: Dock에서 우클릭 -> 종료, Windows: 시스템 트레이에서 우클릭 -> Exit)
   다시 시작합니다. 입력창 아래의 도구(망치) 아이콘에서 create_note와 quiz가 보이면 연결된 것입니다.
   연결되지 않으면 Settings -> Developer 탭의 로그에서 오류 메시지를 확인합니다.
4. 대화창에서 "수학 노트를 만들어줘. 이차방정식 근의 공식을 정리해서."처럼 말로 요청하면 create_note가 호출됩니다.
   Cursor, Claude Code 등 MCP를 지원하는 다른 앱에도 연결 대상만 바꿔 같은 서버를 쓸 수 있습니다.

표시: [보충] 실행에 필요한 코드, [수정] 실행에 맞게 고친 코드, [설명용 코드] 실행되지 않는 설명용 조각
"""

# %% 4. 실습: 학습 도우미 Server 만들기 - ① Tool 정의하기: 실행 가능한 기능
# FastMCP는 MCP Server를 만드는 데 가장 널리 쓰이는 프레임워크로, 아래 명령으로 설치합니다
# (리포지토리 루트에서 uv sync를 실행하면 함께 설치됩니다).
#     pip install fastmcp
# Tool은 LLM이 실행할 수 있는 함수(행동)로, 수업 중 실습 과제에 해당합니다. 2장에서 @tool로 만든 cry, poo, eat과
# 구조가 비슷하고 @mcp.tool()을 쓴다는 점만 다르며, 이렇게 하면 이 도구들에 MCP 프로토콜로 외부에서 접근할 수 있습니다.
# study_server.py
from fastmcp import FastMCP
from datetime import datetime

mcp = FastMCP("Study Helper Server")

# [보충] 작업 폴더를 이 파일의 폴더로 바꾸고, 노트를 저장할 notes/ 폴더를 만듭니다.
import os
from pathlib import Path

os.chdir(Path(__file__).resolve().parent)
os.makedirs("notes", exist_ok=True)

# ── Tool: 실행 가능한 함수 (행동) ──────────────────────────────

@mcp.tool()
def create_note(subject: str, content: str) -> str:
    """과목별 학습 노트를 생성합니다."""
    # Windows에서는 파일 이름에 콜론(:)을 쓸 수 없으므로 "%H:%M"을 "%H-%M"으로 바꿔 실행합니다.
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    filename = f"notes/{subject}_{timestamp}.txt"

    with open(filename, "w") as f:
        f.write(f"[{subject}] {timestamp}\n\n{content}")

    return f"노트가 생성되었습니다: {filename}"

@mcp.tool()
def quiz(subject: str, question: str, answer: str) -> dict:
    """학습 내용을 퀴즈로 확인합니다."""
    return {
        "subject": subject,
        "question": question,
        "user_answer": answer,
        "status": "채점 완료",
        "feedback": f"'{subject}' 과목의 답변이 기록되었습니다."
    }


# %% ② Resource 정의하기: 읽기 전용 데이터
# Resource는 도구와 달리 무언가를 실행하지 않고 정보만 제공하는 읽기 전용 데이터(교재, 참고 자료)입니다.
# @mcp.resource("study://curriculum")처럼 URI 형식으로 접근하는 것이 특징입니다. 두 번째 리소스의 {subject} 부분은
# 리소스 템플릿으로, study://curriculum/수학 으로 요청하면 수학 커리큘럼만 돌려줍니다.
# ── Resource: 읽기 전용 데이터 (정보) ──────────────────────────

CURRICULUM = {
    "수학": ["함수", "미적분", "확률과 통계"],
    "영어": ["문법", "독해", "작문"],
    "코딩": ["Python 기초", "자료구조", "알고리즘"],
}

@mcp.resource("study://curriculum")
def get_curriculum() -> dict:
    """전체 과목의 커리큘럼을 제공합니다."""
    return CURRICULUM

# URI의 {subject} 자리에 들어온 값(예: 수학)이 함수의 subject 인자로 전달됩니다.
@mcp.resource("study://curriculum/{subject}")
def get_subject_curriculum(subject: str) -> list:
    """특정 과목의 커리큘럼을 제공합니다."""
    return CURRICULUM.get(subject, [f"'{subject}' 과목을 찾을 수 없습니다."])


# %% ③ Prompt 정의하기: 대화 템플릿
# @mcp.prompt()로 만드는 Prompt는 Tool이나 Resource와 달리 LLM이 대화를 시작할 때 참고하는 틀(수업 커리큘럼)입니다.
# 과목(subject)과 목표(goal)를 받아, 4주짜리 주차별 학습 계획을 요청하는 문장을 만들어 줍니다.
# ── Prompt: 미리 정의된 대화 템플릿 ─────────────────────────────

@mcp.prompt()
def study_plan(subject: str, goal: str) -> str:
    """과목별 학습 계획을 세우는 프롬프트를 생성합니다."""
    return f"""
    다음 조건에 맞는 학습 계획을 세워주세요:
    - 과목: {subject}
    - 목표: {goal}
    - 형식: 주차별 계획표 (4주)
    - 각 주차마다 핵심 주제와 실습 과제를 포함해주세요.
    """


# %% ④ 서버 실행하기
# Tool(create_note, quiz), Resource(curriculum, curriculum/{subject}), Prompt(study_plan)를 모두 갖춘 서버를 띄웁니다.
# ── 서버 실행 ──────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run()  # 기본: stdio 모드로 실행


# %% 5. 서버 실행 방식 선택하기
# 1절에서 본 것처럼 Server는 로컬(stdio)과 원격(HTTP) 두 방식으로 실행할 수 있고, FastMCP에서는 mcp.run()의
# 인자 하나로 바꿉니다. stdio는 내 컴퓨터에서 혼자 쓸 때, HTTP는 팀원이 네트워크로 접속할 때 씁니다.
# 방법 2처럼 host="0.0.0.0"으로 띄우면 같은 네트워크의 다른 컴퓨터에서도 이 서버에 접속할 수 있습니다.
# [설명용 코드] 위 ④의 mcp.run()을 아래 두 방법 중 하나로 바꿔 실행 방식을 고릅니다.
# # 방법 1: stdio 모드 (로컬, 기본값)
# mcp.run()
#
# # 방법 2: HTTP 모드 (원격)
# mcp.run(transport="http", host="0.0.0.0", port=8000, path="/mcp")
