"""
[9장 3절] [Build] 우리 집 비법 공유하기: 아이의 재능을 세상의 표준으로

이 절에서 배우는 것:
- 2절에서 배운 MultiServerMCPClient로, 이번에는 우리가 직접 만든 study_server.py에 연결합니다.
- get_tools()로 가져온 도구(create_note, quiz)를 확인하고, 에이전트에게 쥐여 주어 수학 노트를 만들게 합니다.

실행 방법:
    uv run python ch09_mcp/03_build/study_client.py
    같은 폴더의 study_server.py는 이 스크립트가 stdio 방식으로 직접 띄웁니다. 함께 출력되는 FastMCP 배너와
    서버 로그(Starting MCP server ...), 경고 메시지는 정상이며, 배너의 업데이트 안내(pip install --upgrade fastmcp)는
    무시합니다(이 리포는 이 장의 코드와 함께 동작하는 fastmcp 2.x를 설치합니다).
    asyncio.run()은 주피터 커널 안에서 호출할 수 없으므로 # %% 셀 단위가 아니라 위 명령처럼 스크립트로 실행합니다.
필요한 환경 변수: ANTHROPIC_API_KEY (리포지토리 루트의 .env 파일에 넣습니다. 발급: https://platform.claude.com/)
에이전트가 만든 노트는 ch09_mcp/03_build/notes/ 폴더에 저장됩니다.

표시: [보충] 실행을 위해 더한 코드, [수정] 책 코드의 오류를 고친 곳, [설명용 코드] 실행되지 않는 설명용 조각
"""

# .env 파일의 API 키를 환경 변수로 불러옵니다.
from dotenv import load_dotenv
load_dotenv()

# [보충] 책의 "./study_server.py"는 작업 폴더 기준 상대 경로입니다. 리포 루트에서 실행해도 서버 파일을 찾도록
#        서버 설정에 "cwd": HERE를 넣어 서버를 이 파일이 있는 폴더(ch09_mcp/03_build/)에서 실행합니다.
from pathlib import Path

HERE = Path(__file__).resolve().parent


# %% 6. 우리 Server에 연결해 보기
# 2절에서 배운 방법 그대로, 우리가 만든 study_server.py를 stdio로 띄워 연결하고 도구를 가져옵니다.
# get_tools()는 Server의 Tool만 도구로 가져오므로, 목록에는 Resource와 Prompt 없이 create_note와 quiz가 보입니다.
# 에이전트는 요청을 보고 create_note 도구를 골라 호출합니다. MCP 도구는 비동기로만 실행되므로 ainvoke를 씁니다.
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent

async def main():
    # 1. 우리가 만든 MCP Server에 연결
    client = MultiServerMCPClient(
        {
            "study": {
                "command": "python",
                "args": ["./study_server.py"],
                "transport": "stdio",
                "cwd": HERE,  # [보충] 서버를 이 파일이 있는 폴더에서 실행
            },
        }
    )
    tools = await client.get_tools()

    # 2. 가져온 도구 확인
    print(f"가져온 도구 수: {len(tools)}")
    for tool in tools:
        print(f"  - {tool.name}: {tool.description}")

    # 3. 에이전트에 연결하여 사용
    agent = create_agent("anthropic:claude-sonnet-4-6", tools)

    response = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "수학 노트를 만들어줘. 내용은 '이차방정식의 근의 공식 정리'로."}]}
    )

    print(response["messages"][-1].content)

asyncio.run(main())
# [책의 실행 결과]
# 가져온 도구 수: 2
#   - create_note: 과목별 학습 노트를 생성합니다.
#   - quiz: 학습 내용을 퀴즈로 확인합니다.
#
# create_note 도구를 사용하여 수학 노트를 생성했습니다!
# 노트가 생성되었습니다: notes/수학_2026-03-21 14:30.txt
