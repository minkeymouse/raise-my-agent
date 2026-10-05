"""
[9장 2절] [Connect] 전문가의 힘 빌리기

이 절에서 배우는 것:
- langchain-mcp-adapters의 MultiServerMCPClient가 1절에서 본 Client(담임선생님) 역할을 맡아 MCP Server에 연결합니다.
- get_tools() 한 번으로 Server가 제공하는 도구 목록과 설명을 LangChain 호환 도구로 가져옵니다.
- 여러 Server를 stdio(로컬)와 HTTP(원격)로 섞어 동시에 연결할 수 있습니다.
- 가져온 도구를 create_agent로 만든 에이전트에게 쥐여 주면, 에이전트가 MCP 도구를 스스로 골라 호출합니다.

실행 방법:
    uv run python ch09_mcp/02_connect/math_client.py
    같은 폴더의 math_server.py는 이 스크립트가 stdio 방식으로 직접 띄웁니다. 함께 출력되는 FastMCP 배너와
    서버 로그(Starting MCP server ...), 경고 메시지는 정상입니다. 배너의 업데이트 안내(pip install --upgrade fastmcp)는
    무시합니다(이 리포는 이 장의 코드와 함께 동작하는 fastmcp 2.x를 설치합니다).
    asyncio.run()은 주피터 커널 안에서 호출할 수 없으므로 # %% 셀 단위가 아니라 위 명령처럼 스크립트로 실행합니다.
필요한 환경 변수: ANTHROPIC_API_KEY (리포지토리 루트의 .env 파일에 넣습니다. 발급: https://platform.claude.com/)
준비물: "4. 여러 Server를 동시에 연결하기"는 http://localhost:8000/mcp 에 HTTP MCP Server(weather)가 떠 있어야 합니다.
    책에는 이 서버 코드가 없어, 서버가 없으면 4에서 연결 오류로 멈추고 5는 실행되지 않습니다.
    끝까지 실행하려면 3절 study_server.py의 ④ mcp.run()을 책의 '방법 2: HTTP 모드' 코드로 바꿔 다른 터미널에서
    uv run python ch09_mcp/03_build/study_server.py 로 먼저 띄웁니다(weather 자리에 학습 도우미 서버가 연결됩니다).
    확인한 뒤에는 ④를 다시 mcp.run()으로 되돌립니다.

표시: [보충] 실행을 위해 더한 코드, [수정] 책 코드의 오류를 고친 곳, [설명용 코드] 실행되지 않는 설명용 조각
"""

# %% 1. 환경 설정 및 라이브러리 설치
# 책은 아래 명령으로 패키지를 설치합니다. 이 리포에서는 루트에서 uv sync를 실행하면 함께 설치됩니다.
#     uv pip install langchain-mcp-adapters langgraph langchain
# langchain-mcp-adapters는 MCP 도구를 LangChain/LangGraph 호환 도구로 바꿔 주는 어댑터이고,
# langgraph는 에이전트의 사고 흐름(그래프)을, langchain은 LLM과 도구를 연결하는 기본 틀을 맡습니다.
import asyncio
from dotenv import load_dotenv, find_dotenv

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent

# 환경 변수 로드 (.env)
load_dotenv(find_dotenv())

# [보충] 책의 "./math_server.py"는 작업 폴더 기준 상대 경로입니다. 리포 루트에서 실행해도 서버 파일을 찾도록
#        서버 설정마다 "cwd": HERE를 넣어 서버를 이 파일이 있는 폴더(ch09_mcp/02_connect/)에서 실행합니다.
from pathlib import Path

HERE = Path(__file__).resolve().parent


# %% 3. MCP Server에 연결하기: MultiServerMCPClient
# 1절의 구조처럼 Host 안에 Client가 생기고 각 Client가 Server에 1:1로 연결되는데, MultiServerMCPClient는
# 이 Client 역할을 코드로 구현한 것입니다. "command"와 "args"로 서버를 로컬 프로세스로 띄우고 stdio로 통신합니다.
# get_tools()를 부르면 능력 협상(Capability Negotiation)이 일어나, math_server.py를 import하지 않고도 도구 목록과 설명을 가져옵니다.
# [보충] 스크립트에서는 함수 밖의 await가 문법 오류이므로, 책의 코드를 async 함수로 감싸 asyncio.run()으로 실행합니다.
async def connect_math_server():
    # MCP Server에 연결하는 Client 생성
    client = MultiServerMCPClient(
        {
            "math": {
                "command": "python",
                "args": ["./math_server.py"],  # 위에서 만든 서버 파일의 경로
                "transport": "stdio",           # 로컬 프로세스 통신
                "cwd": HERE,                    # [보충] 서버를 이 파일이 있는 폴더에서 실행
            },
        }
    )

    # MCP 도구를 LangChain 호환 도구로 변환
    tools = await client.get_tools()

    print(f"가져온 도구 수: {len(tools)}")
    for tool in tools:
        print(f"  - {tool.name}: {tool.description}")


asyncio.run(connect_math_server())  # [보충]
# [책의 실행 결과]
# 가져온 도구 수: 2
#   - add: 두 숫자를 더합니다.
#   - multiply: 두 숫자를 곱합니다.


# %% 4. 여러 Server를 동시에 연결하기
# MultiServerMCPClient라는 이름처럼 dict에 Server를 여러 개 나열하면 동시에 연결됩니다.
# stdio는 외부강사가 학교에 직접 출강하는 것(로컬 프로세스), http는 온라인 화상수업(원격 서버)에 해당합니다.
# 원격 Server는 "command" 대신 "url"에 주소를 적으며, 전송 방식이 달라도 get_tools() 한 번으로 모든 도구를 통합해 가져옵니다.
# 이 셀은 http://localhost:8000/mcp 에 HTTP MCP Server가 떠 있어야 합니다(파일 맨 위 설명의 준비물 참고).
async def connect_multi_servers():  # [보충] 최상위 await를 함수로 감쌉니다
    client = MultiServerMCPClient(
        {
            "math": {
                "command": "python",
                "args": ["./math_server.py"],
                "transport": "stdio",            # 로컬: 직접 출강
                "cwd": HERE,                     # [보충] 서버를 이 파일이 있는 폴더에서 실행
            },
            "weather": {
                "url": "http://localhost:8000/mcp",
                "transport": "http",             # 원격: 온라인 수업
            },
        }
    )

    tools = await client.get_tools()


asyncio.run(connect_multi_servers())  # [보충]


# %% 5. LangGraph 에이전트에 연결하기
# 가져온 MCP 도구를 create_agent(model, tools)에 넣어 에이전트에게 쥐여 줍니다. 에이전트는 질문을 분석해
# add와 multiply를 스스로 골라 호출하고 결과를 조합해 답합니다. 가져온 MCP 도구는 비동기로만 실행되므로 ainvoke를 씁니다.
# 실전에서는 GitHub, Slack, PostgreSQL 등 이미 공개된 MCP Server(https://github.com/modelcontextprotocol/servers)도
# 이처럼 MultiServerMCPClient에 서버 정보만 넣으면 연결됩니다.
async def main():
    # 1. MCP Server 연결 & 도구 가져오기
    client = MultiServerMCPClient(
        {
            "math": {
                "command": "python",
                "args": ["./math_server.py"],
                "transport": "stdio",
                "cwd": HERE,  # [보충] 서버를 이 파일이 있는 폴더에서 실행
            },
        }
    )
    tools = await client.get_tools()

    # 2. 에이전트 생성 (LLM + MCP 도구)
    agent = create_agent(
        "anthropic:claude-sonnet-4-6",  # 사용할 LLM
        tools                            # MCP에서 가져온 도구들
    )

    # 3. 에이전트에게 질문하기
    response = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "(3 + 5) x 12는 얼마야?"}]}
    )

    print(response["messages"][-1].content)

# 실행
asyncio.run(main())
# [책의 실행 결과]
# (3 + 5) × 12 = 96입니다.
# 먼저 add(3, 5)를 호출하여 8을 얻고, multiply(8, 12)를 호출하여 96을 계산했습니다.
