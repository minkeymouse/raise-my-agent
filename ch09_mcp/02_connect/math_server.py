"""
[9장 2절] [Connect] 전문가의 힘 빌리기

이 절에서 배우는 것:
- 이미 있는 MCP Server(방과후 외부강사)에 연결하기 전에, MCP Server가 어떻게 생겼는지 간단한 수학 연산 서버로 살펴봅니다.
- @mcp.tool()을 붙인 함수는 MCP 표준에 맞춰 포장되어, 어떤 MCP Client든 연결해서 쓸 수 있는 도구가 됩니다.
- 이 서버에 연결해 도구를 가져오고 에이전트에게 쥐여 주는 코드는 같은 폴더의 math_client.py에 있습니다.

실행 방법:
    uv run python ch09_mcp/02_connect/math_client.py
    (math_client.py가 이 서버를 stdio 방식으로 직접 띄우므로 따로 실행할 필요가 없습니다.
     이 파일만 단독으로 실행하면 클라이언트의 연결을 기다리며, Ctrl+C로 종료합니다.)
필요한 환경 변수: 없음 (LLM을 호출하지 않습니다)
"""

# %% 2. 연결할 MCP Server 준비하기
# 실습 환경을 갖추려고 서버를 직접 만들지만, 실제로는 누군가 만들어 둔 Server에 연결하는 것이 MCP의 핵심 가치입니다.
# 2장의 @tool 도구는 코드에서 직접 import하는 내 에이전트 전용이었다면, @mcp.tool() 도구는
# MCP 표준을 지원하는 모든 앱이 네트워크(stdio/HTTP)로 연결해 쓸 수 있습니다.
# math_server.py
from fastmcp import FastMCP

# FastMCP로 MCP Server를 만듭니다("Math Server"는 서버 이름).
mcp = FastMCP("Math Server")

# @mcp.tool(): 함수를 MCP 표준 도구로 공개합니다. docstring은 Client가 가져가는 도구 설명이 됩니다.
@mcp.tool()
def add(a: int, b: int) -> int:
    """두 숫자를 더합니다."""
    return a + b

@mcp.tool()
def multiply(a: int, b: int) -> int:
    """두 숫자를 곱합니다."""
    return a * b

# 기본값인 stdio 모드로 실행합니다. 로컬 프로세스로 떠서 표준 입출력(stdin/stdout)으로 Client와 통신합니다.
if __name__ == "__main__":
    mcp.run()
