"""
[4장 1절] 스튜디오 실행 - 1. 파일을 통한 실행 (run_research_agent.py)

이 절에서 배우는 것:
- 에이전트와 대화하려면 인터페이스가 필요합니다. 첫 번째 방법은 파일로 실행해 터미널에서 대화하는 것입니다.
- main 함수의 while 루프가 질문을 계속 받고, exit, quit, 종료 중 하나를 입력하면 끝납니다.
- 질문마다 현재 날짜로 Context를 새로 만들고, 새 메시지만 전달합니다(앞의 대화는 체크포인터에 저장됩니다).
- graph.stream으로 진행 상황을 스트리밍하며 도구 호출과 답변을 함께 출력합니다.
두 번째 방법인 LangGraph Studio 실행은 agent/baby_research.py 머리말에 있습니다.

4장 3절의 질의와 보고서 생성도 이 파일로 실행합니다. 같은 thread_id("1")로 이어지므로 앞의 대화를 기억합니다.
    1. 요즘 유행하는 티니핑에 대해 알려줘 (4장 1절)
    2. 최근 유행하는 어린이 간식에 대해 알려줘 (4장 3절)
    3. 두쫀쿠에 대한 결과는 없어? (4장 3절)
    4. 지금까지의 내용을 바탕으로 보고서를 작성해줘. (4장 3절, ch04_research_agent/output/report.md에 저장)
대화 기록은 메모리(MemorySaver)에만 있어 다시 실행하면 새 대화로 시작합니다. 한 대화에서 모델을 10번 호출하면
ModelCallLimitMiddleware(thread_limit=10) 때문에 그 뒤의 질문에는 "Model call limits exceeded: thread limit (10/10)"만
출력됩니다. 이때는 프로그램을 끝내고 다시 실행합니다.

실행 방법:
    uv run python ch04_research_agent/run_research_agent.py
    (책처럼 폴더 안에서: cd ch04_research_agent && uv run python run_research_agent.py)
필요한 환경 변수 (리포지토리 루트의 .env): OPENAI_API_KEY, NAVER_CLIENT_ID, NAVER_CLIENT_SECRET
    네이버 키 발급 방법은 agent/baby_research.py 머리말의 준비물에 있습니다.
    선택(LangSmith 트레이싱, 4장 1절): LANGSMITH_TRACING=true, LANGSMITH_API_KEY, LANGSMITH_PROJECT
    LangSmith(https://smith.langchain.com)에서 프로젝트를 만들고 그 이름을 LANGSMITH_PROJECT에 넣습니다.
    create_agent로 만든 에이전트는 코드를 바꾸지 않아도 모든 실행 단계가 LangSmith에 기록됩니다.
"""

# .env 파일의 API 키를 환경 변수로 불러옵니다.
from dotenv import load_dotenv
load_dotenv()

# %% 1. 파일을 통한 실행
# main 함수에서 사용자와 대화합니다. 파일을 통한 실행은 지금까지 해 온 방식을 복습하는 셈입니다.
# 체크포인터에 대화가 자동으로 저장되므로, 매번 새로운 메시지만 에이전트에 전달합니다.
# 출력은 스트리밍 방식이며, 자세히 추적할 수 있도록 도구 호출도 함께 보여 줍니다.
"""리서치 에이전트 만들기"""

from agent.baby_research import graph, BabyContext
from langchain_core.messages import HumanMessage


def main():
    """사용자 쿼리로 에이전트 실행"""
    # thread_id 설정 (대화 이어가기)
    config = {"configurable": {"thread_id": "1"}}

    # 연속 대화 루프
    while True:
        # 사용자 입력
        query = input("\n 질문: ").strip()
        # 종료 조건 확인
        if query.lower() in ["exit", "quit", "종료"]:
            print("\n안녕! 다음에 또 만나자!")
            break

        if not query:
            continue

        # 매번 현재 날짜로 Context 생성 (최신 날짜 유지)
        context = BabyContext.create()

        # 새로운 메시지만 전달 (이전 대화는 state에 저장됨)
        messages = [HumanMessage(content=query)]

        print(f"\n현재 날짜: {context.current_date}")
        print("리서치 중...\n")
        print("-" * 60)

        # 진행 상황을 보기 위해 스트리밍
        for chunk in graph.stream({"messages": messages}, config, context=context):
            # 가장 최근 메시지를 추출하여 출력하는 스트리밍 루프
            for node_name, node_state in chunk.items():
                # stream은 노드마다 {노드 이름: 바뀐 상태}를 돌려줍니다. 바꾼 상태가 없는 미들웨어 노드는 None입니다.
                if node_state is None:
                    continue
                if "messages" in node_state and node_state["messages"]:
                    last_message = node_state["messages"][-1]

                    # 도구 호출이 있을 경우 출력합니다.
                    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                        for tool_call in last_message.tool_calls:
                            print(f"\n 도구: {tool_call['name']}")
                            print(f"   변수: {tool_call['args']}")

                    # 내용이 있는 경우 출력합니다.
                    if hasattr(last_message, "content") and last_message.content:
                        if last_message.content.strip():
                            print(f"\n {last_message.content}")

        print("-" * 60)


if __name__ == "__main__":
    main()


# [책의 실행 결과] 4장 1절 - 질문: 요즘 유행하는 티니핑에 대해 알려줘
# 요즘 티니핑은 여러 가지 활동과 이벤트로 인기를 끌고 있어요! 2026년에는 특히 "티니핑런"이라는 어린이 마라톤이 주목받고 있답니다. 이 마라톤은 단순한 달리기 이상의 의미를 가지고 있어서 많은 가족들이 참여하고 있어요. 휠라키즈와 함께하는 이 이벤트는 서울 여의도 물빛광장에서 진행되고, 5세에서 9세 어린이들이 참여할 수 있어요.
#
# 또한, 티니핑은 다양한 캐릭터 상품과 함께 어린이들 사이에서 꾸준한 인기를 얻고 있어요. 특히, 티니핑 캐릭터가 포함된 키링이나 스티커 같은 제품들도 많이 사랑받고 있답니다.
#
# 이렇게 티니핑은 다양한 방식으로 어린이들과 가족들에게 즐거움을 주고 있답니다!

# 4장 3절: TodoListMiddleware가 작업을 계획하고, 여러 검색 도구를 차례로 호출한 뒤 최종 응답을 만듭니다.
# '최근'이라는 말을 알아듣고 검색 쿼리에 현재 연도를 넣는지 도구 호출의 변수에서 확인할 수 있습니다.
# [책의 실행 결과] 4장 3절 - 질문: 최근 유행하는 어린이 간식에 대해 알려줘
# > 요즘 어린이 간식 트렌드에 대해 알아봤어! 요거트, 특히 "스키르 요거트"가 인기야. 이건 고단백 간식으로, 웰니스 트렌드에 맞춰 건강한 선택으로 주목받고 있어. 또, 귀여운 모양의 과자들도 인기가 많아. 예를 들어, "고래밥" 시리즈는 귀여운 모양과 휴대성 덕분에 어린이들이 좋아하는 간식으로 꼽히고 있어.
#
# 에어프라이어로 간단하게 만들 수 있는 떡 간식도 요즘 많이 찾는다고 해. 떡을 에어프라이어에 구워서 간단하게 즐길 수 있대.
#
# 이런 간식들은 건강과 재미를 동시에 잡으려는 부모님들에게 좋은 선택이 될 것 같아!
#
# 더 자세한 정보는 [여기](https://blog.naver.com/jjoung21/224175179286)와 [여기](https://blog.naver.com/nymnym-1102/224149164869)에서 확인할 수 있어!

# 답변이 의도와 다를 때는 이처럼 사용자가 힌트를 주거나 검색 범위를 좁혀 줄 수 있습니다.
# [책의 실행 결과] 4장 3절 - 이어서 질문: 두쫀쿠에 대한 결과는 없어?
# > 두쫀쿠는 "두바이 쫀득 쿠키"의 줄임말로, 최근 인기를 끌고 있는 디저트예요. 여러 기사에서 두쫀쿠에 대한 이야기가 나오고 있는데, 특히 유명 인사들이 두쫀쿠를 선물하거나 관련된 미담이 많이 전해지고 있어요. 예를 들어, 고윤정이 제작진에게 두쫀쿠를 선물하며 훈훈한 미담을 만들었다는 이야기가 있어요. 또한, 두쫀쿠의 창시자인 김나리 제과장이 레시피를 공개한 이유에 대한 기사도 있답니다. 두쫀쿠는 최근까지 인기를 끌었지만, 유행이 빠르게 지나가고 있다는 평가도 있어요.
#
# [관련 기사 보기](https://m.entertain.naver.com/article/003/0013780242)

# 에이전트는 맥락을 보고 도구를 고르므로 항상 보고서를 만들지는 않습니다. 그래서 보고서 작성을 직접 요청합니다.
# 참고: 아래 결과의 경로는 저자의 컴퓨터 경로이며, 이 리포지토리에서는 ch04_research_agent/output/report.md에 저장됩니다.
# [책의 실행 결과] 4장 3절 - 이어서 질문: 지금까지의 내용을 바탕으로 보고서를 작성해줘.
# > 보고서가 /Users/minkeychang/mybooks/output/report.md에 저장되었습니다.
#
# > 보고서를 작성해서 저장했어! [여기서](sandbox:/Users/minkeychang/mybooks/output/report.md) 확인할 수 있어. 보고서에는 2026년 유행하는 어린이 간식에 대한 정보가 담겨 있어. 맛있고 건강한 간식들이 많이 소개되어 있으니 참고해봐!
