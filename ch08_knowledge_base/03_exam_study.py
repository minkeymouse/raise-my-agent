"""
[8장 3절] 에이전트 시험 공부 하기 - 1. 시험 문제 분석하기 ~ 2. 노트 기반 문제 풀이

이 절에서 배우는 것:
- 문학 문제에는 서술적 특징, 상징 해석, 작품 이해, 시대 배경 해석처럼 반복되는 유형이 있고, 유형마다 필요한 메모가 다릅니다.
- 문제는 (1) 문제에서 키워드 추출 -> (2) 지식베이스에서 관련 노트 검색 -> (3) 검색 결과를 종합해 답변 생성 순서로 풉니다.
- 벡터 검색은 문제와 노트의 표현이 달라도 의미가 비슷하면 찾아 줍니다. 다만 관계·사건·해석 문제는 무엇이 정답을 결정하는지
  흐려질 수 있어, Neo4j에서 관계를 먼저 좁히고 스토어에서 근거 문장을 보강하는 편이 안정적입니다.

실행 방법:
    uv run python ch08_knowledge_base/03_exam_study.py
필요한 환경 변수: OPENAI_API_KEY
책의 예시는 새로 만든 빈 스토어를 검색하므로 검색 결과 없이 LLM의 풀이만 출력됩니다.
3절의 '3. 시험공부 에이전트 구축'은 chapter8/ 패키지이며, 실행 방법은 chapter8/__main__.py 머리말에 있습니다.
"""

# .env 파일의 API 키를 환경 변수로 불러옵니다.
from dotenv import load_dotenv
load_dotenv()

# %% 1. 시험 문제 분석하기 - 문제에서 키워드 추출하기
# 기출 40번은 서술적 특징과 효과를 묻는 문제라서 문체, 서술 기법, 어조에 대한 메모가 필요합니다.
# 문제와 보기에서 핵심 키워드를 뽑아 지식베이스 검색에 씁니다. 여기서는 키워드를 직접 적었고, chapter8에서는 LLM이 뽑습니다(ExamKeywords).
# 예시: 문제 40번 분석
question = """
위 글의 서술적 특징과 효과를 <보기>에서 고른 것은?
<보 기>
ㄱ. 독백적인 어조로 현실과 단절된 의식 상태를 표현하고 있다.
ㄴ. 단정적이고 객관적인 진술로 사건에 사실성을 부여하고 있다.
ㄷ. 회상의 기법을 사용하여 현재와 과거의 화해를 지향하고 있다.
ㄹ. 비유적 표현으로 인물의 생각과 인상을 구체적으로 제시하고 있다.
"""

# 키워드 추출
keywords = ["서술적 특징", "독백적 어조", "의식 상태", "비유적 표현"]


# %% 2. 노트 기반 문제 풀이 - 키워드 기반 검색
# 2절에서 노트를 그래프까지 넓혔으므로 검색은 두 가지를 섞습니다. 스토어(벡터 검색)로 보기 문장과 비슷한 근거 메모를 끌어오고,
# Neo4j(관계 탐색)로 관계 자체가 답인 뼈대를 뽑습니다. 이 셀은 스토어 쪽으로, 키워드를 query로 넘겨 《날개》 메모 서랍에서 찾습니다.
from langgraph.store.memory import InMemoryStore
from langchain.embeddings import init_embeddings

store = InMemoryStore(
    index={
        "embed": init_embeddings("openai:text-embedding-3-small"),
        "dims": 1536,
    }
)

# 문제에서 키워드 추출 (LLM이 추출)
question_keywords = "독백적 어조, 의식 상태, 비유적 표현"

# 벡터 검색으로 관련 메모 찾기
user_id = "student_001"
ns_memos = ("literature", "memos", user_id, "날개")
relevant_notes = list(store.search(
    ns_memos,
    query=question_keywords,
    limit=5
))

# 검색 결과 확인
for note in relevant_notes:
    print(f"타입: {note.value['memo_type']}")
    print(f"내용: {note.value['content']}")
    print(f"태그: {note.value['tags']}")
    print("---")


# %% 2. 노트 기반 문제 풀이 - 검색 결과를 종합하여 답변 생성
# 검색된 메모를 '관련 노트' 컨텍스트로 엮어 SystemMessage에 넣고, 문제와 보기는 HumanMessage로 넘겨 노트를 근거로 풀게 합니다.
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 검색된 노트를 컨텍스트로 구성
context = "관련 노트:\n"
for note in relevant_notes:
    context += f"- [{note.value['memo_type']}] {note.value['content']}\n"

# 문제와 보기
question_text = """
위 글의 서술적 특징과 효과를 <보기>에서 고른 것은?
<보 기>
ㄱ. 독백적인 어조로 현실과 단절된 의식 상태를 표현하고 있다.
ㄴ. 단정적이고 객관적인 진술로 사건에 사실성을 부여하고 있다.
ㄷ. 회상의 기법을 사용하여 현재와 과거의 화해를 지향하고 있다.
ㄹ. 비유적 표현으로 인물의 생각과 인상을 구체적으로 제시하고 있다.
"""

# LLM에게 컨텍스트와 함께 질문
system_msg = SystemMessage(content=f"""당신은 문학 기출 문제를 푸는 전문가입니다.

{context}

위 노트를 참고하여 문제를 정확히 풀이하세요. 정답 번호와 그 이유를 설명하세요.""")

response = llm.invoke([system_msg, HumanMessage(content=question_text)])
print(response.content)


# %% 3. 시험공부 에이전트 구축
# 저장소(벡터 메모) + Neo4j(관계·해석)를 함께 쓰는 시험공부 에이전트는 chapter8/ 패키지에 있습니다.
# 실행 방법(ingest, exam)은 chapter8/__main__.py 머리말에, 기출 문항 파일 준비는 chapter8/exam_agent.py 머리말에 있습니다.
