"""
[8장 1절] [neo4j] 지식 베이스 이해하기

이 절에서 배우는 것:
- 체크포인트가 한 대화를 이어 가는 '단기 메모장'이라면, 스토어(Store)는 대화가 끝난 뒤에도 다시 꺼내 쓸 기록을 모아 두는 '서재'입니다.
- 스토어는 namespace(어느 서랍에 넣을지), key(서랍 안에서 무엇을 찾을지), value(실제 내용)로 정보를 나누고,
  put()으로 저장, get()과 search()로 조회합니다.
- 임베딩 인덱스를 붙인 스토어는 search(..., query=...)로 키를 몰라도 의미가 가까운 노트를 찾아 줍니다.
- 그래프 데이터베이스(Neo4j)는 노드와 관계로 데이터를 저장합니다. 같은 절의 Cypher 예제는 01_cypher_basics.cypher에 있습니다.

실행 방법:
    uv run python ch08_knowledge_base/01_store_basics.py
필요한 환경 변수: OPENAI_API_KEY (마지막 '벡터 Store' 셀의 임베딩 모델)
표시: [보충] 실행을 위해 더한 코드, [수정] 책 코드의 오류를 고친 곳, [설명용 코드] 실행되지 않는 설명용 조각

책의 코드는 결과를 변수(all_memories, specific_memory, top)에 담기만 하므로 실행해도 출력이 없습니다.
VS Code나 PyCharm에서 # %% 셀 단위로 실행한 뒤 변수를 확인합니다.

8장 준비물 (2절의 Neo4j 예제와 3절의 시험공부 에이전트에 필요합니다):
- Neo4j Desktop: https://neo4j.com/download/ 에서 받아 설치하고, 왼쪽 Local Instances에서 create instance로 인스턴스를 만듭니다
  (책의 예: 이름 research_agent, 권장(recommended) 버전, 접속 계정과 비밀번호). 접속 주소는 bolt://localhost:7687입니다.
- 또는 Neo4j Aura(클라우드): Aura 콘솔(https://console.neo4j.io)에서 인스턴스를 만들고, 만들 때 내려받는 자격 증명 파일의
  NEO4J_URI(neo4j+s://...), NEO4J_USERNAME, NEO4J_PASSWORD, NEO4J_DATABASE 값을 씁니다. 어느 쪽이든 코드는 같고 URI만 다릅니다.
- 리포지토리 루트의 .env에 아래 값을 적습니다(cp .env.example .env). Aura라면 NEO4J_URI에 neo4j+s://... 주소를 넣고 NEO4J_DATABASE도 적습니다.
      NEO4J_URI=bolt://localhost:7687
      NEO4J_USERNAME=neo4j
      NEO4J_PASSWORD=생성할_때_설정한_비밀번호
      OPENAI_API_KEY=sk-...
  책은 .env.local을 안내합니다. 3절의 chapter8 패키지는 .env.local을 먼저 읽고 .env로 보완하지만, 01~03 파일은 .env만 읽습니다.
  NEO4J_* 값은 chapter8 패키지가 읽습니다. 2절 예제(02a, 02b, 02d)는 책처럼 접속 정보를 코드에 적으므로 YOUR_PASSWORD를 직접 고칩니다.
책은 1절과 2절의 개념과 코드를 먼저 읽고, 3절의 chapter8 패키지를 직접 돌려 보는 흐름을 권장합니다.
"""

# .env 파일의 API 키를 환경 변수로 불러옵니다.
from dotenv import load_dotenv
load_dotenv()

# %% 2. 다양한 Store 예시 - InMemoryStore
# InMemoryStore는 데이터를 메모리(RAM)에 저장하므로 프로그램이 끝나면 사라집니다. 개발·테스트 단계의 빠른 프로토타입에 알맞습니다.
# namespace는 묶음 단위(폴더), key는 개별 항목(파일명) 역할을 합니다. 예를 들어 ("user_001", "preferences")처럼 사용자별로 나눌 수 있습니다.
# search(namespace)는 그 네임스페이스의 항목을 묶음으로 가져오고, get(namespace, key)는 항목 하나를 정확히 지정해 꺼냅니다.
from langgraph.store.memory import InMemoryStore

# 메모리 스토어 초기화
store = InMemoryStore()

# 데이터 저장: put(namespace, key, value)
namespace = ("user_001", "memories")
store.put(namespace, "memory_1", {"content": "치와와를 좋아함"})
store.put(namespace, "memory_2", {"content": "비빔밥을 좋아함"})

# 전체 검색: search(namespace)
all_memories = list(store.search(namespace))

# 특정 키로 조회: get(namespace, key)
specific_memory = store.get(namespace, "memory_1")


# %% 2. 다양한 Store 예시 - 외부 Store (PostgresStore)
# 영구 보관이 필요하거나 데이터가 많아지면 외부 데이터베이스와 연동한 스토어를 씁니다(PostgreSQL + pgvector, Redis, Neo4j 등).
# 외부 스토어는 프로그램이 재시작되어도 데이터가 남고, 여러 프로세스나 서버가 같은 스토어를 공유할 수 있습니다. 이 책은 Neo4j를 씁니다.
# [설명용 코드] PostgreSQL 서버와 기본 의존성에 없는 langgraph-checkpoint-postgres 패키지가 필요해 주석으로 둡니다.
#
# import uuid
# from langgraph.store.postgres import PostgresStore
#
# DB_URI = "postgresql://postgres:postgres@localhost:5442/postgres?sslmode=disable"
#
# # PostgresStore는 context manager로 여는 것이 일반적입니다.
# with PostgresStore.from_conn_string(DB_URI) as store:
#     # 처음 한 번은 테이블/인덱스 생성이 필요합니다.
#     # store.setup()
#
#     namespace = ("user_001", "memories")
#     store.put(namespace, str(uuid.uuid4()), {"text": "치와와를 좋아함"})
#     store.put(namespace, str(uuid.uuid4()), {"text": "비빔밥을 좋아함"})
#
#     # query를 주면(벡터 검색) 의미 기반으로도 검색할 수 있습니다.
#     items = store.search(namespace, query="내가 좋아하는 음식", limit=2)


# %% 2. 다양한 Store 예시 - 벡터 Store
# get()은 키를 알아야 꺼낼 수 있어서 "배고픈데 내가 좋아하는 음식 뭐였지?" 같은 자연어 질문에는 바로 답하기 어렵습니다.
# 스토어에 임베딩 인덱스(index)를 붙이면 저장한 값을 벡터로 만들어 두고, search(..., query=...)로 의미가 가까운 항목을 찾습니다.
# embed는 임베딩 모델, dims는 벡터의 차원 수입니다. "I'm hungry"는 "I love pizza"와 의미가 가까워 그 항목이 먼저 반환됩니다.
from langchain.embeddings import init_embeddings
from langgraph.store.memory import InMemoryStore

# 임베딩 모델 준비
embeddings = init_embeddings("openai:text-embedding-3-small")

# semantic search 활성화: embed + dims 설정
store = InMemoryStore(
    index={
        "embed": embeddings,
        "dims": 1536,
    }
)

namespace = ("user_123", "memories")
store.put(namespace, "1", {"text": "I love pizza"})
store.put(namespace, "2", {"text": "I am a plumber"})

# 자연어 질의로 의미 기반 검색
items = store.search(namespace, query="I'm hungry", limit=1)
top = items[0].value  # {"text": "..."}
