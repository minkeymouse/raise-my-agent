"""
리포지토리 점검 테스트

API 키나 네트워크 없이 몇 초 안에 끝나는 정적 점검만 합니다.
LLM을 호출하거나 예제를 실제로 실행하지는 않습니다.

    uv run pytest

점검 항목:
- ch*/ 아래의 모든 파이썬 파일이 문법 오류 없이 컴파일되는지
- 장 폴더 구성(ch00~ch10)과 예제 파일의 머리말(docstring) 규칙
- 마크다운 문서는 루트 README.md 하나뿐인지 (설명은 코드 주석에 둡니다)
- 루트 README의 상대 링크와 앵커가 실제로 존재하고, 모든 장 폴더를 안내하는지
- langgraph.json이 올바른 JSON이고, 그래프 경로("파일:객체")가 실제 정의를 가리키는지
- 실제 API 키, 토큰, 비밀번호처럼 보이는 문자열이 리포지토리에 없는지
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import urllib.parse
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", ".ruff_cache", ".pytest_cache", ".langgraph_api", "node_modules"}
CHAPTER_DIRS = sorted(p for p in ROOT.glob("ch[0-9][0-9]_*") if p.is_dir())
EXPECTED_CHAPTERS = [f"ch{n:02d}" for n in range(0, 11)]  # ch00 ~ ch10


def _skipped(path: Path) -> bool:
    return any(part in SKIP_DIRS for part in path.relative_to(ROOT).parts)


def _repo_files() -> list[Path]:
    """git이 추적하거나 추적할 수 있는 파일 목록(.gitignore 대상인 .env, .venv 등은 제외)."""
    try:
        out = subprocess.run(
            ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
            cwd=ROOT,
            capture_output=True,
            check=True,
        ).stdout
        files = [ROOT / name for name in out.decode("utf-8").split("\0") if name]
    except (OSError, subprocess.CalledProcessError):
        # git이 없으면 직접 훑되, 비밀 값이 들어 있을 수 있는 .env 계열 파일은 읽지 않습니다.
        files = [
            p
            for p in ROOT.rglob("*")
            if not (p.name.startswith(".env") and p.name != ".env.example")
        ]
    return sorted(p for p in files if p.is_file() and not _skipped(p))


REPO_FILES = _repo_files()
PY_FILES = sorted(p for d in CHAPTER_DIRS for p in d.rglob("*.py") if not _skipped(p))
MD_FILES = [p for p in REPO_FILES if p.suffix == ".md"]
LANGGRAPH_JSONS = [p for p in REPO_FILES if p.name == "langgraph.json"]


def _rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _in_package(path: Path) -> bool:
    """`chapter8/`처럼 __init__.py가 있는 패키지 안의 모듈인지."""
    return (path.parent / "__init__.py").exists()


# ── 1. 파이썬 파일 ────────────────────────────────────────────


def test_chapter_dirs_are_complete():
    found = [d.name[:4] for d in CHAPTER_DIRS]
    assert found == EXPECTED_CHAPTERS, f"장 폴더 구성이 다릅니다: {found}"


def test_only_root_readme_is_markdown():
    # 공개 리포에는 루트 README.md만 두고, 장별 설명은 예제 파일의 머리말과 주석에 둡니다.
    extra = [_rel(p) for p in MD_FILES if p != ROOT / "README.md"]
    assert not extra, f"루트 README.md 외의 마크다운 파일이 있습니다: {extra}"


@pytest.mark.parametrize("path", PY_FILES, ids=_rel)
def test_python_file_compiles(path: Path):
    source = path.read_text(encoding="utf-8")
    compile(source, str(path), "exec")


@pytest.mark.parametrize("path", PY_FILES, ids=_rel)
def test_python_file_has_header_docstring(path: Path):
    doc = ast.get_docstring(ast.parse(path.read_text(encoding="utf-8")))
    assert doc, f"{_rel(path)}에 모듈 docstring(머리말)이 없습니다."
    if _in_package(path) and path.name != "__main__.py":
        return  # 패키지 내부 모듈은 docstring만 있으면 됩니다.
    chapter_no = int(path.relative_to(ROOT).parts[0][2:4])
    assert doc.startswith(f"[{chapter_no}장 "), f"{_rel(path)}의 머리말이 '[{chapter_no}장 ...]'으로 시작하지 않습니다."
    assert "실행 방법" in doc, f"{_rel(path)}의 머리말에 '실행 방법'이 없습니다."


# ── 2. 마크다운 링크 ──────────────────────────────────────────

FENCE_RE = re.compile(r"^(\s*)(```|~~~)")
LINK_RE = re.compile(r"!?\[[^\]\n]*\]\(\s*<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\s*\)")
HTML_ATTR_RE = re.compile(r"\b(?:src|href)=\"([^\"]+)\"")
SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*:")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
HTML_ANCHOR_RE = re.compile(r"<a\s+(?:id|name)=\"([^\"]+)\"")


def _strip_code(text: str) -> list[str]:
    """펜스 코드 블록과 인라인 코드를 지운 줄 목록을 돌려줍니다."""
    lines, in_fence, fence = [], False, ""
    for line in text.splitlines():
        m = FENCE_RE.match(line)
        if m:
            if not in_fence:
                in_fence, fence = True, m.group(2)
            elif m.group(2) == fence:
                in_fence = False
            lines.append("")
            continue
        lines.append("" if in_fence else re.sub(r"`[^`]*`", "``", line))
    return lines


def _github_slug(heading: str) -> str:
    text = re.sub(r"<[^>]+>", "", heading)
    text = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = text.replace("`", "").strip().lower()
    text = re.sub(r"[^\w\- ]", "", text)
    return text.replace(" ", "-")


def _anchors(md: Path) -> set[str]:
    anchors: set[str] = set()
    counts: dict[str, int] = {}
    in_fence, fence = False, ""
    for line in md.read_text(encoding="utf-8").splitlines():
        m = FENCE_RE.match(line)
        if m:
            if not in_fence:
                in_fence, fence = True, m.group(2)
            elif m.group(2) == fence:
                in_fence = False
            continue
        if in_fence:
            continue
        anchors.update(HTML_ANCHOR_RE.findall(line))
        h = HEADING_RE.match(line)
        if h:
            slug = _github_slug(h.group(2))
            n = counts.get(slug, 0)
            counts[slug] = n + 1
            anchors.add(slug if n == 0 else f"{slug}-{n}")
    return anchors


def _links(md: Path) -> list[tuple[int, str]]:
    found = []
    for lineno, line in enumerate(_strip_code(md.read_text(encoding="utf-8")), start=1):
        for target in LINK_RE.findall(line) + HTML_ATTR_RE.findall(line):
            found.append((lineno, target))
    return found


@pytest.mark.parametrize("md", MD_FILES, ids=_rel)
def test_markdown_relative_links_resolve(md: Path):
    broken = []
    for lineno, target in _links(md):
        if SCHEME_RE.match(target):
            continue  # http(s)://, mailto: 등 외부 링크
        path_part, _, fragment = target.partition("#")
        path_part = urllib.parse.unquote(path_part.split("?")[0])
        dest = (md.parent / path_part).resolve() if path_part else md
        if not dest.exists():
            broken.append(f"{lineno}행: {target} (파일 없음)")
            continue
        if fragment and dest.is_file() and dest.suffix == ".md":
            if urllib.parse.unquote(fragment) not in _anchors(dest):
                broken.append(f"{lineno}행: {target} (앵커 없음)")
    assert not broken, f"{_rel(md)}의 깨진 링크:\n" + "\n".join(broken)


def _link_targets(md: Path) -> set[Path]:
    targets = set()
    for _, target in _links(md):
        if SCHEME_RE.match(target):
            continue
        path_part = urllib.parse.unquote(target.partition("#")[0])
        if path_part:
            targets.add((md.parent / path_part).resolve())
    return targets


def test_root_readme_links_every_chapter():
    linked = _link_targets(ROOT / "README.md")
    missing = [chapter.name + "/" for chapter in CHAPTER_DIRS if chapter.resolve() not in linked]
    assert not missing, f"루트 README.md의 장 목록에서 빠진 폴더: {missing}"


# ── 3. langgraph.json ─────────────────────────────────────────


def _top_level_names(py: Path) -> set[str]:
    names: set[str] = set()
    for node in ast.parse(py.read_text(encoding="utf-8")).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                names.update(n.id for n in ast.walk(t) if isinstance(n, ast.Name))
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update((a.asname or a.name).split(".")[0] for a in node.names)
    return names


def test_langgraph_configs_exist():
    found = sorted(_rel(p) for p in LANGGRAPH_JSONS)
    assert found == ["ch04_research_agent/langgraph.json", "ch10_intern_agent/langgraph.json"], found


@pytest.mark.parametrize("config", LANGGRAPH_JSONS, ids=_rel)
def test_langgraph_json_points_to_real_graphs(config: Path):
    data = json.loads(config.read_text(encoding="utf-8"))
    graphs = data.get("graphs")
    assert isinstance(graphs, dict) and graphs, f"{_rel(config)}에 graphs 항목이 없습니다."
    for graph_id, spec in graphs.items():
        ref = spec.get("path") if isinstance(spec, dict) else spec
        assert isinstance(ref, str) and ":" in ref, f"{graph_id}: '파일:객체' 형식이 아닙니다 ({spec!r})"
        file_part, attr = ref.rsplit(":", 1)
        py = (config.parent / file_part).resolve()
        assert py.is_file(), f"{graph_id}: {file_part} 파일이 없습니다."
        assert attr in _top_level_names(py), f"{graph_id}: {file_part}에 최상위 이름 '{attr}'이 없습니다."
    for dep in data.get("dependencies", []):
        if dep.startswith("."):
            assert (config.parent / dep).is_dir(), f"{_rel(config)}: dependencies의 {dep} 폴더가 없습니다."


# ── 4. 비밀 값 ────────────────────────────────────────────────

SECRET_PATTERNS = {
    "OpenAI 프로젝트 키": re.compile(r"sk-proj-[A-Za-z0-9_\-]{16,}"),
    "Anthropic 키": re.compile(r"sk-ant-[A-Za-z0-9_\-]{16,}"),
    "OpenAI 키": re.compile(r"(?<![\w-])sk-[A-Za-z0-9]{32,}"),
    "LangSmith 키": re.compile(r"lsv2_(?:pt|sk)_[A-Za-z0-9_]{16,}"),
    "GitHub 토큰": re.compile(r"(?<!\w)(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})"),
    "AWS 액세스 키": re.compile(r"(?<![A-Z0-9])AKIA[0-9A-Z]{16}(?![A-Z0-9])"),
    "Google API 키": re.compile(r"AIza[0-9A-Za-z_\-]{35}"),
    "Slack 토큰": re.compile(r"xox[abprs]-[A-Za-z0-9-]{10,}"),
    "개인 키": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    # GitHub Actions의 커밋 고정(@<sha>)은 제외합니다.
    "긴 16진수 토큰": re.compile(r"(?<![@\w])[0-9a-f]{32,}(?!\w)"),
}
# uv.lock에는 패키지 해시(sha256)가 들어 있으므로 검사하지 않습니다.
SECRET_SCAN_EXCLUDE = {"uv.lock"}
PASSWORD_RES = [
    re.compile(r"""(?i)\b(?:neo4j_)?password\b\s*[=:]\s*["']([^"'\n]*)["']"""),  # 따옴표로 감싼 비밀번호 대입
    re.compile(r"""(?im)^\s*(?:export\s+)?NEO4J_PASSWORD\s*=\s*([^\s"'#]+)\s*$"""),  # .env 형식의 NEO4J_PASSWORD 값
]
ENV_ASSIGN_RE = re.compile(r"^\s*([A-Z][A-Z0-9_]*)\s*=\s*(.*?)\s*$")
PLACEHOLDER_PASSWORDS = {"", "...", "YOUR_PASSWORD", "생성할_때_설정한_비밀번호"}


def _text_files() -> list[Path]:
    files = []
    for p in REPO_FILES:
        if _rel(p) in SECRET_SCAN_EXCLUDE:
            continue
        try:
            p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue  # 이미지 등 바이너리
        files.append(p)
    return files


@pytest.mark.parametrize("path", _text_files(), ids=_rel)
def test_no_secrets_in_file(path: Path):
    text = path.read_text(encoding="utf-8")
    problems = []
    for label, pattern in SECRET_PATTERNS.items():
        for m in pattern.finditer(text):
            lineno = text.count("\n", 0, m.start()) + 1
            problems.append(f"{lineno}행: {label}처럼 보이는 문자열 ({m.group(0)[:12]}...)")
    for password_re in PASSWORD_RES:
        for m in password_re.finditer(text):
            value = m.group(1)
            if value not in PLACEHOLDER_PASSWORDS and not value.startswith(("YOUR", "<", "$")):
                lineno = text.count("\n", 0, m.start()) + 1
                problems.append(f"{lineno}행: 비밀번호 값이 코드에 적혀 있습니다.")
    assert not problems, f"{_rel(path)}:\n" + "\n".join(problems)


def test_env_example_has_only_placeholders():
    example = ROOT / ".env.example"
    assert example.is_file(), ".env.example이 없습니다."
    for lineno, line in enumerate(example.read_text(encoding="utf-8").splitlines(), start=1):
        m = ENV_ASSIGN_RE.match(line)
        if not m or not re.search(r"KEY|SECRET|PASSWORD|TOKEN", m.group(1)):
            continue
        value = m.group(2).strip("\"'")
        assert value in {"", "sk-..."}, f".env.example {lineno}행: {m.group(1)}에 자리표시자가 아닌 값이 있습니다."


@pytest.mark.parametrize(
    "path",
    [
        ".env",
        ".env.local",
        "ch04_research_agent/.env",
        ".langgraph_api/store.pckl",
        "ch02_first_steps/cry.txt",
        "ch03_interaction/poo.txt",
        "ch04_research_agent/output/report.md",
        "ch08_knowledge_base/chapter8/.memo_cache/literature_memos.json",
        "ch09_mcp/03_build/notes/note.txt",
        "research_report.md",
        "ch10_intern_agent/research_report.md",
    ],
)
def test_secrets_and_generated_files_are_gitignored(path: str):
    try:
        result = subprocess.run(["git", "check-ignore", "-q", path], cwd=ROOT, capture_output=True)
    except OSError:
        pytest.skip("git을 찾을 수 없습니다.")
    if result.returncode == 128:
        pytest.skip("git 저장소가 아닙니다.")
    assert result.returncode == 0, f"{path}가 .gitignore에 포함되어 있지 않습니다."


def test_book_data_files_are_not_gitignored():
    # data/exam/(기출 문항)은 저작권 문제로 리포에 넣지 않으며 .gitignore로 제외합니다.
    for data in (ROOT / "ch08_knowledge_base" / "data").glob("*.txt"):
        try:
            result = subprocess.run(["git", "check-ignore", "-q", _rel(data)], cwd=ROOT, capture_output=True)
        except OSError:
            pytest.skip("git을 찾을 수 없습니다.")
        assert result.returncode != 0, f"{_rel(data)}가 .gitignore에 걸려 커밋되지 않습니다."


# ── 5. GitHub 설정 파일 ───────────────────────────────────────


@pytest.mark.parametrize(
    "path", [p for p in REPO_FILES if _rel(p).startswith(".github/") and p.suffix in {".yml", ".yaml"}], ids=_rel
)
def test_github_yaml_is_valid(path: Path):
    yaml = pytest.importorskip("yaml")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict) and data, f"{_rel(path)}가 비어 있거나 올바른 YAML이 아닙니다."


def test_exam_file_is_not_shipped():
    exam_dir = ROOT / "ch08_knowledge_base" / "data" / "exam"
    try:
        result = subprocess.run(["git", "check-ignore", "-q", _rel(exam_dir / "기출_날개.txt")], cwd=ROOT, capture_output=True)
    except OSError:
        pytest.skip("git을 찾을 수 없습니다.")
    if result.returncode == 128:
        pytest.skip("git 저장소가 아닙니다.")
    assert result.returncode == 0, "8장 기출 문항 파일(data/exam/)이 .gitignore에 포함되어 있지 않습니다."
