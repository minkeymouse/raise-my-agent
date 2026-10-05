"""
[8장 3절] 에이전트 시험 공부 하기 - 구조화 출력 스키마

8장·2절에서 다룬 구조화 추출 스키마(LiteratureExtraction 계열)와,
3절의 시험공부 에이전트가 키워드 추출과 답안 작성에 쓰는 ExamKeywords, ExamSolution입니다.
"""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class Character(BaseModel):
    name: str = Field(description="등장인물 이름(또는 지칭)")
    role: Optional[str] = Field(default=None, description="서술자/주인공/조연 등 역할")
    traits: List[str] = Field(default_factory=list, description="성격/상태/특징 키워드")


class CharacterRelation(BaseModel):
    source: str = Field(description="관계의 주체 인물 이름")
    target: str = Field(description="관계의 대상 인물 이름")
    relation: Literal[
        "conflict",
        "affection",
        "dependence",
        "control",
        "family",
        "friendship",
        "unknown",
    ] = Field(description="관계 유형(필요시 unknown)")
    evidence_quote: Optional[str] = Field(
        default=None, description="관계를 뒷받침하는 짧은 근거 문장(가능하면 원문)"
    )


class Event(BaseModel):
    event_id: str = Field(description="사건 식별자(짧은 문자열)")
    description: str = Field(description="사건 요약(무슨 일이 일어났는지)")
    participants: List[str] = Field(description="사건에 관여한 인물 이름들")
    impact: List[str] = Field(
        default_factory=list, description="사건이 만든 변화/결과(감정/행동/관계 변화)"
    )


class Interpretation(BaseModel):
    subject: str = Field(description="행동/생각/상징 등 해석 대상(짧게)")
    meaning: str = Field(description="그 의미(주제/인물 상태/서술 효과와 연결)")
    evidence_quote: Optional[str] = Field(
        default=None, description="근거 문장(가능하면 원문)"
    )


class LiteratureExtraction(BaseModel):
    """문학 작품에서 '관계'까지 포함해 추출할 구조화된 정보."""

    work_title: str = Field(description="작품 제목")
    author: str = Field(description="작가 이름")
    summary: str = Field(description="작품(또는 발췌)의 간단 요약")
    themes: List[str] = Field(default_factory=list, description="주제 키워드")
    characters: List[Character] = Field(default_factory=list, description="등장인물 목록")
    relations: List[CharacterRelation] = Field(
        default_factory=list, description="등장인물 간 관계"
    )
    events: List[Event] = Field(default_factory=list, description="사건(인물-사건 연결 중심)")
    interpretations: List[Interpretation] = Field(
        default_factory=list, description="행동/생각/상징의 의미 해석(근거 포함)"
    )
    key_quotes: List[str] = Field(default_factory=list, description="핵심 근거 문장 1~5개")


# 3절에서 더한 스키마: 문항에서 검색 키워드를 뽑고(ExamKeywords), 선지 번호와 해설을 받습니다(ExamSolution).
class ExamKeywords(BaseModel):
    """시험 지문에서 검색·추론에 쓸 키워드."""

    keywords: List[str] = Field(description="짧은 키워드 또는 구절")
    focus: str = Field(description="문제가 묻는 초점(예: 서술, 상징, 인물관계)")


class ExamSolution(BaseModel):
    """객관식 답안(번호)과 서술형 근거."""

    answer_number: int = Field(description="선지 번호 1~5 (①=1 … ⑤=5)")
    reasoning: str = Field(description="근거를 들어 한국어로 간결히 설명")
