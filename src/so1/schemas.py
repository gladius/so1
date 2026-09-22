"""Wire schema, mirroring https://api.typesafe.ai/openapi.json (TypeSafe 0.2.0).

Field names, types and optionality are taken from the real spec so the official
typesafe-sdk validates our responses unchanged. Per-question option/level ceilings
are ours and come from config, so they are enforced in the service, not here.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, TypeAlias

from pydantic import BaseModel, Field

JSONContent: TypeAlias = str | dict[str, Any] | list[Any]


# --------------------------------------------------------------------------------------
# Requests
# --------------------------------------------------------------------------------------


class NoulCriteria(BaseModel):
    true_: JSONContent | None = Field(default=None, alias="true")
    false_: JSONContent | None = Field(default=None, alias="false")

    model_config = {"populate_by_name": True}


class NoulQuestion(BaseModel):
    type: Literal["noul"]
    instructions: JSONContent | None = None
    criteria: NoulCriteria | None = None


class ChoiceQuestion(BaseModel):
    type: Literal["choice"]
    instructions: JSONContent | None = None
    criteria: dict[str, JSONContent | None]


class ScoreQuestion(BaseModel):
    type: Literal["score"]
    instructions: JSONContent | None = None
    criteria: list[JSONContent] = Field(min_length=1)


Question: TypeAlias = Annotated[NoulQuestion | ChoiceQuestion | ScoreQuestion, Field(discriminator="type")]


class SystemOneRequest(BaseModel):
    state: JSONContent
    model: str
    questions: dict[str, Question] = Field(min_length=1)


# --------------------------------------------------------------------------------------
# Responses
# --------------------------------------------------------------------------------------


class NoulAnswer(BaseModel):
    type: Literal["noul"]
    noul: float


class ChoiceAnswer(BaseModel):
    type: Literal["choice"]
    choice: str
    confidence: float
    probabilities: dict[str, float]


class ScoreAnswer(BaseModel):
    type: Literal["score"]
    score: float
    confidence: float
    legend: dict[str, JSONContent]
    probabilities: dict[str, float]


Answer: TypeAlias = NoulAnswer | ChoiceAnswer | ScoreAnswer


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int


class SystemOneResponse(BaseModel):
    model: str
    answers: dict[str, Answer] = Field(min_length=1)
    usage: Usage


class ModelMetadata(BaseModel):
    name: str
    description: str
    release_date: str


class ModelMetadataList(BaseModel):
    models: list[ModelMetadata]
