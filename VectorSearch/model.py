from typing import List, Optional

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    text: str = Field(..., description="要搜索的文本")
    topk: Optional[int] = Field(5, description="返回的最相似结果的数量，默认为 5")


class SearchResult(BaseModel):
    name: str = Field(..., description="匹配到的 program 或 card_set 名称")
    similarity: float = Field(..., description="相似度分数")


class SearchResponse(BaseModel):
    results: List[SearchResult]
