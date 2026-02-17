"""Search tools for the agent.

This module provides web search functionality using multiple providers:
- Tavily Search API (internet_search) - best for English queries
- BochaAI Search API (bocha_search) - best for Chinese queries
- Search state tracking and reflection tools for iterative search refinement
"""

import os
import re
import json
from typing import Literal, Any
from tavily import TavilyClient
from dotenv import load_dotenv
import httpx
from datetime import datetime

# Load environment variables
load_dotenv()


# ============================================================================
# Time Utility - 获取当前系统时间
# ============================================================================

def get_current_time() -> dict[str, str]:
    """获取当前系统时间。

    Returns:
        包含各种时间格式的字典：
        - datetime: 完整日期时间 (YYYY-MM-DD HH:MM:SS)
        - date: 日期 (YYYY-MM-DD)
        - year: 年份
        - month: 月份
        - day: 日期
        - weekday: 星期几
        - timestamp: Unix时间戳
    """
    now = datetime.now()
    weekdays = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
    return {
        "datetime": now.strftime("%Y-%m-%d %H:%M:%S"),
        "date": now.strftime("%Y-%m-%d"),
        "year": str(now.year),
        "month": str(now.month),
        "day": str(now.day),
        "weekday": weekdays[now.weekday()],
        "timestamp": str(int(now.timestamp())),
        "message": f"当前系统时间是 {now.strftime('%Y年%m月%d日 %H:%M:%S')} {weekdays[now.weekday()]}"
    }


# ============================================================================
# Search State Tracker - 跟踪搜索状态和轮次
# ============================================================================

class SearchStateTracker:
    """Track search state, count searches, and store results for reflection."""

    def __init__(self, max_search_rounds: int = 5):
        self.max_search_rounds = max_search_rounds
        self.search_rounds: list[dict[str, Any]] = []
        self.collected_info: list[dict[str, Any]] = []
        self.search_queries: list[str] = []
        self._current_task: str = ""
        self._task_criteria: dict[str, Any] = {}

    def reset(self):
        """Reset the tracker for a new search session."""
        self.search_rounds = []
        self.collected_info = []
        self.search_queries = []
        self._current_task = ""
        self._task_criteria = {}

    def set_task(self, task: str, criteria: dict[str, Any] | None = None):
        """Set the current search task and success criteria."""
        self._current_task = task
        self._task_criteria = criteria or {}

    def record_search(
        self,
        query: str,
        freshness: str,
        results_count: int,
        valid_results_count: int = 0,
        notes: str = ""
    ) -> dict[str, Any]:
        """Record a search attempt and return current state."""
        round_num = len(self.search_rounds) + 1
        search_record = {
            "round": round_num,
            "query": query,
            "freshness": freshness,
            "results_count": results_count,
            "valid_results_count": valid_results_count,
            "notes": notes,
            "timestamp": datetime.now().isoformat()
        }
        self.search_rounds.append(search_record)
        self.search_queries.append(query)

        return self.get_status()

    def add_info(
        self,
        content: str,
        source: str,
        publish_time: str | None = None,
        relevance_score: float = 1.0,
        category: str = "general"
    ):
        """Add a piece of collected information."""
        self.collected_info.append({
            "content": content,
            "source": source,
            "publish_time": publish_time,
            "relevance_score": relevance_score,
            "category": category,
            "collected_at": datetime.now().isoformat()
        })

    def get_status(self) -> dict[str, Any]:
        """Get current search status."""
        return {
            "current_round": len(self.search_rounds),
            "max_rounds": self.max_search_rounds,
            "remaining_rounds": self.max_search_rounds - len(self.search_rounds),
            "total_searches": len(self.search_queries),
            "total_info_collected": len(self.collected_info),
            "can_continue": len(self.search_rounds) < self.max_search_rounds,
            "queries_used": self.search_queries.copy(),
            "current_task": self._current_task
        }

    def get_search_history(self) -> list[dict[str, Any]]:
        """Get the full search history."""
        return self.search_rounds.copy()

    def get_collected_info(self) -> list[dict[str, Any]]:
        """Get all collected information."""
        return self.collected_info.copy()

    def get_unique_sources(self) -> list[str]:
        """Get unique source URLs from collected info."""
        return list(set(info["source"] for info in self.collected_info if info.get("source")))


# Global search state tracker instance
_search_tracker = SearchStateTracker(max_search_rounds=5)


# ============================================================================
# State Management Tools
# ============================================================================

def init_search_session(max_search_rounds: int = 5) -> dict[str, Any]:
    """Initialize a new search session with specified maximum search rounds.

    Call this at the beginning of each research task to reset the tracker.

    Args:
        max_search_rounds: Maximum number of search rounds allowed (default 5)

    Returns:
        Status dict with session info
    """
    global _search_tracker
    _search_tracker = SearchStateTracker(max_search_rounds=max_search_rounds)
    return {
        "status": "initialized",
        "max_search_rounds": max_search_rounds,
        "message": f"搜索会话已初始化，最多允许 {max_search_rounds} 轮搜索"
    }


def set_search_task(
    task: str,
    required_info_types: list[str] | None = None,
    min_sources: int = 3,
    time_sensitivity: str = "oneMonth"
) -> dict[str, Any]:
    """Set the current search task using LLM to analyze and optimize search parameters.

    This function uses LLM to:
    1. Understand the user's search intent
    2. Determine optimal time sensitivity (freshness)
    3. Identify required information types
    4. Generate search keywords
    5. Define success criteria

    Args:
        task: User's search query or task description
        required_info_types: Optional override for info types (LLM will infer if not provided)
        min_sources: Minimum number of unique sources required (default 3)
        time_sensitivity: Optional override for time sensitivity (LLM will infer if not provided)

    Returns:
        Dict containing:
        - status: "task_set"
        - task: Original task
        - analyzed_task: LLM-analyzed task structure
        - criteria: Search criteria
        - search_suggestions: Suggested search keywords and approach
        - message: Status message
    """
    from langchain_openai import ChatOpenAI

    # Get current time for context
    current_time = get_current_time()

    # Use LLM to analyze the task
    llm = ChatOpenAI(
        temperature=0.1,
        model="glm-4-flash",
        openai_api_key=os.environ.get("ZHIPUAI_API_KEY"),
        openai_api_base="https://open.bigmodel.cn/api/paas/v4/"
    )

    analysis_prompt = f"""你是一个搜索任务分析专家。请分析以下搜索任务，并返回结构化的分析结果。

当前时间: {current_time['datetime']} ({current_time['weekday']})

搜索任务: {task}

请分析并返回以下信息（JSON格式）：

1. task_type: 任务类型（"news"新闻 / "research"研究 / "data"数据 / "analysis"分析 / "general"通用）
2. time_sensitivity: 时效性要求
   - "oneDay": 股票行情、突发新闻、实时数据
   - "oneWeek": 近期新闻、技术动态、产品发布
   - "oneMonth": 行业分析、研究报告、一般研究（默认）
   - "oneYear": 长期趋势、历史对比
   - "noLimit": 历史资料、不要求时效
3. required_info_types: 需要的信息类型列表（如 ["新闻", "数据", "分析报告", "官方公告"]）
4. search_keywords: 建议的搜索关键词列表（3-5个）
5. success_criteria: 搜索成功的标准（如 "找到至少3个来源验证的核心信息"）
6. entity: 搜索的主体对象（如公司名、产品名、人物名等）
7. topic: 搜索主题分类

请直接返回JSON：
{{"task_type": "...", "time_sensitivity": "...", "required_info_types": [...], "search_keywords": [...], "success_criteria": "...", "entity": "...", "topic": "..."}}"""

    llm_response = llm.invoke(analysis_prompt)
    llm_content = llm_response.content.strip()

    # Parse JSON response
    if llm_content.startswith("```"):
        llm_content = llm_content.split("\n", 1)[1]
        llm_content = llm_content.rsplit("```", 1)[0]

    analyzed = json.loads(llm_content)

    # Use LLM-analyzed values unless explicitly provided
    final_time_sensitivity = time_sensitivity if time_sensitivity != "oneMonth" else analyzed.get("time_sensitivity", "oneMonth")
    final_info_types = required_info_types if required_info_types else analyzed.get("required_info_types", ["general"])

    criteria = {
        "required_info_types": final_info_types,
        "min_sources": min_sources,
        "time_sensitivity": final_time_sensitivity,
        "task_type": analyzed.get("task_type", "general"),
        "entity": analyzed.get("entity"),
        "topic": analyzed.get("topic"),
        "success_criteria": analyzed.get("success_criteria")
    }

    _search_tracker.set_task(task, criteria)

    return {
        "status": "task_set",
        "task": task,
        "analyzed_task": analyzed,
        "criteria": criteria,
        "search_suggestions": {
            "keywords": analyzed.get("search_keywords", []),
            "time_sensitivity": final_time_sensitivity,
            "recommended_sources": min_sources
        },
        "message": f"搜索任务已设置: {task}"
    }



def record_search_result(
    query: str,
    freshness: str,
    total_results: int,
    valid_results: int = 0,
    notes: str = ""
) -> dict[str, Any]:
    """Record a completed search for tracking purposes.

    Call this after each web_search to track your progress.

    Args:
        query: The search query used
        freshness: The freshness parameter used
        total_results: Total number of results returned
        valid_results: Number of results that met criteria (timely, relevant)
        notes: Any observations about the search results

    Returns:
        Current search status including remaining rounds
    """
    status = _search_tracker.record_search(
        query=query,
        freshness=freshness,
        results_count=total_results,
        valid_results_count=valid_results,
        notes=notes
    )
    return status


def add_collected_info(
    content: str,
    source: str,
    publish_time: str | None = None,
    relevance: float = 1.0,
    category: str = "general"
) -> dict[str, Any]:
    """Add collected information to the knowledge base.

    Use this to store valuable information found during searches.

    Args:
        content: The information content
        source: Source URL
        publish_time: When the information was published
        relevance: Relevance score 0.0-1.0
        category: Category of information

    Returns:
        Current collection status
    """
    _search_tracker.add_info(
        content=content,
        source=source,
        publish_time=publish_time,
        relevance_score=relevance,
        category=category
    )
    return {
        "status": "added",
        "total_info_count": len(_search_tracker.collected_info),
        "unique_sources": len(_search_tracker.get_unique_sources())
    }


def get_search_status() -> dict[str, Any]:
    """Get current search status including remaining rounds.

    Returns:
        Detailed status of the current search session
    """
    return _search_tracker.get_status()


def get_search_history() -> list[dict[str, Any]]:
    """Get the complete search history for reflection.

    Returns:
        List of all search attempts with details
    """
    return _search_tracker.get_search_history()

# Initialize clients
tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
bochaai_api_key = os.environ.get("BOCHAAI_API_KEY", "")


def tavily_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
):
    """Run a web search using Tavily

    Args:
        query: Search query string
        max_results: Maximum number of results to return (default 5)
        topic: Search topic - "general", "news", or "finance" (default "general")
        include_raw_content: Whether to include raw page content (default False)

    Returns:
        Search results from Tavily API
    """
    return tavily_client.search(
        query,
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic=topic,
    )


def bocha_search(
    query: str,
    count: int = 10,
    summary: bool = True,
    freshness: Literal["noLimit", "oneDay", "oneWeek", "oneMonth", "oneYear"] = "noLimit",
    include: str | None = None,
    exclude: str | None = None,
):
    """Run a web search using BochaAI (博查搜索)

    BochaAI is optimized for Chinese content and provides AI-friendly structured summaries.

    Args:
        query: Search query string
        count: Number of results to return (1-50, default 10)
        summary: Whether to include detailed AI-generated summaries (default True)
        freshness: Time range filter - "noLimit", "oneDay", "oneWeek", "oneMonth", "oneYear"
        include: Include specific websites (e.g., "csdn.net|zhihu.com")
        exclude: Exclude specific websites (e.g., "baidu.com")

    Returns:
        JSON response with search results including web pages and optionally images.
        Response structure: {"code": 200, "data": {"webPages": {"value": [...]}, ...}}

    Raises:
        httpx.HTTPStatusError: If the API request fails
    """
    url = "https://api.bochaai.com/v1/web-search"
    headers = {
        "Authorization": f"Bearer {bochaai_api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "query": query,
        "count": min(count, 50),  # Max 50 results
        "summary": summary,
    }

    # Add optional parameters
    if freshness != "noLimit":
        payload["freshness"] = freshness
    if include:
        payload["include"] = include
    if exclude:
        payload["exclude"] = exclude

    response = httpx.post(url, headers=headers, json=payload, timeout=30.0)
    response.raise_for_status()
    return response.json()


def _is_chinese_query(query: str, threshold: float = 0.3) -> bool:
    """Detect if query is primarily in Chinese.

    Args:
        query: Search query string
        threshold: Minimum ratio of Chinese characters to consider it Chinese (default 0.3)

    Returns:
        True if query is primarily Chinese, False otherwise
    """
    # Remove whitespace and count characters
    cleaned = re.sub(r'\s+', '', query)
    if not cleaned:
        return False

    # Count Chinese characters (CJK Unified Ideographs)
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', cleaned))
    return chinese_chars / len(cleaned) >= threshold


def web_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
    summary: bool = True,
    freshness: Literal["noLimit", "oneDay", "oneWeek", "oneMonth", "oneYear"] = "noLimit",
):
    """Smart search that automatically chooses the best search engine based on query language.

    - Chinese queries → BochaAI (optimized for Chinese content)
    - English queries → Tavily (optimized for English content)

    Args:
        query: Search query string
        max_results: Maximum number of results to return (default 5)
        topic: Search topic for Tavily - "general", "news", or "finance" (default "general")
        include_raw_content: Whether to include raw page content for Tavily (default False)
        summary: Whether to include detailed AI-generated summaries for BochaAI (default True)
        freshness: Time range filter for BochaAI - "noLimit", "oneDay", "oneWeek", "oneMonth", "oneYear"

    Returns:
        Search results from the appropriate provider
    """
    is_chinese = _is_chinese_query(query)

    if is_chinese:
        # Use BochaAI for Chinese queries
        return bocha_search(
            query=query,
            count=max_results,
            summary=summary,
            freshness=freshness
        )
    else:
        # Use Tavily for English queries
        return tavily_search(
            query=query,
            max_results=max_results,
            topic=topic,
            include_raw_content=include_raw_content
        )


def web_read(url: str) -> dict[str, Any]:
    """Extract content from a webpage URL with LLM-based structured extraction.

    Uses jina.ai reader to extract clean, readable text content from webpages,
    then uses LLM to extract title, publication time, and main content.

    Args:
        url: The webpage URL to read (e.g., "https://example.com/article")

    Returns:
        Dict containing:
        - title: Extracted page title
        - publish_time: Extracted publication time (format: YYYY-MM-DD, or None)
        - content: Main content summary (~500 chars, extracted by LLM)
        - raw_content: Full raw content from jina.ai
        - url: The original URL

    Raises:
        httpx.HTTPStatusError: If the URL request fails
        ValueError: If URL is invalid or content cannot be extracted

    Example:
        >>> result = web_read("https://www.example.com/news/article-123")
        >>> print(result['title'])
        >>> print(result['publish_time'])
        >>> print(result['content'])
    """
    # Validate URL
    if not url or not isinstance(url, str):
        raise ValueError("URL must be a non-empty string")

    # Basic URL validation
    if not url.startswith(("http://", "https://")):
        raise ValueError("URL must start with http:// or https://")

    # Use jina.ai reader to extract content
    reader_url = f"https://r.jina.ai/{url}"

    response = httpx.get(reader_url, timeout=30.0, follow_redirects=True)
    response.raise_for_status()

    raw_content = response.text

    # Check if content extraction was successful
    if not raw_content or len(raw_content.strip()) < 50:
        raise ValueError(f"Failed to extract meaningful content from {url}")

    # Use LLM to extract structured information
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        temperature=0.1,
        model="glm-4-flash",
        openai_api_key=os.environ.get("ZHIPUAI_API_KEY"),
        openai_api_base="https://open.bigmodel.cn/api/paas/v4/"
    )

    # Truncate content if too long for LLM
    max_chars = 8000
    truncated_content = raw_content[:max_chars] if len(raw_content) > max_chars else raw_content

    extraction_prompt = f"""请从以下网页内容中提取关键信息，并以JSON格式返回。

网页内容：
{truncated_content}

请提取以下信息：
1. title: 网页标题
2. publish_time: 发布时间（格式：YYYY-MM-DD，如果没有找到请返回null）
3. content: 主要内容摘要（保留关键信息，控制在500字以内）

请直接返回JSON格式，不要包含```json标记：
{{"title": "...", "publish_time": "..."或null, "content": "..."}}"""

    llm_response = llm.invoke(extraction_prompt)
    llm_content = llm_response.content.strip()

    # Parse JSON response
    # Remove potential markdown code blocks
    if llm_content.startswith("```"):
        llm_content = llm_content.split("\n", 1)[1]
        llm_content = llm_content.rsplit("```", 1)[0]

    extracted = json.loads(llm_content)

    return {
        "title": extracted.get("title"),
        "publish_time": extracted.get("publish_time"),
        "content": extracted.get("content", raw_content[:2000]),
        "raw_content": raw_content,
        "url": url
    }


# ============================================================================
# Reflection and Quality Evaluation Tools
# ============================================================================

def reflect_on_coverage(
    task_description: str,
    covered_aspects: list[str],
    missing_aspects: list[str] | None = None
) -> dict[str, Any]:
    """Reflect on what has been covered and what's still missing.

    Use this tool to evaluate if the current search results adequately
    address the research task.

    Args:
        task_description: The original research task/question
        covered_aspects: List of aspects that have been well-covered
        missing_aspects: List of aspects that still need investigation

    Returns:
        Coverage analysis with recommendations for next steps
    """
    status = _search_tracker.get_status()
    collected = _search_tracker.get_collected_info()

    # Analyze coverage
    coverage_analysis = {
        "task": task_description,
        "covered": covered_aspects,
        "missing": missing_aspects or [],
        "search_rounds_used": status["current_round"],
        "search_rounds_remaining": status["remaining_rounds"],
        "total_info_collected": len(collected),
        "unique_sources": len(_search_tracker.get_unique_sources()),
        "can_continue_searching": status["can_continue"]
    }

    # Generate recommendations
    recommendations = []
    if missing_aspects and status["can_continue"]:
        for aspect in missing_aspects:
            recommendations.append(f"搜索: {aspect}")
    elif not status["can_continue"]:
        recommendations.append("已达到最大搜索轮数，基于现有信息生成报告")
    else:
        recommendations.append("任务已完成，可以生成最终报告")

    coverage_analysis["recommendations"] = recommendations

    return coverage_analysis


def evaluate_search_quality(
    dimension: Literal["completeness", "timeliness", "relevance", "diversity", "credibility"]
) -> dict[str, Any]:
    """Evaluate the quality of search results on a specific dimension.

    Args:
        dimension: The quality dimension to evaluate
            - completeness: 是否覆盖了所有需要的信息维度
            - timeliness: 信息是否足够新鲜
            - relevance: 信息是否与任务相关
            - diversity: 来源是否多样化
            - credibility: 来源是否可信

    Returns:
        Quality evaluation results with score and suggestions
    """
    collected = _search_tracker.get_collected_info()
    search_history = _search_tracker.get_search_history()
    status = _search_tracker.get_status()

    evaluation = {
        "dimension": dimension,
        "score": 0.0,
        "details": {},
        "suggestions": []
    }

    if dimension == "completeness":
        # Check if multiple aspects were covered
        categories = set(info.get("category", "general") for info in collected)
        score = min(1.0, len(categories) / 3) if categories else 0.0
        evaluation["score"] = score
        evaluation["details"]["categories_found"] = list(categories)
        if score < 0.7:
            evaluation["suggestions"].append("尝试从不同角度搜索以获得更全面的信息")

    elif dimension == "timeliness":
        # Check if information is recent
        if not collected:
            evaluation["suggestions"].append("尚未收集到信息")
        else:
            # Count info with recent publish times
            recent_count = sum(1 for info in collected if info.get("publish_time"))
            score = min(1.0, recent_count / max(len(collected), 1))
            evaluation["score"] = score
            evaluation["details"]["total_info"] = len(collected)
            evaluation["details"]["info_with_timestamp"] = recent_count
            if score < 0.5:
                evaluation["suggestions"].append("使用更短的freshness参数获取更新鲜的信息")

    elif dimension == "relevance":
        # Check average relevance score
        if not collected:
            evaluation["suggestions"].append("尚未收集到信息")
        else:
            avg_relevance = sum(info.get("relevance_score", 0) for info in collected) / len(collected)
            evaluation["score"] = avg_relevance
            evaluation["details"]["average_relevance"] = avg_relevance
            if avg_relevance < 0.7:
                evaluation["suggestions"].append("优化搜索关键词以提高相关性")

    elif dimension == "diversity":
        # Check source diversity
        sources = _search_tracker.get_unique_sources()
        score = min(1.0, len(sources) / 3) if sources else 0.0
        evaluation["score"] = score
        evaluation["details"]["unique_sources"] = len(sources)
        evaluation["details"]["total_info"] = len(collected)
        if score < 0.5:
            evaluation["suggestions"].append("尝试不同的搜索词以获得更多元的来源")

    elif dimension == "credibility":
        # Check for credible domains (basic heuristic)
        credible_domains = [".gov", ".edu", "reuters", "bloomberg", "xinhua", "people"]
        if not collected:
            evaluation["suggestions"].append("尚未收集到信息")
        else:
            credible_count = sum(
                1 for info in collected
                if any(domain in info.get("source", "").lower() for domain in credible_domains)
            )
            score = min(1.0, credible_count / max(len(collected), 1))
            evaluation["score"] = score
            evaluation["details"]["credible_sources"] = credible_count
            if score < 0.3:
                evaluation["suggestions"].append("搜索时可以指定权威来源")

    return evaluation


def should_continue_searching(
    task_complete: bool = False,
    reasons_to_stop: list[str] | None = None
) -> dict[str, Any]:
    """Determine if more searches are needed or if the task is complete.

    Call this after each search round to decide whether to continue.

    Args:
        task_complete: Whether you believe the task is complete
        reasons_to_stop: List of reasons why searching should stop

    Returns:
        Decision on whether to continue with reasoning
    """
    status = _search_tracker.get_status()

    decision = {
        "should_continue": False,
        "reason": "",
        "remaining_rounds": status["remaining_rounds"],
        "recommendations": []
    }

    # Check hard limit
    if not status["can_continue"]:
        decision["reason"] = "已达到最大搜索轮数上限"
        decision["recommendations"].append("基于现有信息生成报告")
        return decision

    # Check if task is marked complete
    if task_complete:
        decision["reason"] = "任务已标记为完成"
        decision["recommendations"].append("生成最终报告")
        return decision

    # Check for explicit reasons to stop
    if reasons_to_stop:
        decision["reason"] = "; ".join(reasons_to_stop)
        decision["recommendations"].append("考虑是否需要调整搜索策略或结束搜索")
        return decision

    # Default: continue if rounds remain
    decision["should_continue"] = True
    decision["reason"] = f"还有 {status['remaining_rounds']} 轮搜索机会"
    decision["recommendations"].append("继续搜索以获取更完整的信息")

    return decision


def get_collected_summary() -> dict[str, Any]:
    """Get a summary of all collected information.

    Returns:
        Summary of collected info grouped by category with source counts
    """
    collected = _search_tracker.get_collected_info()

    # Group by category
    by_category: dict[str, list] = {}
    for info in collected:
        cat = info.get("category", "general")
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append({
            "content_preview": info["content"][:100] + "..." if len(info["content"]) > 100 else info["content"],
            "source": info["source"],
            "publish_time": info.get("publish_time")
        })

    return {
        "total_items": len(collected),
        "unique_sources": len(_search_tracker.get_unique_sources()),
        "categories": list(by_category.keys()),
        "by_category": by_category
    }


# ============================================================================
# LLM Rerank Tool - 基于LLM的搜索结果重排序
# ============================================================================

# 权威网站域名列表（按领域分类）
AUTHORITATIVE_DOMAINS = {
    "finance": [
        "eastmoney.com", "finance.sina.com.cn", "finance.qq.com", "caixin.com",
        "yicai.com", "stcn.com", "cs.com.cn", "cnstock.com", "hexun.com",
        "cailianpress.com", "secutimes.com", "fund.eastmoney.com",
        "reuters.com", "bloomberg.com", "wsj.com", "ft.com", "cn.reuters.com",
        "sse.com.cn", "szse.cn", "csrc.gov.cn"
    ],
    "news": [
        "xinhuanet.com", "people.com.cn", "cctv.com", "china.com.cn",
        "thepaper.cn", "ifeng.com", "sohu.com", "163.com", "qq.com",
        "bjnews.com.cn", "caijing.com.cn", "infzm.com", "gmw.cn"
    ],
    "tech": [
        "36kr.com", "leiphone.com", "csdn.net", "infoq.cn", "ithome.com",
        "techcrunch.com", "wired.com", "arstechnica.com"
    ],
    "academic": [
        "cnki.net", "wanfangdata.com.cn", "nature.com", "science.org",
        "ieee.org", "acm.org", "arxiv.org"
    ]
}


def _extract_domain(url: str) -> str:
    """从URL中提取主域名"""
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        # 移除 www. 前缀
        if domain.startswith("www."):
            domain = domain[4:]
        return domain
    except:
        return ""


def _check_domain_authority(domain: str, topic: str = "general") -> tuple[bool, float]:
    """检查域名是否为权威来源

    Returns:
        (is_authoritative, authority_score)
    """
    domain_lower = domain.lower()

    # 检查所有领域的权威网站
    for category, domains in AUTHORITATIVE_DOMAINS.items():
        for auth_domain in domains:
            if auth_domain in domain_lower:
                # 根据领域相关性调整分数
                if category == topic or topic == "general":
                    return True, 1.0
                else:
                    return True, 0.8

    # .gov 和 .edu 域名
    if ".gov" in domain_lower or ".edu" in domain_lower:
        return True, 0.9

    return False, 0.5


def llm_rerank_results(
    results: list[dict[str, Any]],
    task_description: str,
    top_k: int = 10,
    freshness_requirement: str = "oneMonth"
) -> dict[str, Any]:
    """使用LLM对搜索结果进行智能重排序和筛选。

    综合考虑以下维度对搜索结果进行评估：
    1. 信息价值：内容与任务的相关性和信息量
    2. 时效性：发布时间是否满足要求
    3. 内容质量：信息是否详实、可靠
    4. 来源权威性：网站在该领域是否权威

    Args:
        results: 搜索结果列表，每个结果应包含 title, url, snippet/content 等字段
        task_description: 搜索任务描述，用于评估相关性
        top_k: 返回的最优结果数量（默认10）
        freshness_requirement: 时效性要求 ("oneDay"/"oneWeek"/"oneMonth"/"oneYear"/"noLimit")

    Returns:
        Dict containing:
        - ranked_results: 重排序后的top_k结果，包含评分和理由
        - total_input: 输入的结果总数
        - rerank_summary: 重排序决策摘要
        - message: 状态消息
    """
    from langchain_openai import ChatOpenAI

    if not results:
        return {
            "ranked_results": [],
            "total_input": 0,
            "rerank_summary": "无搜索结果需要重排序",
            "message": "输入结果为空"
        }

    # 获取当前时间
    current_time = get_current_time()

    # 标准化结果格式
    normalized_results = []
    for i, r in enumerate(results):
        title = r.get("title") or r.get("name", "无标题")
        url = r.get("url") or r.get("link", "")
        snippet = r.get("content") or r.get("snippet") or r.get("summary", "")
        publish_time = r.get("publish_time") or r.get("published_date") or r.get("datePublished")

        domain = _extract_domain(url)
        is_auth, auth_score = _check_domain_authority(domain)

        normalized_results.append({
            "index": i,
            "title": title[:200] if title else "无标题",  # 截断过长的标题
            "url": url,
            "snippet": snippet[:500] if snippet else "",  # 截断过长的摘要
            "publish_time": publish_time,
            "domain": domain,
            "is_authoritative": is_auth,
            "authority_score": auth_score
        })

    # 如果结果数量少于等于 top_k，直接返回
    if len(normalized_results) <= top_k:
        return {
            "ranked_results": normalized_results,
            "total_input": len(normalized_results),
            "rerank_summary": f"结果数量({len(normalized_results)})不超过top_k({top_k})，无需筛选",
            "message": "结果数量较少，全部保留"
        }

    # 使用 glm-4.7 进行重排序
    llm = ChatOpenAI(
        temperature=0.1,
        model="glm-4.7",
        openai_api_key=os.environ.get("ZHIPUAI_API_KEY"),
        openai_api_base="https://open.bigmodel.cn/api/paas/v4/"
    )

    # 构建结果列表供LLM评估
    results_text = ""
    for r in normalized_results:
        auth_marker = "★" if r["is_authoritative"] else ""
        results_text += f"""
[{r['index']}] {auth_marker}
标题: {r['title']}
来源: {r['domain']}
时间: {r['publish_time'] or '未知'}
摘要: {r['snippet'][:300]}
---"""

    rerank_prompt = f"""你是一个专业的信息评估专家。请对以下搜索结果进行质量评估和排序。

## 当前时间
{current_time['datetime']} ({current_time['weekday']})

## 搜索任务
{task_description}

## 时效性要求
{freshness_requirement} (oneDay=1天内, oneWeek=1周内, oneMonth=1个月内, oneYear=1年内, noLimit=不限)

## 评估标准（按重要性排序）

1. **信息价值 (40%)**：与搜索任务的相关性，是否包含关键信息
2. **时效性 (30%)**：发布时间是否满足要求，越新鲜越好
3. **内容质量 (20%)**：信息是否详实、具体、有深度
4. **来源权威性 (10%)**：网站在该领域是否权威（标记★的为权威来源）

## 搜索结果列表
{results_text}

## 任务要求

请从以上结果中选出 **最优的 {top_k} 条**，并按质量从高到低排序。

对于每条选中的结果，请提供：
1. index: 原始索引号
2. score: 综合评分 (0-100)
3. reason: 简短的选中理由（不超过20字）

请直接返回JSON格式（不要包含```json标记）：
{{
  "selected": [
    {{"index": 0, "score": 95, "reason": "权威来源，信息最新"}},
    {{"index": 3, "score": 88, "reason": "内容详实，时效性好"}},
    ...
  ],
  "summary": "筛选决策的简要说明"
}}

注意：
- 严格选择 {top_k} 条最优结果
- 评分要有区分度，不要都是相同的分数
- 优先选择权威来源和时效性好的结果
- 如果结果时效性不满足要求，应在评分中体现"""

    try:
        llm_response = llm.invoke(rerank_prompt)
        llm_content = llm_response.content.strip()

        # 清理可能的 markdown 代码块
        if llm_content.startswith("```"):
            llm_content = llm_content.split("\n", 1)[1]
            llm_content = llm_content.rsplit("```", 1)[0]

        rerank_result = json.loads(llm_content)

        # 构建排序后的结果列表
        ranked_results = []
        for item in rerank_result.get("selected", []):
            idx = item.get("index")
            if idx is not None and 0 <= idx < len(normalized_results):
                result = normalized_results[idx].copy()
                result["llm_score"] = item.get("score", 0)
                result["llm_reason"] = item.get("reason", "")
                ranked_results.append(result)

        return {
            "ranked_results": ranked_results,
            "total_input": len(normalized_results),
            "rerank_summary": rerank_result.get("summary", "LLM重排序完成"),
            "message": f"已从 {len(normalized_results)} 条结果中筛选出 {len(ranked_results)} 条最优结果"
        }

    except json.JSONDecodeError as e:
        # JSON解析失败，返回前top_k个结果（按权威性排序）
        sorted_by_auth = sorted(
            normalized_results,
            key=lambda x: x.get("authority_score", 0),
            reverse=True
        )
        return {
            "ranked_results": sorted_by_auth[:top_k],
            "total_input": len(normalized_results),
            "rerank_summary": f"LLM响应解析失败，按权威性排序返回前{top_k}条",
            "message": f"重排序降级处理：{str(e)}"
        }
    except Exception as e:
        # 其他错误，返回前top_k个结果
        return {
            "ranked_results": normalized_results[:top_k],
            "total_input": len(normalized_results),
            "rerank_summary": f"重排序失败，返回前{top_k}条原始结果",
            "message": f"重排序错误：{str(e)}"
        }


def web_search_with_rerank(
    query: str,
    task_description: str = "",
    max_results: int = 50,
    top_k: int = 10,
    topic: Literal["general", "news", "finance"] = "general",
    freshness: Literal["noLimit", "oneDay", "oneWeek", "oneMonth", "oneYear"] = "oneMonth"
) -> dict[str, Any]:
    """执行搜索并使用LLM对结果进行重排序。

    这是对 web_search + llm_rerank_results 的便捷封装：
    1. 先执行搜索，召回 max_results 条结果（默认50条）
    2. 使用LLM对结果进行智能重排序，筛选出 top_k 条最优结果

    Args:
        query: 搜索关键词
        task_description: 搜索任务描述（用于评估相关性，如为空则使用query）
        max_results: 召回的结果数量（默认50，扩大召回以提高覆盖）
        top_k: 最终返回的最优结果数量（默认10）
        topic: 搜索主题类型
        freshness: 时效性要求

    Returns:
        Dict containing:
        - query: 搜索关键词
        - results: 重排序后的top_k结果
        - total_found: 搜索到的总结果数
        - rerank_summary: 重排序摘要
    """
    # 执行搜索
    search_results = web_search(
        query=query,
        max_results=max_results,
        topic=topic,
        freshness=freshness
    )

    # 提取结果列表
    if 'results' in search_results:
        # Tavily格式
        results_list = search_results['results']
    elif 'data' in search_results and 'webPages' in search_results['data']:
        # BochaAI格式
        results_list = search_results['data']['webPages']['value']
    else:
        results_list = []

    # 使用LLM重排序
    rerank_response = llm_rerank_results(
        results=results_list,
        task_description=task_description or query,
        top_k=top_k,
        freshness_requirement=freshness
    )

    return {
        "query": query,
        "results": rerank_response["ranked_results"],
        "total_found": len(results_list),
        "total_returned": len(rerank_response["ranked_results"]),
        "rerank_summary": rerank_response["rerank_summary"],
        "freshness": freshness
    }
