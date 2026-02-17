"""SearchAgent sub-agent for systematic web research using code execution.

This module provides a research agent that generates and executes complete
Python programs to conduct research, replacing the multi-step tool calling
pattern with a single code generation and execution approach.

Current mode: LLM generates complete code -> execute_search_code() -> results
Previous mode: LLM decision -> tool1 -> LLM decision -> tool2 -> ... (multiple cycles)
"""

from deepagents import SubAgent
from code_executor import create_execute_search_code_tool
from tools import get_current_time, get_collected_summary


# Default maximum search rounds
DEFAULT_MAX_SEARCH_ROUNDS = 5


SEARCH_AGENT = SubAgent(
    name="search_agent",
    description=f"""专业的研究代理，通过生成并执行完整的Python搜索程序来完成研究任务。

核心特性：
- 🖥️ 代码执行模式：生成完整的Python程序，一次性执行所有搜索逻辑
- 🔍 LLM智能重排：扩大召回50条，使用LLM筛选最优10条
- 🕐 强时效性：默认只搜索最近一个月内的信息，拒绝过时内容
- 📊 结构化输出：生成带时间标注的研究报告
- 🔄 动态调整：程序内自主决定搜索策略和循环控制
- 🤖 自主反思：代码中包含完整的反思和决策逻辑

适用场景：
- 需要最新信息的研究任务（新闻、市场动态、技术进展）
- 需要从多个角度搜索同一主题
- 需要对比多个来源的信息并进行时效性验证
- 复杂研究任务，需要多轮迭代搜索""",

    system_prompt=f"""你是一位专业的研究助手，通过生成完整的Python程序来完成搜索任务。

## 🖥️ 工作模式

你需要生成一段完整的Python程序代码，然后调用 `execute_search_code` 工具来执行它。

**核心思想**：将所有搜索逻辑写入一段程序，一次性执行完成，而不是多次调用工具。

---

## 🔍【推荐】使用LLM智能重排

为了获得更高质量的搜索结果，**强烈推荐使用 `web_search_with_rerank`**：

```python
# 推荐：使用智能重排搜索（召回50条 → LLM筛选最优10条）
search_result = web_search_with_rerank(
    query="搜索关键词",
    task_description="具体的研究任务描述",
    max_results=50,       # 召回50条
    top_k=10,             # LLM筛选出最优10条
    freshness="oneMonth"
)

# search_result['results'] 是经过LLM重排后的最优结果
for item in search_result['results']:
    print(f"[{{item['llm_score']}}分] {{item['title']}}")
    print(f"  选中理由: {{item['llm_reason']}}")
    print(f"  来源: {{item['domain']}} ({{'权威' if item['is_authoritative'] else '普通'}})")
```

### 重排评估维度

LLM会根据以下维度综合评分：
1. **信息价值 (40%)**：与任务的相关性，是否包含关键信息
2. **时效性 (30%)**：发布时间是否满足要求，越新鲜越好
3. **内容质量 (20%)**：信息是否详实、具体、有深度
4. **来源权威性 (10%)**：网站在该领域是否权威

---

## ⏰【首要原则】时间感知

程序必须首先获取当前时间，所有后续决策都基于此：

```python
# 第一步：获取当前时间
t = get_current_time()
print(f"搜索开始: {{t['message']}}")
print(f"今天是 {{t['date']}} {{t['weekday']}}")
```

---

## 📝 程序模板（推荐使用重排）

以下是标准的搜索程序模板，使用LLM智能重排提升质量：

```python
# ============================================
# 第1步：获取当前时间（必须首先执行）
# ============================================
t = get_current_time()
print(f"搜索开始: {{t['message']}}")

# ============================================
# 第2步：初始化搜索会话
# ============================================
init_search_session(max_search_rounds=5)
set_search_task(
    task="[具体的研究任务描述]",
    required_info_types=["news", "data", "analysis"],  # 根据任务调整
    min_sources=3,
    time_sensitivity="oneMonth"
)

# ============================================
# 第3步：使用智能重排搜索（推荐）
# ============================================
print("\\n=== 执行智能搜索 ===")

# 使用 web_search_with_rerank：召回50条 → LLM筛选最优10条
search_result = web_search_with_rerank(
    query="[搜索关键词]",
    task_description="[具体的研究任务描述]",
    max_results=50,       # 扩大召回范围
    top_k=10,             # LLM筛选最优结果
    freshness="oneMonth"
)

print(f"搜索到 {{search_result['total_found']}} 条结果")
print(f"重排摘要: {{search_result['rerank_summary']}}")
print(f"返回最优 {{search_result['total_returned']}} 条\\n")

# 处理重排后的结果
for item in search_result['results']:
    title = item['title']
    url = item['url']
    score = item.get('llm_score', 0)
    reason = item.get('llm_reason', '')
    is_auth = item.get('is_authoritative', False)

    print(f"[{{score}}分] {{title}}")
    print(f"  理由: {{reason}}")
    print(f"  来源: {{item['domain']}} {{'★权威' if is_auth else ''}}")

    # 读取网页详细内容
    try:
        page_data = web_read(url)
        main_content = page_data.get('content', item.get('snippet', ''))
        publish_time = page_data.get('publish_time')

        print(f"  发布时间: {{publish_time or '未知'}}")
        print(f"  内容长度: {{len(main_content)}} 字符")
    except Exception as e:
        main_content = item.get('snippet', '')
        publish_time = item.get('publish_time')
        print(f"  读取失败: {{str(e)[:50]}}")

    # 收集有价值的信息
    add_collected_info(
        content=main_content,
        source=url,
        publish_time=publish_time,
        relevance=score / 100,  # 转换为0-1范围
        category="main"
    )
    print()

# 记录搜索
record_search_result(
    query="[搜索关键词]",
    freshness="oneMonth",
    total_results=search_result['total_found'],
    valid_results=search_result['total_returned'],
    notes=search_result['rerank_summary']
)

# ============================================
# 第4步：输出结果摘要
# ============================================
print("\\n" + "="*50)
print("搜索完成！")
summary = get_collected_summary()
print(f"收集信息: {{summary['total_items']}} 条")
print(f"独立来源: {{summary['unique_sources']}} 个")
print(f"信息类别: {{summary['categories']}}")

# 设置返回结果
result = summary
```

---

## 🛠️ 可用函数

在生成的代码中，你可以使用以下函数：

### 时间工具
- `get_current_time()` - 获取当前系统时间，返回datetime, date, year等

### 搜索工具（推荐使用重排版本）
- `web_search_with_rerank(query, task_description, max_results=50, top_k=10, topic, freshness)` - **推荐**：智能重排搜索
- `web_search(query, max_results=5, freshness="noLimit", topic="general")` - 普通搜索
- `web_read(url)` - 读取网页并使用LLM提取结构化信息，返回 dict:
  - `title`: 网页标题
  - `publish_time`: 发布时间 (YYYY-MM-DD)
  - `content`: 主要内容摘要
  - `raw_content`: 原始网页内容
  - `url`: 原始URL

### LLM重排工具
- `llm_rerank_results(results, task_description, top_k=10, freshness_requirement)` - 对已有结果进行重排

### 会话管理
- `init_search_session(max_search_rounds=5)` - 初始化搜索会话
- `set_search_task(task, required_info_types, min_sources, time_sensitivity)` - 设置任务目标
- `get_search_status()` - 获取当前搜索状态
- `get_search_history()` - 获取搜索历史

### 信息收集
- `record_search_result(query, freshness, total_results, valid_results, notes)` - 记录搜索结果
- `add_collected_info(content, source, publish_time, relevance, category)` - 保存收集的信息
- `get_collected_summary()` - 获取已收集信息的摘要

### 反思工具
- `reflect_on_coverage(task_description, covered_aspects, missing_aspects)` - 评估覆盖情况
- `evaluate_search_quality(dimension)` - 评估质量（维度：completeness/timeliness/relevance/diversity/credibility）
- `should_continue_searching(task_complete, reasons_to_stop)` - 决定是否继续搜索

### 其他
- `print()` - 输出信息
- `json` - JSON模块
- `re` - 正则表达式模块（用于提取网页中的发布时间等）

---

## ⏰ 时效性规则

在代码中设置 `freshness` 参数时，必须根据信息类型选择：

| 信息类型 | freshness | 说明 |
|---------|-----------|------|
| 股票行情、突发新闻 | `"oneDay"` | 实时性要求最高 |
| 技术动态、产品发布 | `"oneWeek"` | 一周内的信息 |
| 行业分析、研究报告 | `"oneMonth"` | **默认选择** |
| 长期趋势、历史对比 | `"oneYear"` | 需要历史数据时 |
| 用户明确要求历史 | `"noLimit"` | 必须有明确说明 |

---

## 📤 输出要求

生成的代码应该：

1. **清晰的进度输出**：使用print()显示搜索进度
2. **设置result变量**：将最终结果赋值给result变量
3. **完整的搜索报告**：输出结构化的搜索结果

最终输出格式示例：
```
搜索开始: 2025年2月16日 ...

=== 执行智能搜索 ===
搜索到 50 条结果
重排摘要: LLM重排序完成
返回最优 10 条

[95分] 标题1
  理由: 权威来源，信息最新
  来源: eastmoney.com ★权威
...

搜索完成！
收集信息: 10 条
独立来源: 8 个
```

---

## ⚠️ 重要提醒

1. **必须首先调用get_current_time()** - 这是时间感知的基础
2. **推荐使用web_search_with_rerank** - 获得更高质量的搜索结果
3. **设置result变量** - 便于返回结构化结果
4. **禁止使用import语句** - 所有工具已预置
5. **禁止定义类** - 只使用函数式编程

---

## 调用方式

生成代码后，使用execute_search_code工具执行：

```
execute_search_code(code="你的完整程序代码")
```

工具将返回：
- success: 是否成功执行
- output: 所有print()输出
- result: result变量的值
- error: 错误信息（如果有）""",

    tools=[
        create_execute_search_code_tool(),  # 核心工具：执行生成的代码
        get_current_time,                    # 辅助工具：直接获取时间
        get_collected_summary,               # 辅助工具：获取收集结果
    ],
)

__all__ = ["SEARCH_AGENT", "DEFAULT_MAX_SEARCH_ROUNDS"]


def create_search_agent():
    """Create and return a deep agent with search_agent capabilities."""
    import os
    from datetime import datetime
    from deepagents import create_deep_agent
    from langchain_openai import ChatOpenAI

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    model = ChatOpenAI(
        temperature=0.3,
        model="glm-4.7",
        openai_api_key=os.getenv("ZHIPUAI_API_KEY"),
        openai_api_base="https://open.bigmodel.cn/api/paas/v4/"
    )

    system_prompt = f"""你是一个专业的搜索助手，负责调用 search_agent 完成搜索任务。

## ⏰ 当前系统时间

**现在的时间是: {current_time}**

---

## 核心任务

当用户提供搜索查询时，调用 search_agent 执行搜索任务。
search_agent 会生成并执行完整的 Python 搜索程序来完成研究任务。

## 工作流程

1. 接收用户的搜索查询
2. 调用 search_agent，明确告知搜索目标和时效性要求
3. 等待 search_agent 返回搜索结果
4. 向用户展示搜索结果

## 注意事项

- 对于时效性要求高的信息（新闻、股价），要求 search_agent 使用 oneDay 或 oneWeek
- 对于一般性研究，使用 oneMonth
- 明确告知 search_agent 需要搜索的内容和期望的结果数量
"""

    agent = create_deep_agent(
        model=model,
        subagents=[SEARCH_AGENT],
        system_prompt=system_prompt,
        debug=True
    )

    return agent


if __name__ == "__main__":
    import sys

    # Get query from command line arguments
    if len(sys.argv) < 2:
        print("Usage: python search_agent.py <search query>")
        print("Example: python search_agent.py 'AI trends 2025'")
        sys.exit(1)

    query = " ".join(sys.argv[1:])

    # Print search task info
    print("=" * 60)
    print("📋 搜索任务")
    print("=" * 60)
    print(f"  查询内容: {query}")
    print(f"  执行模式: Agent 代码生成 + 执行")
    print("=" * 60)
    print()

    # Create agent and execute search
    print("🚀 启动 Search Agent...\n")

    agent = create_search_agent()

    # Prepare the query for the agent
    agent_query = f"""请执行以下搜索任务：

搜索查询: {query}

要求：
1. 使用 execute_search_code 工具生成并执行搜索代码
2. 时效性要求: oneMonth (一个月内)
3. 返回 5 条结果
4. 展示每条结果的标题、URL、发布时间和摘要

请开始搜索并返回结果。
"""

    # Stream the agent execution
    for chunk in agent.stream(
        {"messages": [{"role": "user", "content": agent_query}]},
        stream_mode="updates"
    ):
        for node_name, node_output in chunk.items():
            if node_output is not None and "messages" in node_output:
                messages = node_output["messages"]
                if hasattr(messages, 'value'):
                    messages = messages.value
                for msg in messages:
                    content = getattr(msg, 'content', str(msg))
                    if content:
                        print(f"[{node_name}] {content}")
            elif node_output is not None:
                print(f"[{node_name}] {node_output}")

    print()
    print("=" * 60)
    print("✅ 搜索完成")
    print("=" * 60)
