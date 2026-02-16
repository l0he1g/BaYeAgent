"""Stock Researcher Agent - 个股研究员

This module provides a specialized agent for researching individual stocks from a
value investment perspective. It generates and executes Python code to conduct
systematic research across six dimensions.
"""

from deepagents import SubAgent
from code_executor import execute_search_code
from tools import get_current_time, get_collected_summary


# 股票研究代码模板 - 六维度分析
STOCK_RESEARCH_CODE_TEMPLATE = '''
# ============================================
# 股票研究程序 - 六维度价值分析
# 股票: {stock_name} ({stock_code})
# ============================================

# 第1步：获取当前时间（必须首先执行）
t = get_current_time()
print("=" * 60)
print(f"📊 股票研究报告")
print(f"   股票: {stock_name} ({stock_code})")
print(f"   时间: {{t['message']}}")
print("=" * 60)

# 第2步：初始化搜索会话
init_search_session(max_search_rounds=8)
set_search_task(
    task="{stock_name}({stock_code})投资价值研究",
    required_info_types=["财务数据", "公司新闻", "股价估值", "高管动态", "行业分析", "机构观点"],
    min_sources=6,
    time_sensitivity="oneMonth"
)

# 存储各维度研究结果
research_data = {{
    "stock_name": "{stock_name}",
    "stock_code": "{stock_code}",
    "research_time": t['datetime'],
    "dimensions": {{}}
}}

# ============================================
# 维度一：公司基本面分析
# ============================================
print("\\n📌 [维度1/6] 公司基本面分析")
print("-" * 40)

results_1 = web_search(
    query="{stock_name} 财务报告 营收 利润 ROE",
    max_results=5,
    freshness="oneMonth"
)

pages_1 = results_1.get('results', []) if 'results' in results_1 else results_1.get('data', {{}}).get('webPages', {{}}).get('value', [])
print(f"搜索到 {{len(pages_1)}} 条基本面信息")

fundamentals = []
for page in pages_1[:3]:
    title = page.get('title', page.get('name', 'N/A'))
    url = page.get('url', page.get('link', ''))
    snippet = page.get('content', page.get('snippet', page.get('summary', '')))
    print(f"  - {{title}}")

    try:
        content = web_read(url)
        # 提取关键财务指标
        main_content = content[:2000]
        fundamentals.append({{"title": title, "url": url, "content": main_content}})
        add_collected_info(content=main_content, source=url, relevance=0.9, category="fundamentals")
    except Exception as e:
        fundamentals.append({{"title": title, "url": url, "snippet": snippet}})
        add_collected_info(content=snippet, source=url, relevance=0.7, category="fundamentals")

record_search_result(
    query="{stock_name} 财务报告",
    freshness="oneMonth",
    total_results=len(pages_1),
    valid_results=len(fundamentals),
    notes="基本面信息收集"
)
research_data["dimensions"]["fundamentals"] = fundamentals

# ============================================
# 维度二：最新公司新闻
# ============================================
print("\\n📌 [维度2/6] 最新公司新闻")
print("-" * 40)

results_2 = web_search(
    query="{stock_name} 最新消息 新闻",
    max_results=5,
    freshness="oneWeek"
)

pages_2 = results_2.get('results', []) if 'results' in results_2 else results_2.get('data', {{}}).get('webPages', {{}}).get('value', [])
print(f"搜索到 {{len(pages_2)}} 条新闻")

news_items = []
for page in pages_2[:3]:
    title = page.get('title', page.get('name', 'N/A'))
    url = page.get('url', page.get('link', ''))
    snippet = page.get('content', page.get('snippet', page.get('summary', '')))
    print(f"  - {{title}}")

    try:
        content = web_read(url)
        main_content = content[:2000]
        news_items.append({{"title": title, "url": url, "content": main_content}})
        add_collected_info(content=main_content, source=url, relevance=0.85, category="news")
    except Exception as e:
        news_items.append({{"title": title, "url": url, "snippet": snippet}})
        add_collected_info(content=snippet, source=url, relevance=0.6, category="news")

record_search_result(
    query="{stock_name} 最新消息",
    freshness="oneWeek",
    total_results=len(pages_2),
    valid_results=len(news_items),
    notes="最新新闻收集"
)
research_data["dimensions"]["news"] = news_items

# ============================================
# 维度三：股价估值分析
# ============================================
print("\\n📌 [维度3/6] 股价估值分析")
print("-" * 40)

results_3 = web_search(
    query="{stock_name} 股价 PE 估值 PB",
    max_results=5,
    freshness="oneWeek"
)

pages_3 = results_3.get('results', []) if 'results' in results_3 else results_3.get('data', {{}}).get('webPages', {{}}).get('value', [])
print(f"搜索到 {{len(pages_3)}} 条估值信息")

valuation_data = []
for page in pages_3[:3]:
    title = page.get('title', page.get('name', 'N/A'))
    url = page.get('url', page.get('link', ''))
    snippet = page.get('content', page.get('snippet', page.get('summary', '')))
    print(f"  - {{title}}")

    try:
        content = web_read(url)
        main_content = content[:2000]
        valuation_data.append({{"title": title, "url": url, "content": main_content}})
        add_collected_info(content=main_content, source=url, relevance=0.9, category="valuation")
    except Exception as e:
        valuation_data.append({{"title": title, "url": url, "snippet": snippet}})
        add_collected_info(content=snippet, source=url, relevance=0.7, category="valuation")

record_search_result(
    query="{stock_name} 股价估值",
    freshness="oneWeek",
    total_results=len(pages_3),
    valid_results=len(valuation_data),
    notes="估值数据收集"
)
research_data["dimensions"]["valuation"] = valuation_data

# ============================================
# 维度四：高管动态
# ============================================
print("\\n📌 [维度4/6] 高管动态")
print("-" * 40)

results_4 = web_search(
    query="{stock_name} 高管变动 董事长 总经理 增持减持",
    max_results=5,
    freshness="oneMonth"
)

pages_4 = results_4.get('results', []) if 'results' in results_4 else results_4.get('data', {{}}).get('webPages', {{}}).get('value', [])
print(f"搜索到 {{len(pages_4)}} 条高管信息")

management_info = []
for page in pages_4[:3]:
    title = page.get('title', page.get('name', 'N/A'))
    url = page.get('url', page.get('link', ''))
    snippet = page.get('content', page.get('snippet', page.get('summary', '')))
    print(f"  - {{title}}")

    try:
        content = web_read(url)
        main_content = content[:2000]
        management_info.append({{"title": title, "url": url, "content": main_content}})
        add_collected_info(content=main_content, source=url, relevance=0.8, category="management")
    except Exception as e:
        management_info.append({{"title": title, "url": url, "snippet": snippet}})
        add_collected_info(content=snippet, source=url, relevance=0.6, category="management")

record_search_result(
    query="{stock_name} 高管动态",
    freshness="oneMonth",
    total_results=len(pages_4),
    valid_results=len(management_info),
    notes="高管信息收集"
)
research_data["dimensions"]["management"] = management_info

# ============================================
# 维度五：行业趋势
# ============================================
print("\\n📌 [维度5/6] 行业趋势")
print("-" * 40)

results_5 = web_search(
    query="{industry} 行业趋势 前景 景气度",
    max_results=5,
    freshness="oneMonth"
)

pages_5 = results_5.get('results', []) if 'results' in results_5 else results_5.get('data', {{}}).get('webPages', {{}}).get('value', [])
print(f"搜索到 {{len(pages_5)}} 条行业信息")

industry_info = []
for page in pages_5[:3]:
    title = page.get('title', page.get('name', 'N/A'))
    url = page.get('url', page.get('link', ''))
    snippet = page.get('content', page.get('snippet', page.get('summary', '')))
    print(f"  - {{title}}")

    try:
        content = web_read(url)
        main_content = content[:2000]
        industry_info.append({{"title": title, "url": url, "content": main_content}})
        add_collected_info(content=main_content, source=url, relevance=0.75, category="industry")
    except Exception as e:
        industry_info.append({{"title": title, "url": url, "snippet": snippet}})
        add_collected_info(content=snippet, source=url, relevance=0.5, category="industry")

record_search_result(
    query="{industry} 行业趋势",
    freshness="oneMonth",
    total_results=len(pages_5),
    valid_results=len(industry_info),
    notes="行业分析收集"
)
research_data["dimensions"]["industry"] = industry_info

# ============================================
# 维度六：机构观点
# ============================================
print("\\n📌 [维度6/6] 机构观点")
print("-" * 40)

results_6 = web_search(
    query="{stock_name} 券商研报 目标价 评级 机构调研",
    max_results=5,
    freshness="oneMonth"
)

pages_6 = results_6.get('results', []) if 'results' in results_6 else results_6.get('data', {{}}).get('webPages', {{}}).get('value', [])
print(f"搜索到 {{len(pages_6)}} 条机构观点")

analyst_views = []
for page in pages_6[:3]:
    title = page.get('title', page.get('name', 'N/A'))
    url = page.get('url', page.get('link', ''))
    snippet = page.get('content', page.get('snippet', page.get('summary', '')))
    print(f"  - {{title}}")

    try:
        content = web_read(url)
        main_content = content[:2000]
        analyst_views.append({{"title": title, "url": url, "content": main_content}})
        add_collected_info(content=main_content, source=url, relevance=0.85, category="analyst")
    except Exception as e:
        analyst_views.append({{"title": title, "url": url, "snippet": snippet}})
        add_collected_info(content=snippet, source=url, relevance=0.6, category="analyst")

record_search_result(
    query="{stock_name} 券商研报",
    freshness="oneMonth",
    total_results=len(pages_6),
    valid_results=len(analyst_views),
    notes="机构观点收集"
)
research_data["dimensions"]["analyst"] = analyst_views

# ============================================
# 生成研究摘要
# ============================================
print("\\n" + "=" * 60)
print("📊 研究完成！")
print("=" * 60)

summary = get_collected_summary()
print(f"收集信息: {{summary['total_items']}} 条")
print(f"独立来源: {{summary['unique_sources']}} 个")
print(f"信息类别: {{summary['categories']}}")

# 反思覆盖度
coverage = reflect_on_coverage(
    task_description="{stock_name}({stock_code})投资价值研究",
    covered_aspects=["基本面", "新闻", "估值", "高管", "行业", "机构观点"],
    missing_aspects=[]
)
print(f"\\n覆盖度评估: {{coverage['recommendations']}}")

# 设置返回结果
research_data["summary"] = summary
research_data["coverage"] = coverage
result = research_data

print("\\n✅ 研究数据已准备完成，可以生成投资分析报告")
'''


def research_stock(stock_name: str, stock_code: str = "", industry: str = "") -> dict:
    """执行股票研究并返回研究数据。

    Args:
        stock_name: 股票名称（如"贵州茅台"）
        stock_code: 股票代码（如"600519"），可选
        industry: 所属行业（如"白酒"），可选

    Returns:
        包含六个维度研究数据的字典
    """
    # 生成研究代码
    code = STOCK_RESEARCH_CODE_TEMPLATE.format(
        stock_name=stock_name,
        stock_code=stock_code or stock_name,
        industry=industry or f"{stock_name}所属行业"
    )

    # 执行研究代码
    execution_result = execute_search_code(code, timeout=180)

    return {
        "stock_name": stock_name,
        "stock_code": stock_code,
        "execution": execution_result,
        "research_data": execution_result.get("result"),
    }


def create_stock_research_code(stock_name: str, stock_code: str = "", industry: str = "") -> str:
    """生成股票研究代码字符串。

    Args:
        stock_name: 股票名称
        stock_code: 股票代码
        industry: 所属行业

    Returns:
        可执行的Python研究代码
    """
    return STOCK_RESEARCH_CODE_TEMPLATE.format(
        stock_name=stock_name,
        stock_code=stock_code or stock_name,
        industry=industry or f"{stock_name}所属行业"
    )


STOCK_RESEARCHER = SubAgent(
    name="stock_researcher",
    description="""专业的个股研究员代理，从价值投资角度对个股进行深入分析。

核心功能：
- 📊 公司基本面分析：财务数据、业务模式、竞争优势
- 📰 最新新闻追踪：公司动态、重大公告、市场情绪
- 📈 股价走势分析：近期表现、技术指标、成交量变化
- 👔 高管动态监控：管理层变动、股权交易、战略决策
- 🏭 行业趋势研判：行业景气度、政策影响、竞争格局
- 🏦 机构观点汇总：券商研报、目标价、机构调研

工作模式：
- 生成完整的六维度研究代码
- 通过 execute_search_code 一次性执行
- 高效完成多维度信息收集

输出结果：
- 未来三个月投资潜力评估
- 风险因素分析
- 投资建议（买入/持有/观察/回避）

适用场景：
- 用户输入个股名称（如"贵州茅台"）或代码（如"600519"）
- 需要从价值投资角度评估个股
- 需要综合多维度信息进行投资决策""",

    system_prompt="""你是一位资深的价值投资研究员，遵循巴菲特的投资理念，专注于从基本面角度分析个股的投资价值。

## ⏰【首要任务】获取当前时间

**在开始任何研究之前，先获取当前时间！**

```python
current_time = get_current_time()
print(f"当前时间: {current_time['message']}")
```

这对于股票分析至关重要：
1. 判断信息是否过时
2. 确定财报所属期间
3. 正确解读股价数据时效性

---

## 核心任务

当用户提供股票名称或代码时，你需要：
1. 确认股票基本信息（公司名称、代码、行业）
2. 生成并执行六维度研究代码
3. 从价值投资角度给出未来三个月投资评估

---

## 🖥️ 工作模式：代码执行

你通过生成并执行完整的Python代码来完成研究，而不是多次调用工具。

### 研究代码结构

```python
# 1. 获取当前时间
t = get_current_time()
print(f"研究时间: {t['message']}")

# 2. 初始化搜索会话
init_search_session(max_search_rounds=8)
set_search_task(
    task="[股票名]投资价值研究",
    required_info_types=["财务数据", "公司新闻", "股价估值", "高管动态", "行业分析", "机构观点"],
    min_sources=6
)

# 3. 六维度搜索循环
# 维度1: 基本面 - web_search(query="[股票名] 财务报告", freshness="oneMonth")
# 维度2: 新闻 - web_search(query="[股票名] 最新消息", freshness="oneWeek")
# 维度3: 估值 - web_search(query="[股票名] 股价 PE 估值", freshness="oneWeek")
# 维度4: 高管 - web_search(query="[股票名] 高管变动", freshness="oneMonth")
# 维度5: 行业 - web_search(query="[行业名] 行业趋势", freshness="oneMonth")
# 维度6: 机构 - web_search(query="[股票名] 券商研报", freshness="oneMonth")

# 4. 收集信息摘要
summary = get_collected_summary()
result = research_data
```

---

## 📋 六维度研究框架

### 维度一：公司基本面
- 搜索词：`"{股票名} 财务报告 营收 利润 ROE"`
- 时效性：oneMonth
- 重点：营收利润趋势、ROE、毛利率、现金流

### 维度二：最新公司新闻
- 搜索词：`"{股票名} 最新消息 新闻"`
- 时效性：oneWeek
- 重点：重大新闻、管理层表态、战略动向

### 维度三：股价估值
- 搜索词：`"{股票名} 股价 PE 估值 PB"`
- 时效性：oneWeek
- 重点：PE/PB估值、历史估值位置

### 维度四：高管动态
- 搜索词：`"{股票名} 高管变动 董事长 总经理"`
- 时效性：oneMonth
- 重点：管理层稳定性、增持减持

### 维度五：行业趋势
- 搜索词：`"{行业名} 行业趋势 前景"`
- 时效性：oneMonth
- 重点：行业景气度、政策变化

### 维度六：机构观点
- 搜索词：`"{股票名} 券商研报 目标价"`
- 时效性：oneMonth
- 重点：券商评级、目标价

---

## 🛠️ 可用函数

在生成的代码中，你可以使用以下函数：

### 时间工具
- `get_current_time()` - 获取当前系统时间

### 搜索工具
- `web_search(query, max_results=5, freshness="oneMonth")` - 执行网络搜索
- `web_read(url)` - 读取网页详细内容

### 会话管理
- `init_search_session(max_search_rounds=8)` - 初始化搜索会话
- `set_search_task(task, required_info_types, min_sources)` - 设置任务目标
- `get_search_status()` - 获取当前搜索状态

### 信息收集
- `record_search_result(query, freshness, total_results, valid_results, notes)` - 记录搜索
- `add_collected_info(content, source, publish_time, relevance, category)` - 保存信息
- `get_collected_summary()` - 获取收集摘要

### 反思工具
- `reflect_on_coverage(task_description, covered_aspects, missing_aspects)` - 评估覆盖度
- `should_continue_searching(task_complete)` - 决定是否继续

### 其他
- `print()` - 输出信息
- `json`, `re` - JSON和正则模块

---

## ⏰ 时效性规则

| 信息类型 | freshness | 说明 |
|---------|-----------|------|
| 股价行情、突发新闻 | `"oneDay"` | 实时性要求最高 |
| 公司新闻 | `"oneWeek"` | 一周内的信息 |
| 财务数据、行业分析 | `"oneMonth"` | **默认选择** |

---

## 输出报告格式

```markdown
# 📊 [股票名称/代码] 投资分析报告

## 基本信息
- **股票名称**：
- **股票代码**：
- **所属行业**：
- **报告日期**：

## 一、公司基本面分析
### 财务表现
[营收、利润、ROE等关键指标]

### 估值水平
[PE、PB、历史估值比较]

## 二、最新动态
[重要新闻及影响分析]

## 三、管理层分析
[高管稳定性、增持减持等]

## 四、行业分析
[行业景气度、竞争格局]

## 五、机构观点
[券商研报摘要、目标价、评级]

## 六、风险因素
1. [主要风险1]
2. [主要风险2]
3. [主要风险3]

## 七、投资潜力评估

### 投资评级
- ⭐⭐⭐⭐⭐ **强烈推荐**
- ⭐⭐⭐⭐ **建议买入**
- ⭐⭐⭐ **持有观望**
- ⭐⭐ **谨慎观察**
- ⭐ **建议回避**

### 投资建议
[具体操作建议和理由]

## 免责声明
⚠️ 本报告仅供参考，不构成投资建议。
```

---

## 调用方式

生成代码后，调用 execute_search_code 工具执行：

```
execute_search_code(code="你的完整研究代码")
```

---

## 重要原则

1. **客观中立**：基于事实和数据
2. **风险意识**：始终关注下行风险
3. **长期视角**：关注企业内在价值
4. **时效性**：明确标注信息时间
5. **来源多元**：多方验证
6. **诚实透明**：如实说明局限性""",

    tools=[
        get_current_time,
        get_collected_summary,
    ],
)


def create_stock_researcher_agent():
    """Create and return a deep agent with stock_researcher capabilities."""
    import os
    from datetime import datetime
    from deepagents import create_deep_agent
    from langchain_openai import ChatOpenAI
    from code_executor import create_execute_search_code_tool

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    model = ChatOpenAI(
        temperature=0.3,
        model="glm-4.7",
        openai_api_key=os.getenv("ZHIPUAI_API_KEY"),
        openai_api_base="https://open.bigmodel.cn/api/paas/v4/"
    )

    system_prompt = f"""你是一个专业的股票研究助手，负责调用 stock_researcher 完成股票研究任务。

## ⏰ 当前系统时间

**现在的时间是: {current_time}**

---

## 核心任务

当用户提供股票名称或代码时：
1. 调用 stock_researcher 执行六维度研究
2. stock_researcher 会生成并执行完整的研究代码
3. 基于研究结果生成投资分析报告

## 工作流程

1. 接收用户的股票查询（名称或代码）
2. 调用 stock_researcher，明确告知股票名称、代码和行业
3. 等待研究完成
4. 生成结构化的投资分析报告

## 研究维度

stock_researcher 会从六个维度进行研究：
- 基本面（财务数据、业务模式）
- 最新新闻（时效性 oneWeek）
- 股价估值（PE/PB）
- 高管动态
- 行业趋势
- 机构观点（券商研报）
"""

    agent = create_deep_agent(
        model=model,
        subagents=[STOCK_RESEARCHER],
        system_prompt=system_prompt,
        debug=True
    )

    return agent


__all__ = [
    "STOCK_RESEARCHER",
    "STOCK_RESEARCH_CODE_TEMPLATE",
    "research_stock",
    "create_stock_research_code",
    "create_stock_researcher_agent",
]
