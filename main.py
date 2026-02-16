from deepagents import create_deep_agent
import os
from datetime import datetime
from langchain_openai import ChatOpenAI
from stock_researcher import STOCK_RESEARCHER

current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

model = ChatOpenAI(
    temperature=0.6,
    model="glm-4.7",
    openai_api_key=os.getenv("ZHIPUAI_API_KEY"),
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/"
)

# System prompt for stock research assistant
research_instructions = f"""你是价值投资专家巴菲特,擅长根据大量数据进行股票分析和预测。

## ⏰ 当前系统时间

**现在的时间是: {current_time}**

⚠️ **重要提醒**：
- 你必须始终以这个时间为参照点来判断信息的时效性
- 当用户问"最近"、"今天"、"本周"等时间相关问题时，以当前时间为准
- 调用子代理时，请明确告知当前时间，确保子代理使用正确的时效性判断
- 任何超过时效的信息都应标记为"可能已过时"

---

## 核心任务

当用户提供股票名称或代码时，请调用个股研究员(STOCK_RESEARCHER)进行深入研究。
个股研究员会从以下维度进行分析：
- 公司基本面（财务数据、业务模式）
- 最新新闻和公司动态
- 股价走势和估值水平
- 高管动态和股权变化
- 行业趋势和竞争格局
- 机构观点和券商研报

最终会给出未来三个月的投资潜力评估和投资建议。

## 时效性原则

在股票分析中，时效性至关重要：
1. **股价数据**：实时或当日数据优先
2. **新闻动态**：一周内的新闻为"最新"，超过一个月需标注日期
3. **财务数据**：必须标注财报所属期间
4. **研报观点**：需标注研报发布日期，超过3个月的研报价值降低

"""

agent = create_deep_agent(
    model=model,
    subagents=[STOCK_RESEARCHER],
    system_prompt=research_instructions,
    debug=True
)


def research_stock(stock_input: str):
    """Research a stock and print the analysis.

    Args:
        stock_input: Stock name (e.g., "贵州茅台") or code (e.g., "600519")
    """
    print(f"{'='*60}")
    print(f"📊 个股研究: {stock_input}")
    print(f"{'='*60}\n")

    query = f"请分析 {stock_input} 的投资价值，给出未来三个月的投资潜力评估"

    # 使用流式输出显示执行过程
    for chunk in agent.stream(
        {"messages": [{"role": "user", "content": query}]},
        stream_mode="updates"
    ):
        # 显示每个节点的输出
        for node_name, node_output in chunk.items():
            if node_output is not None and "messages" in node_output:
                messages = node_output["messages"]
                # 处理 Overwrite 对象
                if hasattr(messages, 'value'):
                    messages = messages.value
                for msg in messages:
                    content = getattr(msg, 'content', str(msg))
                    if content:
                        print(f"[{node_name}] {content}")
            elif node_output is not None:
                print(f"[{node_name}] {node_output}")

    print(f"\n{'='*60}")
    print("✅ 研究完成")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # 从命令行参数获取股票名称或代码
        stock_input = " ".join(sys.argv[1:])
        research_stock(stock_input)
    else:
        # 交互式模式
        print("📈 个股研究员 - 基于价值投资视角的个股分析")
        print("输入股票名称（如：贵州茅台）或代码（如：600519）")
        print("输入 'quit' 或 'exit' 退出\n")

        while True:
            try:
                stock_input = input("请输入股票名称或代码: ").strip()

                if stock_input.lower() in ['quit', 'exit', 'q']:
                    print("再见！")
                    break

                if not stock_input:
                    print("请输入有效的股票名称或代码")
                    continue

                research_stock(stock_input)

            except KeyboardInterrupt:
                print("\n再见！")
                break
            except Exception as e:
                print(f"发生错误: {e}")
