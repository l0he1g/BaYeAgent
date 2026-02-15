from deepagents import create_deep_agent
import os
from datetime import datetime
from langchain_openai import ChatOpenAI
from search_agent import SEARCH_AGENT

current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

model = ChatOpenAI(
    temperature=0.6,
    model="glm-4.7",
    openai_api_key=os.getenv("ZHIPUAI_API_KEY"),
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/"
)

# System prompt to steer the agent to be an expert researcher
research_instructions = f"""你是价值投资专家巴菲特,擅长根据大量数据进行股票分析和预测. 请利用你的专业知识和搜索工具,帮助用户找到最有潜力上涨的A股股票.
- 你可以使用搜索子代理来获取最新的市场数据和新闻.
- 当前系统时间: {current_time}
"""

agent = create_deep_agent(
    model=model,
    subagents=[SEARCH_AGENT],
    system_prompt=research_instructions,
    debug=True
)
    
if __name__ == "__main__":
    query = "A股明天哪支股票上涨概率最大?"
    print(f"用户查询: {query}\n")

    # 使用流式输出显示执行过程
    for chunk in agent.stream(
        {"messages": [{"role": "user", "content": query}]},
        stream_mode="updates"
    ):
        # 显示每个节点的输出
        for node_name, node_output in chunk.items():
            if "messages" in node_output:
                messages = node_output["messages"]
                # 处理 Overwrite 对象
                if hasattr(messages, 'value'):
                    messages = messages.value
                for msg in messages:
                    content = getattr(msg, 'content', str(msg))
                    if content:
                        print(f"[{node_name}] {content}")
            else:
                print(f"[{node_name}] {node_output}")
