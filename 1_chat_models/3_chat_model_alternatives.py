from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

# Setup environment variables and messages
load_dotenv()

messages = [
    SystemMessage(content="Solve the following math problems"),
    HumanMessage(content="What is 81 divided by 9?"),
]


# ---- LangChain OpenAI Chat Model Example ----

# Create a ChatOpenAI model
model = ChatOpenAI(model="gpt-4o")

# Invoke the model with messages
result = model.invoke(messages)
print(f"Answer from OpenAI: {result.content}")


# ---- Anthropic Chat Model Example ----

# Create a Anthropic model
# Anthropic models: https://docs.anthropic.com/en/docs/models-overview
model = ChatAnthropic(
                model="claude-3-sonnet-20240229",
                temperature=0,
                max_tokens=1024,
                timeout=None,
                max_retries=2
            )

result = model.invoke(messages)
print(f"Answer from Anthropic: {result.content}")


# ---- Google Chat Model Example ----

# https://console.cloud.google.com/gen-app-builder/engines
# https://ai.google.dev/gemini-api/docs/models/gemini
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

result = model.invoke(messages)
print(f"Answer from Google: {result.content}")
