from crewai import Agent, Task
from langchain_community.llms import Ollama
import os

class ResponseAgent:
    def __init__(self, response_chain):
        self.response_chain = response_chain
        ollama_model = os.getenv("OLLAMA_MODEL", "granite3.2:latest")
        ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
        self.llm = Ollama(model=ollama_model, base_url=ollama_base_url)
        self.agent = Agent(
            role="Response Generator",
            goal="Generate personalized greetings incorporating weather data",
            backstory="You are an AI assistant that crafts friendly, context-aware messages for family members based on their identity and local weather conditions in Melbourne, Australia.",
            llm=self.llm,
            verbose=True
        )

    def generate_response(self, name, timestamp):
        task = Task(
            description=f"""
            Generate a personalized greeting for {name} based on the event timestamp {timestamp}.
            Use the provided chain to fetch weather data for Melbourne, Australia, and incorporate it into the response.
            Example: "Welcome home, {name}! It's sunny, 15°C in Melbourne."
            """,
            agent=self.agent,
            expected_output="A string containing the personalized greeting"
        )
        result = self.response_chain.run(name=name, timestamp=timestamp)
        # Remove extra quotes if present
        if isinstance(result, str):
            result = result.strip().strip('"')
        return result