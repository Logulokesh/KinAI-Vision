from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class ResponseChain:
    def __init__(self):
        ollama_model = os.getenv("OLLAMA_MODEL", "granite3.2:latest")
        ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
        self.llm = Ollama(model=ollama_model, base_url=ollama_base_url)
        self.weather_api_key = os.getenv("OPENWEATHERMAP_API_KEY", "b084152b7d66a6b21ecd1850259fdae4")
        self.prompt = PromptTemplate(
            input_variables=["name", "weather", "temperature"],
            template="""
            Generate a concise greeting for {name} in Melbourne, Australia, incorporating the current weather.
            Weather: {weather}, Temperature: {temperature}°C.
            The greeting must follow this exact format: "Welcome home, {name}! It's {weather}, {temperature}°C in Melbourne."
            Do not add extra text or deviate from the format.
            """
        )
        self.chain = RunnableSequence(self.prompt | self.llm)

    def get_weather(self):
        # Fetch weather for Melbourne, AU (lat: -37.8136, lon: 144.9631)
        url = f"http://api.openweathermap.org/data/2.5/weather?lat=-37.8136&lon=144.9631&appid={self.weather_api_key}&units=metric"
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            weather = data["weather"][0]["description"]
            temperature = data["main"]["temp"]
            return weather, temperature
        except Exception as e:
            return "unknown", 15  # Fallback values for Melbourne

    def run(self, name, timestamp):
        weather, temperature = self.get_weather()
        return self.chain.invoke({"name": name, "weather": weather, "temperature": temperature})