from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List, Optional
from litellm import completion
from crewai_tools import SerperDevTool, ScrapeWebsiteTool
import os

# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators


@CrewBase
class EventsCrew:
    """Events Crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    def __init__(self, gemini_api_key: str, gemini_model: str = "gemini/gemini-2.0-flash"):
        """Initialize the Events Crew with custom Gemini configuration.
        
        Args:
            gemini_api_key (str): The Gemini API key to use for this crew
            gemini_model (str, optional): The Gemini model to use. Defaults to "gemini/gemini-2.0-flash".
        """
        self.gemini_api_key = gemini_api_key
        self.gemini_model = gemini_model
        os.environ["GEMINI_API_KEY"] = self.gemini_api_key

    # If you would lik to add tools to your crew, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools
    @agent
    def events_searcher(self) -> Agent:
        return Agent(
            config=self.agents_config["events_searcher"],
            llm=lambda x: completion(model=self.gemini_model, messages=[{"role": "user", "content": x}]),
            tools=[SerperDevTool()],
            verbose=True
        )

    @agent
    def events_scraper(self) -> Agent:
        return Agent(
            config=self.agents_config["events_scraper"],
            llm=lambda x: completion(model=self.gemini_model, messages=[{"role": "user", "content": x}]),
            tools=[ScrapeWebsiteTool()],
            verbose=True
        )

    @agent
    def events_compiler(self) -> Agent:
        return Agent(
            config=self.agents_config["events_compiler"],
            llm=lambda x: completion(model=self.gemini_model, messages=[{"role": "user", "content": x}]),
            verbose=True
        )

    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task
    @task
    def events_search(self) -> Task:
        return Task(
            config=self.tasks_config["events_search"],  # type: ignore[index]
        )

    @task
    def events_scrape(self) -> Task:
        return Task(
            config=self.tasks_config["events_scrape"],  # type: ignore[index]
        )

    @task
    def events_compile(self) -> Task:
        return Task(
            config=self.tasks_config["events_compile"],  # type: ignore[index]
        )
    
    @crew
    def crew(self) -> Crew:
        """Creates the Research Crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
        )
