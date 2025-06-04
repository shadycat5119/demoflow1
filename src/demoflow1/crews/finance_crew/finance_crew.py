from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List, Optional
from langchain_google_genai import ChatGoogleGenerativeAI

# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators


@CrewBase
class FinanceCrew:
    """Finance Crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    # If you would lik to add tools to your crew, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools

    def __init__(self, gemini_api_key: str, gemini_model: str = "gemini-2.0-flash"):
        """Initialize the Finance Crew with custom Gemini configuration.
        
        Args:
            gemini_api_key (str): The Gemini API key to use for this crew
            gemini_model (str, optional): The Gemini model to use. Defaults to "gemini-2.0-flash".
        """
        self.gemini_api_key = gemini_api_key
        self.gemini_model = gemini_model
        self.llm = ChatGoogleGenerativeAI(
            model=self.gemini_model,
            google_api_key=self.gemini_api_key
        )

    @agent
    def finance_searcher(self) -> Agent:
        return Agent(
            config=self.agents_config["finance_searcher"],  # type: ignore[index]
            llm=self.llm
        )

    @agent
    def finance_scraper(self) -> Agent:
        return Agent(
            config=self.agents_config["finance_scraper"],  # type: ignore[index]
            llm=self.llm
        )

    @agent
    def finance_compiler(self) -> Agent:
        return Agent(
            config=self.agents_config["finance_compiler"],  # type: ignore[index]
            llm=self.llm
        )

    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task
    @task
    def finance_search(self) -> Task:
        return Task(
            config=self.tasks_config["finance_search"],  # type: ignore[index]
        )

    @task
    def finance_scrape(self) -> Task:
        return Task(
            config=self.tasks_config["finance_scrape"],  # type: ignore[index]
        )

    @task
    def finance_compile(self) -> Task:
        return Task(
            config=self.tasks_config["finance_compile"],  # type: ignore[index]
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Finance Crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
        )
