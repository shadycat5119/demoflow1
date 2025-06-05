#!/usr/bin/env python
from random import randint
import os
from typing import Optional

from pydantic import BaseModel

from crewai.flow import Flow, listen, start, and_
from crewai import Process

from demoflow1.crews.writer_crew.writer_crew import WriterCrew
from demoflow1.crews.events_crew.events_crew import EventsCrew
from demoflow1.crews.finance_crew.finance_crew import FinanceCrew
from demoflow1.crews.people_crew.people_crew import PeopleCrew


class ContentState(BaseModel):
    finance_data: Optional[str] = None
    events_data: Optional[str] = None
    people_data: Optional[str] = None
    final_content: Optional[str] = None
    errors: list[str] = []


class ContentFlow(Flow[ContentState]):

    def __init__(self):
        super().__init__()
        # Load API keys from environment variables
        self.finance_api_key = os.getenv("FINANCE_API_KEY", "AIzaSyD_JSAYsWUsUfPV5JcO_ifV9xZ1hgG89Zw")
        self.events_api_key = os.getenv("EVENTS_API_KEY", "AIzaSyDUahLAaCSOLASINlluebkex2mJW5dPKZ4")
        self.people_api_key = os.getenv("PEOPLE_API_KEY", "AIzaSyBZtyReq9X5kZsEJRV0WqejLy4GZ9qB8FM")
        self.writer_api_key = os.getenv("WRITER_API_KEY", "AIzaSyBhb1gl9FanhHDg57Nx_9AW8WtH-oCrAQg")
        self.gemini_model = "gemini/gemini-2.0-flash"  # Correct litellm format for Gemini models

    @start()
    def gather_finance_data(self):
        try:
            print("Gathering finance data")
            result = (
                FinanceCrew(gemini_api_key=self.finance_api_key, gemini_model=self.gemini_model)
                .crew()
                .kickoff()
            )
            self.state.finance_data = result.raw
        except Exception as e:
            self.state.errors.append(f"Finance crew error: {str(e)}")
            self.state.finance_data = ""

    @start()
    def gather_events_data(self):
        try:
            print("Gathering events data")
            result = (
                EventsCrew(gemini_api_key=self.events_api_key, gemini_model=self.gemini_model)
                .crew()
                .kickoff()
            )
            self.state.events_data = result.raw
        except Exception as e:
            self.state.errors.append(f"Events crew error: {str(e)}")
            self.state.events_data = ""

    @start()
    def gather_people_data(self):
        try:
            print("Gathering people data")
            result = (
                PeopleCrew(gemini_api_key=self.people_api_key, gemini_model=self.gemini_model)
                .crew()
                .kickoff()
            )
            self.state.people_data = result.raw
        except Exception as e:
            self.state.errors.append(f"People crew error: {str(e)}")
            self.state.people_data = ""

    @listen(and_(gather_finance_data, gather_events_data, gather_people_data))
    def compile_content(self):
        try:
            print("Compiling final content")
            # Check if we have any data to compile
            if not any([self.state.finance_data, self.state.events_data, self.state.people_data]):
                raise ValueError("No data available to compile")

            result = (
                WriterCrew(gemini_api_key=self.writer_api_key)
                .crew()
                .kickoff(inputs={
                    "finance_data": self.state.finance_data or "",
                    "events_data": self.state.events_data or "",
                    "people_data": self.state.people_data or ""
                })
            )
            self.state.final_content = result.raw
        except Exception as e:
            self.state.errors.append(f"Writer crew error: {str(e)}")
            self.state.final_content = ""

    @listen(compile_content)
    def save_content(self):
        try:
            print("Saving final content")
            if self.state.final_content:
                with open("compiled_content.txt", "w") as f:
                    f.write(self.state.final_content)
            
            # Save errors if any
            if self.state.errors:
                with open("errors.log", "w") as f:
                    f.write("\n".join(self.state.errors))
        except Exception as e:
            print(f"Error saving content: {str(e)}")


def kickoff():
    content_flow = ContentFlow()
    content_flow.kickoff()


def plot():
    content_flow = ContentFlow()
    content_flow.plot()


if __name__ == "__main__":
    kickoff()
