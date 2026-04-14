"""
2-Agent Research Assistant — Hackathon Starter (Groq version)
==============================================================
Agents:  Researcher  →  Writer
Model:   Llama 3 8B via Groq (free API, no GPU needed)
Requires:
    pip install crewai crewai-tools duckduckgo-search streamlit langchain-groq ddgs

Usage:
    set GROQ_API_KEY=your_key_here
    python research_crew.py
"""

import os
import sys

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    print("\nERROR: GROQ_API_KEY is not set.")
    print("Run this first:  set GROQ_API_KEY=your_key_here\n")
    sys.exit(1)

os.environ["GROQ_API_KEY"] = GROQ_API_KEY

from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from ddgs import DDGS
from pydantic import Field

# ── 1. LLM — passed as a string (CrewAI new-style) ───────────────────────────
LLM = "groq/llama-3.3-70b-versatile"

# ── 2. Search tool as a proper BaseTool subclass ─────────────────────────────

class WebSearchTool(BaseTool):
    name: str = "Web Search"
    description: str = "Search the web for information about any topic and return results with sources."

    def _run(self, query: str) -> str:
        try:
            results = []
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=5):
                    results.append(f"- {r['title']}: {r['body']} (Source: {r['href']})")
            return "\n".join(results) if results else "No results found."
        except Exception as e:
            return f"Search failed: {str(e)}"

search_tool = WebSearchTool()

# ── 3. Agents ─────────────────────────────────────────────────────────────────

researcher = Agent(
    role="Senior Research Analyst",
    goal=(
        "Find accurate, well-sourced information about the given topic. "
        "Always include URLs for your sources."
    ),
    backstory=(
        "You are a meticulous research analyst who never guesses. "
        "You search multiple sources, cross-reference facts, and always "
        "cite where you found information. You flag anything uncertain."
    ),
    tools=[search_tool],
    llm=LLM,
    verbose=True,
    max_iter=5,
    allow_delegation=False,
)

writer = Agent(
    role="Research Report Writer",
    goal=(
        "Turn raw research findings into a clear, well-structured markdown report "
        "that a non-expert can understand."
    ),
    backstory=(
        "You are a skilled technical writer who specialises in making complex "
        "research accessible. You structure reports with an executive summary, "
        "key findings, and a sources section."
    ),
    tools=[],
    llm=LLM,
    verbose=True,
    max_iter=3,
    allow_delegation=False,
)

# ── 4. Tasks ──────────────────────────────────────────────────────────────────

def build_crew(query: str) -> Crew:
    research_task = Task(
        description=(
            f"Research the following topic thoroughly:\n\n{query}\n\n"
            "Search for at least 3 different sources. "
            "Return a structured list of findings with source URLs."
        ),
        expected_output=(
            "A bullet-point list of key facts and findings, each with a source URL. "
            "Flag any conflicting information you found."
        ),
        agent=researcher,
    )

    write_task = Task(
        description=(
            "Using the research findings provided, write a clear markdown report. "
            "Structure it as:\n"
            "1. Executive summary (2-3 sentences)\n"
            "2. Key findings (numbered list)\n"
            "3. Important caveats or conflicting information\n"
            "4. Sources\n\n"
            "Keep the tone neutral and factual."
        ),
        expected_output=(
            "A complete markdown report with all four sections. "
            "The executive summary must be 2-3 sentences max."
        ),
        agent=writer,
        context=[research_task],
    )

    return Crew(
        agents=[researcher, writer],
        tasks=[research_task, write_task],
        process=Process.sequential,
        verbose=True,
    )

# ── 5. Run modes ──────────────────────────────────────────────────────────────

def run_cli():
    query = input("\nEnter your research question:\n> ").strip()
    if not query:
        print("No query entered. Exiting.")
        sys.exit(1)

    print(f"\nStarting research crew for: '{query}'\n{'─'*60}\n")
    crew = build_crew(query)
    result = crew.kickoff()

    print("\n" + "─"*60)
    print("FINAL REPORT")
    print("─"*60)
    print(result)


def run_streamlit():
    import streamlit as st

    st.set_page_config(page_title="AI Research Assistant", layout="wide")
    st.title("AI Research Assistant")
    st.caption("Powered by CrewAI + Llama 3 on Groq")

    query = st.text_input(
        "What do you want researched?",
        placeholder="e.g. What is the current state of quantum computing?"
    )

    if st.button("Run research", type="primary") and query:
        with st.spinner("Agents are working... this takes 1-3 minutes"):
            crew = build_crew(query)
            result = crew.kickoff()

        st.markdown("## Report")
        st.markdown(str(result))

        st.download_button(
            label="Download report",
            data=str(result),
            file_name="research_report.md",
            mime="text/markdown",
        )


# ── 6. Entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if "streamlit" in sys.modules:
        run_streamlit()
    else:
        try:
            import streamlit as st
            run_streamlit()
        except Exception:
            run_cli()
