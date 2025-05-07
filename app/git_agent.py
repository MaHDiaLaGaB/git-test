import os
import subprocess
import argparse
import asyncio
from llama_index.core.workflow import (
    Workflow, step, StartEvent, StopEvent, Event
)
from llama_index.llms.ollama import Ollama
from llama_index.core.prompts import PromptTemplate


def find_repo_root() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except subprocess.CalledProcessError:
        return os.getcwd()

# Custom events for workflow steps
typing = None  # silence unused import
class DiffEvent(Event):
    diff: str

class MessageEvent(Event):
    message: str

class GitCommitWorkflow(Workflow):
    """
    Workflow to automatically commit Git changes via AI:
      1. get_diff          (StartEvent -> DiffEvent)
      2. generate_message  (DiffEvent -> MessageEvent)
      3. stage_changes     (MessageEvent -> MessageEvent)
      4. commit_changes    (MessageEvent -> StopEvent)
    """
    def __init__(self, model_name: str = "llama3.2"):
        super().__init__()
        self.repo_root = find_repo_root()
        self.llm = Ollama(model=model_name)

    @step()
    async def get_diff(self, ev: StartEvent) -> DiffEvent:
        """Step 1: Retrieve git diff."""
        output = subprocess.check_output(
            ["git", "diff", "--relative=", "."],
            cwd=self.repo_root
        ).decode().strip()
        return DiffEvent(diff=output)

    @step()
    async def generate_message(self, ev: DiffEvent) -> MessageEvent:
        """Step 2: Generate commit message using LLM."""
        diff = ev.diff
        if not diff.strip():
            return MessageEvent(message="No changes detected.")
        prompt = (
            "Generate a concise, descriptive git commit message for the following diff:\n\n" + diff
        )
        the_prompt = PromptTemplate(prompt)
        # Use predict to avoid chat-event schema validation issues
        msg = self.llm.predict(the_prompt)
        return MessageEvent(message=msg.strip())

    @step()
    async def stage_changes(self, ev: MessageEvent) -> MessageEvent:
        """Step 3: Stage all changes for commit."""
        if ev.message == "No changes detected.":
            return ev
        subprocess.check_call(["git", "add", "-A"], cwd=self.repo_root)
        return ev

    @step()
    async def commit_changes(self, ev: MessageEvent) -> StopEvent:
        """Step 4: Commit staged changes with the generated message."""
        if ev.message == "No changes detected.":
            return StopEvent(result=ev.message)
        subprocess.check_call(["git", "commit", "-m", ev.message], cwd=self.repo_root)
        return StopEvent(result=f"Committed with message: {ev.message}")

async def main(mode: str):
    workflow = GitCommitWorkflow()
    if mode == "once":
        result = await workflow.run()
        print(result)
    else:
        last_output = None
        while True:
            output = subprocess.check_output(
                ["git", "diff", "--relative=", "."],
                cwd=workflow.repo_root
            )
            if output and output != last_output:
                last_output = output
                result = await workflow.run()
                print(result)
            await asyncio.sleep(2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("AI-powered Git commit workflow")
    parser.add_argument(
        "mode", choices=["once", "watch"], nargs="?", default="once",
        help="once: commit immediately; watch: poll for changes"
    )
    args = parser.parse_args()
    asyncio.run(main(args.mode))
