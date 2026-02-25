import gradio as gr
from dotenv import load_dotenv

from assignment_chat.main import assignment_chat
from utils.logger import get_logger

_logs = get_logger(__name__)

load_dotenv(".env")
load_dotenv(".secrets")

chat = gr.ChatInterface(
    fn=assignment_chat,
    type="messages",
    title="North Star Guide",
    description="Weather + semantic AI knowledge + weekly study planning with guardrails.",
)

if __name__ == "__main__":
    _logs.info("Starting Assignment Chat App...")
    chat.launch()

