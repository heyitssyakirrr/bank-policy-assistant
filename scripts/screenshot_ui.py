"""Drive the Streamlit UI in a headless browser and save screenshots.

    python -m scripts.screenshot_ui chat "Can a customer be onboarded without meeting face to face?"
    python -m scripts.screenshot_ui compare "What does e-KYC paragraph 8.10 require?"
    python -m scripts.screenshot_ui evaluation
    python -m scripts.screenshot_ui chat --open-history     # reopen the latest conversation: no API calls

Needs the API and the UI running (uvicorn app.main:app; streamlit run ui/app.py)
and Playwright (dev only: pip install playwright; playwright install chromium).
Each question costs real API calls (3-5 Gemini calls per agentic answer).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import httpx
from playwright.sync_api import Page, sync_playwright

from app.core.config import settings

UI_URL = "http://localhost:8501"
ANSWER_TIMEOUT_MS = 180_000
PAGE_PATHS = {"chat": "", "compare": "compare", "evaluation": "evaluation"}


def ask_in_chat(page: Page, question: str) -> None:
    page.get_by_test_id("stChatInputTextArea").fill(question)
    page.keyboard.press("Enter")
    page.get_by_text("LLM calls").first.wait_for(timeout=ANSWER_TIMEOUT_MS)


def ask_in_compare(page: Page, question: str) -> None:
    page.get_by_label("Question").fill(question)
    page.get_by_role("button", name="Ask both").click()
    page.get_by_text("What each mode cost").wait_for(timeout=ANSWER_TIMEOUT_MS * 2)


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    cli.add_argument("page", choices=list(PAGE_PATHS))
    cli.add_argument("question", nargs="*")
    cli.add_argument("--out", type=Path, default=Path("docs/screenshots"))
    cli.add_argument("--open-history", action="store_true", help="chat: open the latest conversation instead")
    args = cli.parse_args()
    question = " ".join(args.question)
    args.out.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport={"width": 1500, "height": 1000})
        page.goto(f"{UI_URL}/{PAGE_PATHS[args.page]}")
        page.get_by_test_id("stApp").wait_for()
        page.wait_for_timeout(3000)  # first render, sidebar history
        if args.open_history and args.page == "chat":
            latest = httpx.get(f"{settings.api_url}/conversations").json()[0]["title"]
            page.get_by_test_id("stSidebar").get_by_role("button", name=latest).first.click()
            page.get_by_text("LLM calls").first.wait_for(timeout=ANSWER_TIMEOUT_MS)
        elif question and args.page == "chat":
            ask_in_chat(page, question)
        elif question and args.page == "compare":
            ask_in_compare(page, question)
        page.wait_for_timeout(1500)
        path = args.out / f"{args.page}.png"
        page.screenshot(path=str(path), full_page=True)
        print(f"saved {path}")
        browser.close()


if __name__ == "__main__":
    main()
