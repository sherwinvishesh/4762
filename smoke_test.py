import json
import time
from pathlib import Path

from agent import ReasoningAgent

TEST_DATA = Path("cse_476_final_project_test_data.json")
N = 3 

def main() -> None:
    if not TEST_DATA.exists():
        raise FileNotFoundError(f"Missing {TEST_DATA} in current directory")

    with TEST_DATA.open("r") as fp:
        questions = json.load(fp)

    print(f"Smoke testing on first {N} of {len(questions)} test questions\n")
    agent = ReasoningAgent(max_calls=20)

    for i, item in enumerate(questions[:N], start=1):
        q = item["input"]
        start = time.time()
        try:
            ans = agent.answer(q)
        except Exception as exc:
            ans = f"ERROR: {exc}"
        elapsed = time.time() - start
        print(f"[{i}/{N}] ({elapsed:.1f}s, calls={agent.call_count})")
        print(f"  Q: {q[:140]}{'...' if len(q) > 140 else ''}")
        print(f"  A: {ans!r}\n")


if __name__ == "__main__":
    main()
