"""Some models narrate before answering. `result.text` is the ANSWER.

Codex-family models emit commentary parts tagged `phase="commentary"`; the
answer is tagged `final_answer` or left untagged. Concatenating every text part
gives you the narration glued to the answer, which is wrong in a way that reads
as the model being chatty rather than as a parsing bug.

`result.text` is the final answer. The narration is still available on
`result.parts` for anyone who wants to show thinking-out-loud in a UI -- kept,
not discarded, because throwing it away is also a decision the caller should
make.

Deterministic: a stub returns a two-phase response.
"""

from _check import check, report

from combycode_llm_sdk import LLM, TransportResponse

TWO_PHASE = {
    "id": "resp_1",
    "output": [
        {"type": "message", "content": [
            {"type": "output_text", "text": "Let me think about this. ", "phase": "commentary"},
            {"type": "output_text", "text": "42", "phase": "final_answer"},
        ]},
    ],
    "usage": {"input_tokens": 10, "output_tokens": 5},
}


def stub(request):
    return TransportResponse(status=200, body=TWO_PHASE)


llm = LLM(model="openai/gpt-5.4-nano", api_key="k", transport=stub)
result = llm.complete("What is the answer?")

check(result.text == "42", f"text must be the final answer alone, got {result.text!r}")

commentary = [p for p in result.parts if p.phase == "commentary"]
check(len(commentary) == 1, "the narration must still be reachable")
check("Let me think" in commentary[0].text, "commentary content is preserved verbatim")

report(text=result.text, commentary_parts=len(commentary))
