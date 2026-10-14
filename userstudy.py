"""Grading for the comprehension study.

Ported from notebook 22. The notebook exists because Google Forms under-marked
14 of the 29 participants: two answer options appear in two spellings that differ
only in the dash character, and the form's answer key matched one of them. The
mis-grading was one-directional - 14 under, 0 over - which is what shows it was a
form defect rather than selective re-grading.
"""
import re

TARGET = 0.80          # fixed in the research proposal, before recruitment

KEY = [
    "The last 100 days, which are -7.9%",
    "The market has recovered, but the system waits for the new reading to hold 10 days in a "
    "row before switching",
    "It would be buying while the market is expected to fall, and the market also looked unsettled",
    "The market is rising, is expected to keep rising, and the news agrees - nothing pulls "
    "against the trade",
    "All 5 companies are trading above their long-run average price",
    "FLAT - it expects the fall to level off",
    "The news about the company was negative",
    "The market was drifting with no clear momentum, and there was no significant news to give "
    "any reason for a price move",
    "The market is only drifting, so it gives the trade no support either way",
    "The market looked about to change, which makes the reading less reliable",
]

EXPERIENCE_COL = "How would you describe your experience with stock markets?"
NON_EXPERT = ("None at all", "A little")
SKIP = ("clear", "consent", "experience")


def normalise(s):
    """Fold the differences that are not differences of meaning.

    Dash characters, repeated whitespace and case. This is the whole reason the
    re-grading exists.
    """
    s = str(s).replace("—", "-").replace("–", "-")
    return re.sub(r"\s+", " ", s).strip().lower()


def question_columns(raw):
    """The ten graded items, found by their [Score] companion column."""
    score_cols = [c for c in raw.columns
                  if c.endswith("[Score]") and not any(k in c.lower() for k in SKIP)][:10]
    q_cols = [c[: -len(" [Score]")] for c in score_cols]
    if len(q_cols) != 10:
        raise ValueError(f"expected 10 graded items, found {len(q_cols)}")
    return q_cols, score_cols


def grade(raw):
    """Re-grade every answer and return the full result dictionary."""
    import pandas as pd

    q_cols, score_cols = question_columns(raw)

    graded = pd.DataFrame({
        i: raw[q].map(normalise).eq(normalise(KEY[i])).astype(int)
        for i, q in enumerate(q_cols)
    })

    n, n_items = len(raw), len(q_cols)
    n_answers = n * n_items
    correct = int(graded.values.sum())
    accuracy = correct / n_answers
    se = (accuracy * (1 - accuracy) / n_answers) ** 0.5
    ci = (accuracy - 1.96 * se, accuracy + 1.96 * se)

    google = raw["Total score"].str.split("/").str[0].astype(float)
    per_participant = graded.sum(axis=1)

    # experience split. Non-expert is the group the research question is about.
    mask = raw[EXPERIENCE_COL].str.startswith(NON_EXPERT)
    groups = {}
    for name, m in [("non_expert", mask), ("market_aware", ~mask)]:
        k = int(m.sum())
        ck = int(per_participant[m].sum())
        groups[name] = {"n": k, "correct": ck, "accuracy": ck / (k * n_items)}

    clarity_cols = [c for c in raw.columns
                    if c.startswith("How clear") and not c.endswith(("[Score]", "[Feedback]"))]
    clarity = raw[clarity_cols].apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")

    free_text = {}
    for name, prefix in [("confusing", "Was anything"), ("trust", "Anything else")]:
        col = next((c for c in raw.columns if c.startswith(prefix)), None)
        free_text[name] = ([str(t).strip() for t in raw[col].dropna() if str(t).strip()]
                           if col else [])

    return {
        "n_participants": n,
        "n_items": n_items,
        "n_answers": n_answers,
        "target": TARGET,
        "comprehension": {
            "correct": correct,
            "accuracy": round(accuracy, 4),
            "ci95": [round(ci[0], 4), round(ci[1], 4)],
            "target_met": bool(ci[0] >= TARGET),
        },
        "google_reported": {
            "correct": int(google.sum()),
            "accuracy": round(google.sum() / n_answers, 4),
            "participants_under_marked": int((per_participant > google).sum()),
            "participants_over_marked": int((per_participant < google).sum()),
            "cause": "Q4 and Q6 each appear in two spellings differing only in the dash "
                     "character; the form's answer key matched one of them.",
        },
        "per_item": [
            {"card": i + 1, "correct": int(graded[i].sum()), "n": n,
             "accuracy": round(float(graded[i].mean()), 4)}
            for i in range(n_items)
        ],
        "by_experience": groups,
        "clarity": {
            "overall_mean": round(float(clarity.values.mean()), 3),
            "per_card": [round(float(v), 3) for v in clarity.mean()],
        },
        "free_text": free_text,
    }
