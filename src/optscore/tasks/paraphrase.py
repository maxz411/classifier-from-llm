"""Paraphrased label names for held-out tasks with small label sets.

Used to test whether a trained model reads the option text (the LM-head readout) or
has merely memorized label strings: a model that reads the text should be nearly
unaffected by renaming the labels. Tasks with per-example options or very large label
sets are not paraphrased.
"""

PARAPHRASES: dict[str, dict[str, str]] = {
    "sst5": {
        "very negative": "strongly unfavorable",
        "negative": "unfavorable",
        "neutral": "mixed or indifferent",
        "positive": "favorable",
        "very positive": "strongly favorable",
    },
    "tweet_emotion": {"anger": "furious", "joy": "happiness", "optimism": "hopeful", "sadness": "sorrow"},
    "tweet_hate": {"non-hate": "not hateful", "hate": "hateful"},
    "rotten_tomatoes": {"negative": "thumbs down", "positive": "thumbs up"},
    "sms_spam": {"ham": "legitimate message", "spam": "unsolicited junk"},
    "counterfactual": {
        "not-counterfactual": "describes what actually happened",
        "counterfactual": "describes something that did not happen",
    },
    "strategyqa": {"yes": "true", "no": "false"},
}


def paraphrase_examples(task_name: str, examples):
    """Rename the options (and their listing in the prompt) of every example."""
    mapping = PARAPHRASES.get(task_name)
    if mapping is None:
        return None
    out = []
    for e in examples:
        new_options = [mapping[o] for o in e.options]
        prompt = e.prompt.replace(", ".join(e.options), ", ".join(new_options))
        for old, new in mapping.items():  # multiple-choice prompts list options one per line
            prompt = prompt.replace(f"- {old}\n", f"- {new}\n")
        out.append(type(e)(prompt, new_options, e.gold))
    return out
