import os

import numpy as np
import pytest
import torch

from packreadout import choice, load_model, score, score_examples, score_options, score_options_naive, to_probs, true_false
from packreadout.bench.cached_baseline import score_options_cached
from packreadout.formatting import DEFAULT_FORMAT, LM_EVAL_FORMAT, OptionFormat
from packreadout.normalize import fit_temperature, log_softmax

MODEL = os.environ.get("OPTSCORE_TEST_MODEL", "Qwen/Qwen3-0.6B-Base")  # e.g. openai/gpt-oss-20b on a big GPU


@pytest.fixture(scope="module")
def lm():
    return load_model(MODEL, dtype=torch.float32)


PROMPT = "Question: What color is a clear daytime sky?\nAnswer:"
OPTIONS = ["blue", "green", "a very dark shade of purple", "New York", "New York City"]
MANY = [
    "card arrival",
    "lost or stolen card",
    "pin blocked",
    "top up failed",
    "exchange rate",
    "why verify identity",
    "a",
    "a very long option that goes on for quite a few tokens indeed",
]


def assert_same(a, b):
    np.testing.assert_allclose(a.sum_logprob, b.sum_logprob, atol=2e-3, rtol=1e-4)
    for x, y in zip(a.token_logprobs, b.token_logprobs):
        np.testing.assert_allclose(x, y, atol=2e-3, rtol=1e-4)


def test_packed_matches_naive(lm):
    model, tok = lm
    for prompt, options in [(PROMPT, OPTIONS), ("Classify the query.\nText: my card never came\nLabel:", MANY)]:
        assert_same(score_options(model, tok, prompt, options), score_options_naive(model, tok, prompt, options))
    assert score_options(model, tok, PROMPT, OPTIONS).argmax() == 0


def test_bulk_matches_single(lm):
    """Batched packed scoring equals scoring each example alone, packed or naive."""
    model, tok = lm
    prompts = [PROMPT, "Question: What is 2+2?\nAnswer:", "Classify.\nText: hi\nLabel:"]
    sets = [OPTIONS, ["4", "5", "twenty two"], MANY]
    packed = score_examples(model, tok, prompts, sets, batch_size=2)
    for prompt, options, a in zip(prompts, sets, packed):
        assert_same(a, score_options(model, tok, prompt, options))
        assert_same(a, score_options_naive(model, tok, prompt, options))
    assert packed[1].argmax() == 0


def test_scalar_readout_is_the_same_in_training_and_evaluation(lm):
    """A scalar head scored through score_examples gives the logits option_logits trains on."""
    from packreadout.tasks.base import Example
    from packreadout.train import option_logits

    model, tok = lm
    torch.manual_seed(0)
    head = torch.nn.Linear(model.config.hidden_size, 1, bias=False, device=model.device)
    examples = [Example(PROMPT, OPTIONS, 0), Example("Classify.\nText: hi\nLabel:", MANY, 2)]
    trained = option_logits(model, tok, examples, "scalar", head)
    evaluated = score_examples(
        model, tok, [e.prompt for e in examples], [e.options for e in examples], readout="scalar", head=head
    )
    for a, b in zip(trained, evaluated):
        np.testing.assert_allclose(a.detach().cpu().numpy(), b.sum_logprob, atol=1e-4)
        assert all(len(lp) == 1 for lp in b.token_logprobs)  # one "token" per option: its logit


def test_gptoss_implementation_matches_main_on_a_dense_model(lm):
    """packreadout.gptoss is a complete copy for gpt-oss; on a plain decoder it must agree with packreadout."""
    from packreadout.gptoss.scoring import score_options as score_options_gptoss

    model, tok = lm
    model.config._attn_implementation = "eager"  # the gpt-oss implementation loads models with eager attention
    try:
        assert_same(score_options_gptoss(model, tok, PROMPT, OPTIONS), score_options_naive(model, tok, PROMPT, OPTIONS))
    finally:
        model.config._attn_implementation = "sdpa"


def test_terminator_separates_prefix_options(lm):
    model, tok = lm
    prompt = "Question: Which US city is the largest?\nAnswer:"
    opts = ["New York", "New York City"]
    without = score_options(model, tok, prompt, opts, fmt=LM_EVAL_FORMAT)
    assert without.sum_logprob[0] >= without.sum_logprob[1]  # a prefix can never lose without a terminator
    with_term = score_options(model, tok, prompt, opts, fmt=DEFAULT_FORMAT)
    assert with_term.n_tokens[0] == without.n_tokens[0] + 1
    assert not np.isclose(np.diff(with_term.sum_logprob), np.diff(without.sum_logprob))


def test_shared_prefix_cancels(lm):
    model, tok = lm
    full = score_options(model, tok, "Question: Which state borders Pennsylvania?\nAnswer:", ["New York", "New Jersey"])
    np.testing.assert_allclose(full.token_logprobs[0][0], full.token_logprobs[1][0], atol=1e-5)
    suffix_scores = np.array([lp[1:].sum() for lp in full.token_logprobs])
    np.testing.assert_allclose(full.probs("raw"), to_probs(suffix_scores), atol=1e-6)


def test_uncond_pmi(lm):
    model, tok = lm
    r = score_options(model, tok, PROMPT, OPTIONS, uncond=True)
    assert r.uncond_logprob.shape == (len(OPTIONS),)
    np.testing.assert_allclose(r.scores("pmi"), r.sum_logprob - r.uncond_logprob)
    np.testing.assert_allclose(r.scores("char"), r.sum_logprob / [len(o) for o in OPTIONS])


def test_primitives(lm):
    model, tok = lm
    ticket = "Hi, I was charged twice for my subscription this month. Please refund the duplicate charge."
    probs = choice(model, tok, ticket, "Which team should handle this?", ["billing", "technical", "sales"])
    assert abs(sum(probs.values()) - 1) < 1e-6 and max(probs, key=probs.get) == "billing"
    expected, dist = score(
        model,
        tok,
        ticket,
        "How frustrated is the customer?",
        ["calm, just stating facts", "frustrated but civil", "very angry, strong language"],
    )
    assert 0 <= expected <= 2 and abs(sum(dist) - 1) < 1e-6
    p_yes = true_false(model, tok, ticket, "Does the customer request a refund?")
    p_no = true_false(model, tok, ticket, "Does the customer ask about pricing plans?")
    assert 0 <= p_no <= p_yes <= 1


def test_normalizations_are_consistent():
    s = np.array([-2.0, -8.0, -6.0])
    p = to_probs(s)
    assert np.isclose(p.sum(), 1.0) and np.argmax(p) == 0
    assert np.allclose(np.exp(log_softmax(s)), p)
    assert 0.05 <= fit_temperature([s, s * 3], [0, 0]) <= 20


def test_option_format_renders():
    assert OptionFormat().render("cat") == " cat\n"
    assert OptionFormat(terminator="").render(" cat") == " cat"


def test_training_objective_moves_probability_to_gold(lm):
    """A few LoRA steps on two examples must raise the gold option's probability."""
    from packreadout.tasks.base import Example
    from packreadout.train import add_lora, loss_fn, option_logits

    model, tok = lm
    examples = [
        Example(
            "Classify the review.\nLabels: negative, positive\nText: what a waste of time\nLabel:",
            ["negative", "positive"],
            0,
        ),
        Example(
            "Classify the review.\nLabels: negative, positive\nText: loved every minute\nLabel:",
            ["negative", "positive"],
            1,
        ),
    ]
    golds = [e.gold for e in examples]
    peft_model = add_lora(model, rank=4)
    opt = torch.optim.AdamW([p for p in peft_model.parameters() if p.requires_grad], lr=1e-3)
    before = loss_fn(option_logits(peft_model, tok, examples, "lm_head"), golds, "set", {}).item()
    for _ in range(5):
        loss = loss_fn(option_logits(peft_model, tok, examples, "lm_head"), golds, "set", {})
        opt.zero_grad()
        loss.backward()
        opt.step()
    after = loss.item()
    assert after < before
    peft_model.unload()  # leave the shared fixture model unchanged for other tests


def test_encode_options_matches_full_joint_tokenization(lm):
    """The cue-line shortcut must give exactly the tokens of tokenizing prompt + option."""
    from packreadout.scoring import encode_options, tokenize_pieces
    from packreadout.tasks import get_task

    _, tok = lm
    for name in ["banking77", "sst5", "arc_challenge", "strategyqa", "hellaswag_lmeval"]:
        task = get_task(name)
        for e in task.load("test", 3):
            prompt_ids, opt_ids = encode_options(tok, e.prompt, e.options, task.fmt)
            assert prompt_ids == tok(e.prompt).input_ids
            for opt, ids in zip(e.options, opt_ids):
                assert ids == tokenize_pieces(tok, [e.prompt, task.fmt.render(opt)])[1]


def test_cached_matches_naive(lm):
    model, tok = lm
    assert_same(score_options_cached(model, tok, PROMPT, OPTIONS), score_options_naive(model, tok, PROMPT, OPTIONS))
