from lm_eval.filters.extraction import MultiChoiceRegexFilter, StripThinkFilter
from lm_eval.filters import build_filter_ensemble
from lm_eval.api.instance import Instance


def test_multi_choice_regex_all_empty_capture_groups_falls_back_to_choice_text():
    filt = MultiChoiceRegexFilter(
        regex_pattern=r"()()",
        ignore_case=True,
        ignore_punctuation=True,
    )

    resps = [["alpha"]]
    docs = [{"choices": ["alpha", "beta"]}]

    assert filt.apply(resps, docs) == [["(A)"]]


def test_multi_choice_regex_all_empty_capture_groups_falls_back_to_bare_letter():
    filt = MultiChoiceRegexFilter(regex_pattern=r"()()")

    resps = [[": B"]]
    docs = [{"choices": ["alpha", "beta"]}]

    assert filt.apply(resps, docs) == [["(B)"]]


def test_redux_filter_ignores_reasoning_but_keeps_original_responses():
    pipeline = build_filter_ensemble(
        "default",
        [
            ("strip_think", None),
            ("regex", {"regex_pattern": r"\b([A-D])\b", "group_select": -1}),
            ("take_first", None),
        ],
    )
    responses = [
        ["<think>Option A looks right</think>\nFinal answer: C"],
        ["<think>Option A looks right, but let me check"],
        ["B"],
    ]

    instances = [Instance(request_type="generate_until", doc={}, arguments=("", {}), idx=index)
                 for index in range(len(responses))]
    for instance, response in zip(instances, responses):
        instance.resps = response
    pipeline.apply(instances)

    assert [instance.filtered_resps["default"] for instance in instances] == ["C", "[invalid]", "B"]
    assert responses[0][0].startswith("<think>")
    assert StripThinkFilter().apply(responses, [{}, {}, {}])[1] == [""]
