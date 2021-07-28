raise NotImplementedError

from glob import glob
import json
import seaborn as sns

result_paths = glob("bert-base-uncased_pararel_results_*.json")
results = {}

for p in result_paths:
    with open(p) as f:
        results.update(json.load(f))

to_plot = {}

# plot Figure 3 from the paper -
# the decreasing ratio of the probability of the correct answer after suppressing knowledge neurons
for uuid, data in results.items():
    if to_plot.get(data["relation_name"]) is None:
        to_plot[data["relation_name"]] = {"related": [], "unrelated": []}

    # we want to get the mean change in probability for answers that the model got correct *before* suppressing
    related_data = data["related"]
    related_correct_prob_diff = []
    for prob, correct in zip(
        related_data["prob_diffs"], related_data["correct_before"]
    ):
        if correct:
            related_correct_prob_diff.append(prob)

    unrelated_data = data["unrelated"]
    unrelated_correct_prob_diff = []
    for prob, correct in zip(
        unrelated_data["prob_diffs"], unrelated_data["correct_before"]
    ):
        unrelated_correct_prob_diff.append(prob)

    if data["n_refined_neurons"] > 0 and data["n_unrelated_neurons"] > 0:
        # for some prompts we didn't get any neurons back, it would be unfair to include them
        if related_correct_prob_diff:
            related_correct_prob_diff = sum(related_correct_prob_diff) / len(
                related_correct_prob_diff
            )
            if unrelated_correct_prob_diff:
                unrelated_correct_prob_diff = sum(unrelated_correct_prob_diff) / len(
                    unrelated_correct_prob_diff
                )
            else:
                unrelated_correct_prob_diff = 0.0
            to_plot[data["relation_name"]]["related"].append(related_correct_prob_diff)
            to_plot[data["relation_name"]]["unrelated"].append(
                unrelated_correct_prob_diff
            )

for relation_name, data in to_plot.items():
    if data["related"]:
        data["related"] = sum(data["related"]) / len(data["related"])
    if data["unrelated"]:
        data["unrelated"] = sum(data["unrelated"]) / len(data["unrelated"])
    
print("done!")
