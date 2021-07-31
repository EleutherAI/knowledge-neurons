from glob import glob
import json
import seaborn as sns
import pandas as pd 



# plot Figure 3 + 4 from the paper -
# the decreasing ratio of the probability of the correct answer after suppressing knowledge neurons

def format_data(results_data, key='suppression'):
    formatted = {}
    for uuid, data in results_data.items():
        if formatted.get(data["relation_name"]) is None:
            formatted[data["relation_name"]] = {"related": [], "unrelated": []}

        related_data = data[key]["related"]
        related_change = []
        for prob in related_data["pct_change"]:
            related_change.append(prob)

        unrelated_data = data[key]["unrelated"]
        unrelated_change = []
        for prob in unrelated_data["pct_change"]:
            unrelated_change.append(prob)

        if data["n_refined_neurons"] > 0 and data["n_unrelated_neurons"] > 0:
            # for some prompts we didn't get any neurons back, it would be unfair to include them
            if related_change:
                related_change = sum(related_change) / len(
                    related_change
                )
                if unrelated_change:
                    unrelated_change = sum(unrelated_change) / len(
                        unrelated_change
                    )
                else:
                    unrelated_change = 0.0
                formatted[data["relation_name"]]["related"].append(related_change)
                formatted[data["relation_name"]]["unrelated"].append(
                    unrelated_change
                )

    for relation_name, data in formatted.items():
        if data["related"]:
            data["related"] = sum(data["related"]) / len(data["related"])
        else:
            data["related"] = float("nan")
        if data["unrelated"]:
            data["unrelated"] = sum(data["unrelated"]) / len(data["unrelated"])
        else:
            data["unrelated"] = float("nan")

    pandas_format = {'relation_name': [], 'related': [], 'pct_change': []}
    for relation_name, data in formatted.items():
        pandas_format['relation_name'].append(relation_name)
        pandas_format['pct_change'].append(data['related'])
        pandas_format['related'].append("Suppressing knowledge neurons for related facts")

        pandas_format['relation_name'].append(relation_name)
        pandas_format['pct_change'].append(data['unrelated'])
        pandas_format['related'].append("Suppressing knowledge neurons for unrelated facts")
    return pd.DataFrame(pandas_format).dropna()

def plot_data(pd_df, title, out_path='test.png'):
    sns.set_theme(style="whitegrid")

    # Draw a nested barplot by species and sex
    g = sns.catplot(
        data=pd_df, kind="bar",
        x="relation_name", y="pct_change", hue="related",
        ci="sd", palette="dark", alpha=.6, height=6, aspect=4
    )
    g.despine(left=True)
    g.set_axis_labels("relation name", "Correct probability change")
    g.legend.set_title(title)
    g.savefig(out_path)

if __name__ == "__main__":
    result_paths = glob("bert-base-uncased_pararel_results_*.json")
    results = {}

    for p in result_paths:
        with open(p) as f:
            results.update(json.load(f))
    
    suppression_data = format_data(results, key='suppression')

    plot_data(suppression_data, "Suppressing knowledge neurons", "suppress.png")

    enhancement_data = format_data(results, key='enhancement')

    plot_data(enhancement_data, "Enhancing knowledge neurons", "enhance.png")