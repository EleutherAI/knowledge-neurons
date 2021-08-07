from knowledge_neurons import (
    KnowledgeNeurons,
    initialize_model_and_tokenizer,
    model_type,
)
import random

MODEL_NAME = "bert-base-multilingual-uncased"
TEXT = "Sarah was visiting [MASK], the capital of france"
GROUND_TRUTH = "paris"
BATCH_SIZE = 10
STEPS = 20
PERCENTILE = 99.5
ENG_TEXTS = [
    "Sarah was visiting [MASK], the capital of france",
    "The capital of france is [MASK]",
    "[MASK] is the capital of france",
    "France's capital [MASK] is a hotspot for romantic vacations",
    "The eiffel tower is situated in [MASK]",
    "[MASK] is the most populous city in france",
    "[MASK], france's capital, is one of the most popular tourist destinations in the world",
]
FRENCH_TEXTS = [
    "Sarah visitait [MASK], la capitale de la france",
    "La capitale de la france est [MASK]",
    "[MASK] est la capitale de la france",
    "La capitale de la France [MASK] est un haut lieu des vacances romantiques",
    "La tour eiffel est située à [MASK]",
    "[MASK] est la ville la plus peuplée de france",
    "[MASK], la capitale de la france, est l'une des destinations touristiques les plus prisées au monde",
]

TEXTS = ENG_TEXTS + FRENCH_TEXTS
P = 0.5

# setup model
ml_model, tokenizer = initialize_model_and_tokenizer(MODEL_NAME)
kn_ml = KnowledgeNeurons(ml_model, tokenizer)

refined_neurons_eng = kn_ml.get_refined_neurons(
    ENG_TEXTS,
    GROUND_TRUTH,
    p=P,
    batch_size=BATCH_SIZE,
    steps=STEPS,
)
refined_neurons_fr = kn_ml.get_refined_neurons(
    FRENCH_TEXTS,
    GROUND_TRUTH,
    p=P,
    batch_size=BATCH_SIZE,
    steps=STEPS,
)
refined_neurons = kn_ml.get_refined_neurons(
    TEXTS,
    GROUND_TRUTH,
    p=P,
    batch_size=BATCH_SIZE,
    steps=STEPS,
)

# how many neurons are shared between the french prompts and the english ones?

print("N french neurons: ", len(refined_neurons_fr))
print("N english neurons: ", len(refined_neurons_eng))
shared_neurons = [i for i in refined_neurons_eng if i in refined_neurons_fr]
print(f"N shared neurons: ", len(shared_neurons))

print("\nSuppressing refined neurons: \n")
results_dict, unpatch_fn = kn_ml.suppress_knowledge(
    TEXT, GROUND_TRUTH, refined_neurons
)

print("\nSuppressing random neurons: \n")
random_neurons = [
    [
        random.randint(0, ml_model.config.num_hidden_layers - 1),
        random.randint(0, ml_model.config.intermediate_size - 1),
    ]
    for i in range(len(refined_neurons))
]
results_dict, unpatch_fn = kn_ml.suppress_knowledge(
    TEXT, GROUND_TRUTH, random_neurons
)

print("\nSuppressing refined neurons for an unrelated prompt: \n")
results_dict, unpatch_fn = kn_ml.suppress_knowledge(
    "[MASK] is the official language of the solomon islands",
    "english",
    refined_neurons,
)

print(
    "\nSuppressing refined neurons (found by french text) using english prompt: \n"
)
results_dict, unpatch_fn = kn_ml.suppress_knowledge(
    TEXT, GROUND_TRUTH, refined_neurons_fr
)

print("\nEnhancing refined neurons: \n")
results_dict, unpatch_fn = kn_ml.enhance_knowledge(
    TEXT, GROUND_TRUTH, refined_neurons
)

print("\nEnhancing random neurons: \n")
results_dict, unpatch_fn = kn_ml.enhance_knowledge(
    TEXT, GROUND_TRUTH, random_neurons
)