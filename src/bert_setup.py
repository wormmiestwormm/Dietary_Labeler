from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import Dataset

import torch
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd

# Load the food.csv file into a DataFrame
df_food = pd.read_csv('../data/food.csv')
df_ingredients = pd.read_csv('../data/branded_food.csv')


# select needed columns, ignoring every other column
selected_columns_food= df_food[['fdc_id']]
selected_columns_ingredients = df_ingredients[['fdc_id', 'ingredients']]


# merge two dataframes on fdc_id
merged_df = pd.merge(selected_columns_food, selected_columns_ingredients, on='fdc_id', how='inner')
print(merged_df.shape)

#Dataset from merged dataframe
dataset = Dataset.from_pandas(merged_df.head(10000))

print(dataset.shape)

allergens = []

def clean_data(example):
    ingredients = example['ingredients']
    # Check if ingredients is None or if it's an empty string after stripping whitespace
    if ingredients is None or (isinstance(ingredients, str) and ingredients.strip() == ''):
        return False
    return True

# Apply filter to remove invalid rows
dataset = dataset.filter(clean_data)

print(dataset.features)
print("Dataset filtered successfully.")
print(dataset.shape)

allergens = [
    "butter", "casein",
    "cheese", "cream", "curd", "custard", "ghee", "half-and-half",
    "lactalbumin", "lactalbumin phosphate", "lactic acid starter culture", "lactoferrin",
    "lactoglobulin", "lactose", "lactulose", "milk",
    "pudding", "recaldent", "simplesse", "tagatose", "whey",
    "yogurt", "albumin","albumen", "apovitellin",
    "avidin globulin", "egg", "eggnog", "lysozyme",
    "mayonnaise", "meringue", "ovalbumin", "ovomucoid", "ovomucin",
    "ovovitellin", "surimi", "vitellin", "arachis oil",
    "lupin", "mandelonas", "soy",  "edamame", "miso", "natto", "okara", "shoyu",
    "tamari", "tempeh",
    "textured vegetable protein", "tvp", "tofu", "bread crumbs", "bulgur",
    "cereal extract", "couscous", "cracker meal", "einkorn", "emmer", "matzo",
    "matza", "pasta", "seitan", "semolina", "peanut",
    "spelt", "triticale",
    "wheat","bran", "durum", "germ", "gluten", "grass", "malt", "sprouts", "starch","almond",
    "cashew", "filbert", "gianduja", "marzipan",
    "nut", "pecan", "pesto",
    "pistachio", "praline", "barnacle", "crab", "crawfish", "crayfish", "ecrevisse",
    "krill", "lobster", "langouste", "langoustine", "moreton bay bugs", "scampi", "tomalley",
    "prawns", "shrimp", "crevette", "benne", "benniseed", "gingelly", "gomasio", "halvah", "sesame", "sesamol", "sesamum indicum", "sim sim",
    "tahini", "tahina", "tehina", "til",    "anchovies", "bass","catfish","cod", "flounder", "grouper",
    "haddock", "hake", "halibut", "herring", "mahi mahi","perch","pike","pollock",
    "salmon","scrod","sole","snapper","swordfish","tilapia","trout","tuna","surimi"
    ]

allergens = {
    "milk": [
        "butter", "casein", "cheese", "cream", "curd", "custard", "ghee", "half-and-half",
        "lactalbumin", "lactalbumin phosphate", "lactic acid starter culture", "lactoferrin",
        "lactoglobulin", "lactose", "lactulose", "milk", "pudding", "recaldent", "simplesse",
        "tagatose", "whey", "yogurt"
    ],
    "eggs": [
        "albumin", "albumen", "apovitellin", "avidin globulin", "egg", "eggnog", "lysozyme",
        "mayonnaise", "meringue", "ovalbumin", "ovomucoid", "ovomucin", "ovovitellin",
        "surimi", "vitellin"
    ],
    "fish": [
        "anchovies", "bass", "catfish", "cod", "flounder", "grouper", "haddock", "hake",
        "halibut", "herring", "mahi mahi", "perch", "pike", "pollock", "salmon", "scrod",
        "sole", "snapper", "swordfish", "tilapia", "trout", "tuna", "surimi"
    ],
    "shellfish": [
        "barnacle", "crab", "crawfish", "crayfish", "ecrevisse", "krill", "lobster",
        "langouste", "langoustine", "moreton bay bugs", "scampi", "tomalley", "prawns",
        "shrimp", "crevette"
    ],
    "nuts": [
        "arachis oil", "lupin", "mandelonas", "peanut", "almond", "cashew", "filbert",
        "gianduja", "marzipan", "nut", "pecan", "pesto", "pistachio", "praline"
    ],
    "soy": [
        "soy", "edamame", "miso", "natto", "okara", "shoyu", "tamari", "tempeh",
        "textured vegetable protein", "tvp", "tofu"
    ],
    "wheat": [
        "bread crumbs", "bulgur", "cereal extract", "couscous", "cracker meal", "einkorn",
        "emmer", "matzo", "matza", "pasta", "seitan", "semolina", "spelt", "triticale",
        "wheat", "bran", "durum", "germ", "gluten", "grass", "malt", "sprouts", "starch"
    ],
    "sesame": [
        "benne", "benniseed", "gingelly", "gomasio", "halvah", "sesame", "sesamol",
        "sesamum indicum", "sim sim", "tahini", "tahina", "tehina", "til"
    ]
}

def detect_allergens(dataset):
    ingredients = str(dataset['ingredients']).lower()

    for category, keywords in allergens.items():
        dataset[category + "_present"] = int(
            any(keyword in ingredients for keyword in keywords)
        )

    return dataset

def assign_label(dataset):
    dataset["label"] = int(
        dataset["milk_present"] +
        dataset["eggs_present"] +
        dataset["fish_present"] +
        dataset["shellfish_present"] +
        dataset["nuts_present"] +
        dataset["soy_present"] +
        dataset["wheat_present"] +
        dataset["sesame_present"] > 0
    )
    return dataset
# Apply the function to the dataset
dataset_with_allergens = dataset.map(detect_allergens)
dataset_with_allergens = dataset_with_allergens.map(assign_label)

print(dataset_with_allergens.features)
print(dataset_with_allergens[0]) 
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(batch):
    texts = [str(text) for text in batch["ingredients"]]

    tok = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=256
    )

    tok["labels"] = batch["label"]
    return tok

# Tokenize the dataset
tokenized_ds = dataset_with_allergens.map(tokenize, batched=True)

print(tokenized_ds.features)

# Split the dataset into train and test sets
tokenized_ds = tokenized_ds.train_test_split(test_size=0.15)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='binary'
    )
    acc = accuracy_score(labels, preds)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# Load pre-trained model for sequence classification
num_labels = 2
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels
)

# Define training arguments
training_args = TrainingArguments(
    eval_strategy="epoch",
    num_train_epochs=3,
    per_device_train_batch_size=10,
    per_device_eval_batch_size=10,
    learning_rate=2e-5,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["test"],
    compute_metrics=compute_metrics
)
trainer.train()

model.save_pretrained("../trainer_output/model")
tokenizer.save_pretrained("../trainer_output/model")