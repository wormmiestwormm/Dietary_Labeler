from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Load the fine-tuned model and tokenizer
model_path = "../trainer_output/model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

clf = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer
)

if __name__ == "__main__":
    user_input = input("Enter your ingredients list: ")
    print("You entered: " + user_input)
    
    # Run prediction
    result = clf(user_input)
    if result[0]['label'] == 'LABEL_1':
        print("The ingredients list is likely to contain allergens.")
    else:
        print("The ingredients list is unlikely to contain allergens.")