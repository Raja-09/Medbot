import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import LlamaForSequenceClassification
from transformers import LlamaTokenizer
import numpy as np 
df = pd.read_csv("symptoms_diseases.csv")

tokenizer = LlamaTokenizer.from_pretrained("nvidia/Llama3-ChatQA-1.5-70B")
df["input_ids"] = df["symptoms"].apply(lambda x: tokenizer.encode(x, add_special_tokens=True))
df["attention_mask"] = df["input_ids"].apply(lambda x: [1] * len(x))

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

model = LlamaForSequenceClassification.from_pretrained("nvidia/Llama3-ChatQA-1.5-70B", num_labels=8)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(train_df["input_ids"], train_df["attention_mask"], epochs=5)

loss, accuracy = model.evaluate(test_df["input_ids"], test_df["attention_mask"])
print(f"Test loss: {loss:.3f}, Test accuracy: {accuracy:.3f}")

new_patient = "headache, fatigue, and fever"
input_ids = tokenizer.encode(new_patient, add_special_tokens=True)
attention_mask = [1] * len(input_ids)
prediction = model.predict([input_ids], [attention_mask])
predicted_disease = df["disease"].iloc[np.argmax(prediction)]
print(f"Predicted disease: {predicted_disease}")