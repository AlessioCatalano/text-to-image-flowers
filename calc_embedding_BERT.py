import os
import torch
from transformers import BertTokenizer, BertModel
import numpy as np

# Funzione per calcolare l'embedding BERT
def get_bert_embedding(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

# Percorso alla cartella delle classi e alla cartella di destinazione
classes_path = '/content/drive/MyDrive/Colab Notebooks/Flowers/cvpr2016_flowers/text_c10'
dest_path = '/content/file_pt'

# Inizializzazione del tokenizer e del modello BERT
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
model = BertModel.from_pretrained('bert-large-uncased', output_hidden_states=True)

# Itera su ogni classe
for class_folder in sorted(os.listdir(classes_path)):
    if class_folder.startswith("class"):
        class_path = os.path.join(classes_path, class_folder)
        class_dest_path = os.path.join(dest_path, class_folder)
        os.makedirs(class_dest_path, exist_ok=True)

        # Itera su ogni file .txt nella classe
        for txt_file in sorted(os.listdir(class_path)):
            if txt_file.endswith(".txt"):
                file_path = os.path.join(class_path, txt_file)
                with open(file_path, 'r') as f:
                    lines = f.readlines()

                # Calcola gli embeddings
                embeddings = [get_bert_embedding(line.strip(), tokenizer, model) for line in lines]
                embeddings_array = np.array(embeddings).squeeze()

                # Salva il file .pt
                pt_filename = txt_file.replace(".txt", ".pt")
                pt_file_path = os.path.join(class_dest_path, pt_filename)
                torch.save({'img': txt_file.replace('txt', 'jpg'), 'txt': embeddings_array}, pt_file_path)
