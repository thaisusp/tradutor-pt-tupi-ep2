import pandas as pd
import torch
import gc
import re
import unicodedata
import os
import platform
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer, 
    DataCollatorForSeq2Seq
)

# Configuração de caminhos
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Pega a raiz do projeto (uma pasta acima)
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Caminhos absolutos baseados na raiz
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "models", "nllb_finetuned_moderno") # Mude o nome caso necessário

# Garante que a pasta de saída existe
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Configurações do Modelo
MODEL_ID = "facebook/nllb-200-distilled-600M"

def limpar_memoria():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

def limpar_texto(text):
    if not isinstance(text, str): return ""
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def carregar_csv(nome_arquivo):
    caminho = os.path.join(DATA_DIR, nome_arquivo)
    
    if not os.path.exists(caminho):
        print(f"Arquivo não encontrado: {caminho}")
        return None
        
    try:
        df = pd.read_csv(caminho, sep=',', engine='python', on_bad_lines='skip')
        cols = df.columns
        if len(cols) >= 2:
            df = df.rename(columns={cols[0]: 'pt', cols[1]: 'tupi'})
        
        df['pt'] = df['pt'].apply(limpar_texto).astype(str)
        df['tupi'] = df['tupi'].apply(limpar_texto).astype(str)
        
        df = df[(df['pt'].str.len() > 2) & (df['tupi'].str.len() > 2)]
        df = df.reset_index(drop=True)
        return df
    except Exception as e:
        print(f"Erro ao ler {caminho}: {e}")
        return None

# CPU ou GPU
def obter_dispositivo():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available(): # Verifica se é Mac com chip M1/M2/M3
        return "mps"
    else:
        return "cpu"

def main():
    limpar_memoria()
    
    # Setup do Dispositivo
    device = obter_dispositivo()
    print(f"Rodando em: {device.upper()}")
    if device == "mps":
        print("Aceleração da GPU (Metal)")
    elif device == "cpu":
        print("Rodando na CPU")

    # Carregar Modelo e Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, src_lang="por_Latn", use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID).to(device)

    # Preparar Dados (Agora aponta para a pasta correta automaticamente)
    print(f"Carregando dados de {DATA_DIR}...")
    df_train = carregar_csv('train.csv')
    df_val = carregar_csv('val.csv')

    if df_train is None or df_val is None:
        print("Faltando arquivos train.csv ou val.csv na pasta data/processed.")
        return

    raw_datasets = DatasetDict({
        'train': Dataset.from_pandas(df_train),
        'validation': Dataset.from_pandas(df_val)
    })

    # Tokenização
    def preprocess_function(examples):
        inputs = examples["pt"]
        targets = examples["tupi"]
        
        tokenizer.src_lang = "por_Latn"
        model_inputs = tokenizer(inputs, max_length=128, truncation=True)
        
        tokenizer.src_lang = "grn_Latn"
        labels_batch = tokenizer(targets, max_length=128, truncation=True)
        
        model_inputs["labels"] = labels_batch["input_ids"]
        return model_inputs

    print("Tokenizando dataset...")
    tokenized_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        load_from_cache_file=False,
        remove_columns=raw_datasets["train"].column_names
    )

    # Configuração do Treino 
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, 
        model=model, 
        padding=True,
        pad_to_multiple_of=8
    )

    # Detecta se deve usar FP16
    usar_fp16 = (device == "cuda") 

    args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=5e-5,
        per_device_train_batch_size=4, 
        gradient_accumulation_steps=2,
        num_train_epochs=5,
        
        fp16=usar_fp16, 
        
        eval_strategy="epoch",  
        save_strategy="epoch",
        
        save_total_limit=1,
        logging_steps=50,
        dataloader_num_workers=0, 
        report_to="none"
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    print("Iniciando Fine-Tuning")
    trainer.train()
    
    print("Treino concluído")
    
    # Salva na pasta results/models/nllb_finetuned
    final_path = os.path.join(OUTPUT_DIR, "final")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"Modelo salvo em: {final_path}")

if __name__ == "__main__":
    main()