import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, MBart50TokenizerFast, MBartForConditionalGeneration
from tqdm import tqdm
import evaluate
import unicodedata
import re
import os
import gc
import numpy as np
from sklearn.model_selection import train_test_split



# SETUP E INSTALAÇÃO

# Configurar GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Dispositivo: {device}")

# Função para limpar RAM 
def limpar_memoria():
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    print("Memória limpa.")

# Função de Limpeza de Texto
def limpar_texto(text):
    if not isinstance(text, str): return ""
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'[«»""“”…*_]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def main():
    
    # PREPARAÇÃO DOS DADOS (Antigo e Moderno)

    print("\n Preparando dados")

    # Ajuste de caminhos
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    # Raiz do projeto (uma pasta acima)
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

    # Pastas de Dados
    DATA_RAW = os.path.join(PROJECT_ROOT, "data", "raw")
    DATA_PROC = os.path.join(PROJECT_ROOT, "data", "processed")
    RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "predictions", "zero-shot")

    # Garante que as pastas existem
    os.makedirs(DATA_PROC, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Arquivos Originais
    ARQUIVO_ANTIGO = os.path.join(DATA_RAW, "Cópia de portugues-guarani-tupi antigo.xlsx")
    ARQUIVO_MODERNO = os.path.join(DATA_RAW, "tupiantigo_portugues_moderno.csv")

    # Arquivos Processados
    TEST_ANTIGO = os.path.join(DATA_PROC, "test_antigo.csv")
    TEST_MODERNO = os.path.join(DATA_PROC, "test_moderno.csv")

    # DADOS ANTIGOS
    if os.path.exists(ARQUIVO_ANTIGO):
        try:
            print(f"Processando: {ARQUIVO_ANTIGO}")
            df_raw = pd.read_excel(ARQUIVO_ANTIGO)
            # Ajuste: assume 1ª col=PT, 2ª col=Tupi
            df_raw.columns = ['source_text', 'target_text'] + list(df_raw.columns[2:])
            df_raw = df_raw.dropna(subset=['source_text', 'target_text'])
            df_raw['tupi_clean'] = df_raw['target_text'].apply(limpar_texto)
            
            # Split 70/15/15
            train, temp = train_test_split(df_raw, test_size=0.30, random_state=42)
            val, test = train_test_split(temp, test_size=0.50, random_state=42)
            
            test.to_csv(TEST_ANTIGO, index=False)
            print(f" Dados Antigos prontos: {TEST_ANTIGO}")
        except Exception as e:
            print(f"Erro ao processar arquivo antigo: {e}")
    else:
        print(f"Erro: arquivo antigo não encontrado em {ARQUIVO_ANTIGO}")

    # DADOS MODERNOS 
    if os.path.exists(ARQUIVO_MODERNO):
        try:
            print(f"Processando: {ARQUIVO_MODERNO}")
            try:
                df_mod = pd.read_csv(ARQUIVO_MODERNO, sep=';') # Tenta ponto e virgula
            except:
                df_mod = pd.read_csv(ARQUIVO_MODERNO) # Tenta virgula
            
            # Identifica colunas (PT Moderno, Tupi)
            col_pt = df_mod.columns[0]
            col_tupi = df_mod.columns[1]
            
            df_mod['pt_moderno_clean'] = df_mod[col_pt].apply(limpar_texto)
            df_mod['tupi_clean'] = df_mod[col_tupi].apply(limpar_texto)
            
            # Cria teste moderno
            _, temp_m = train_test_split(df_mod, test_size=0.30, random_state=42)
            _, test_m = train_test_split(temp_m, test_size=0.50, random_state=42)
            
            test_m.to_csv(TEST_MODERNO, index=False)
            print(f"Dados Modernos prontos: {TEST_MODERNO}")
        except Exception as e:
            print(f"ERRO ao processar arquivo moderno: {e}")
    else:
        print(f"ERRO: Faltando arquivo moderno em {ARQUIVO_MODERNO}")


    # PIPELINE mBART-50 (ZERO-SHOT)
    print("\n Iniciando mBART-50")
    try:
        # Limpeza preventiva
        if 'model' in locals(): del model
        if 'tokenizer' in locals(): del tokenizer
        limpar_memoria()

        tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
        model = MBartForConditionalGeneration.from_pretrained(
            "facebook/mbart-large-50-many-to-many-mmt",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32, 
            low_cpu_mem_usage=True
        ).to(device)
        
        def run_mbart(df, col_in, src_code, tgt_code):
            model.eval()
            tokenizer.src_lang = src_code
            res = []
            # Batch size ajustado
            batch_size = 16
            for i in tqdm(range(0, len(df), batch_size), desc="mBART"):
                batch = df[col_in].iloc[i:i+batch_size].tolist()
                inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
                gen = model.generate(**inputs, forced_bos_token_id=tokenizer.lang_code_to_id[tgt_code], max_length=128)
                res.extend(tokenizer.batch_decode(gen, skip_special_tokens=True))
            return res

        # Antigo 
        if os.path.exists(TEST_ANTIGO):
            print("Processando mBART Antigo...")
            df = pd.read_csv(TEST_ANTIGO)
            df['pred_ida'] = run_mbart(df, 'source_text', "pt_XX", "pt_XX")
            df['pred_volta'] = run_mbart(df, 'tupi_clean', "pt_XX", "pt_XX")
            df.to_csv(os.path.join(RESULTS_DIR, 'results_mbart_antigo.csv'), index=False)
            print("Salvo.")
            
        # Moderno 
        if os.path.exists(TEST_MODERNO):
            print("Processando mBART Moderno...")
            df = pd.read_csv(TEST_MODERNO)
            df['pred_ida'] = run_mbart(df, 'pt_moderno_clean', "pt_XX", "pt_XX")
            df['pred_volta'] = run_mbart(df, 'tupi_clean', "pt_XX", "pt_XX")
            df.to_csv(os.path.join(RESULTS_DIR, 'results_mbart_moderno.csv'), index=False)
            print("Salvo.")

        del model, tokenizer
        limpar_memoria()
    except Exception as e: print(f"Erro mBART: {e}")

    # PIPELINE NLLB-200 (ZERO-SHOT)
    print("\n Iniciando NLLB-200")
    try:
        if 'model' in locals(): del model
        if 'tokenizer' in locals(): del tokenizer
        limpar_memoria()

        tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
        model = AutoModelForSeq2SeqLM.from_pretrained(
            "facebook/nllb-200-distilled-600M", 
            torch_dtype=torch.float16 if device == "cuda" else torch.float32, 
            low_cpu_mem_usage=True
        ).to(device)
        
        def run_nllb(df, col_in, src_code, tgt_code):
            model.eval()
            tokenizer.src_lang = src_code
            res = []
            batch_size = 16
            for i in tqdm(range(0, len(df), batch_size), desc="NLLB"):
                batch = df[col_in].iloc[i:i+batch_size].tolist()
                inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
                # NLLB usa convert_tokens
                bos = tokenizer.convert_tokens_to_ids(tgt_code)
                gen = model.generate(**inputs, forced_bos_token_id=bos, max_length=128)
                res.extend(tokenizer.batch_decode(gen, skip_special_tokens=True))
            return res

        # Antigo (PT->Tupi via Guarani)
        if os.path.exists(TEST_ANTIGO):
            print("Processando NLLB Antigo...")
            df = pd.read_csv(TEST_ANTIGO)
            df['pred_ida'] = run_nllb(df, 'source_text', "por_Latn", "grn_Latn")
            df['pred_volta'] = run_nllb(df, 'tupi_clean', "grn_Latn", "por_Latn")
            df.to_csv(os.path.join(RESULTS_DIR, 'results_nllb_antigo.csv'), index=False)
            print("Salvo.")

        # Moderno (PT->Tupi via Guarani)
        if os.path.exists(TEST_MODERNO):
            print("Processando NLLB Moderno...")
            df = pd.read_csv(TEST_MODERNO)
            df['pred_ida'] = run_nllb(df, 'pt_moderno_clean', "por_Latn", "grn_Latn")
            df['pred_volta'] = run_nllb(df, 'tupi_clean', "grn_Latn", "por_Latn")
            df.to_csv(os.path.join(RESULTS_DIR, 'results_nllb_moderno.csv'), index=False)
            print("Salvo.")

        del model, tokenizer
        limpar_memoria()
    except Exception as e: print(f"Erro NLLB: {e}")

    # PIPELINE mT5-small (ZERO-SHOT)

    print("\n Iniciando mT5-small")
    try:
        if 'model' in locals(): del model
        if 'tokenizer' in locals(): del tokenizer
        limpar_memoria()

        tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")
        model = AutoModelForSeq2SeqLM.from_pretrained(
            "google/mt5-small", 
            torch_dtype=torch.float16 if device == "cuda" else torch.float32, 
            low_cpu_mem_usage=True
        ).to(device)
        
        def run_mt5(df, col_in, prompt):
            model.eval()
            res = []
            inputs_text = [prompt + str(t) for t in df[col_in].tolist()]
            batch_size = 32
            for i in tqdm(range(0, len(df), batch_size), desc="mT5"):
                batch = inputs_text[i:i+batch_size]
                inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
                gen = model.generate(**inputs, max_length=128, num_beams=4, early_stopping=True)
                res.extend(tokenizer.batch_decode(gen, skip_special_tokens=True))
            return res

        # Antigo
        if os.path.exists(TEST_ANTIGO):
            print("Processando mT5 Antigo...")
            df = pd.read_csv(TEST_ANTIGO)
            df['pred_ida'] = run_mt5(df, 'source_text', "translate Portuguese to Tupi: ")
            df['pred_volta'] = run_mt5(df, 'tupi_clean', "translate Tupi to Portuguese: ")
            df.to_csv(os.path.join(RESULTS_DIR, 'results_mt5_antigo.csv'), index=False)
            print("Salvo.")

        # Moderno
        if os.path.exists(TEST_MODERNO):
            print("Processando mT5 Moderno...")
            df = pd.read_csv(TEST_MODERNO)
            df['pred_ida'] = run_mt5(df, 'pt_moderno_clean', "translate Portuguese to Tupi: ")
            df['pred_volta'] = run_mt5(df, 'tupi_clean', "translate Tupi to Portuguese: ")
            df.to_csv(os.path.join(RESULTS_DIR, 'results_mt5_moderno.csv'), index=False)
            print("Salvo.")

        del model, tokenizer
        limpar_memoria()
    except Exception as e: print(f"Erro mT5: {e}")

    # TABELA FINAL
    print("\n Gerando tabela final...")
    bleu = evaluate.load("sacrebleu")
    chrf = evaluate.load("chrf")

    def calc_met(df, col_pred, col_ref, modelo, direcao, tipo):
        if col_pred not in df.columns: return None
        preds = [str(x) if pd.notna(x) else "" for x in df[col_pred]]
        refs = [[str(x) if pd.notna(x) else ""] for x in df[col_ref]]
        
        return {
            "Modelo": modelo, "Dados": tipo, "Direção": direcao,
            "BLEU": round(bleu.compute(predictions=preds, references=refs)['score'], 2),
            "chrF1": round(chrf.compute(predictions=preds, references=refs, beta=1)['score'], 2),
            "chrF3": round(chrf.compute(predictions=preds, references=refs, beta=3)['score'], 2)
        }

    lista_res = []

    # Loop para processar os 6 arquivos gerados
    configs = [
        # (Arquivo, Modelo, Tipo, Col_Ida, Col_Volta)
        ('results_mbart_antigo.csv', 'mBART', 'Arcaico', 'pred_ida', 'pred_volta'),
        ('results_mbart_moderno.csv', 'mBART', 'Moderno', 'pred_ida', 'pred_volta'),
        ('results_nllb_antigo.csv', 'NLLB', 'Arcaico', 'pred_ida', 'pred_volta'),
        ('results_nllb_moderno.csv', 'NLLB', 'Moderno', 'pred_ida', 'pred_volta'),
        ('results_mt5_antigo.csv', 'mT5', 'Arcaico', 'pred_ida', 'pred_volta'),
        ('results_mt5_moderno.csv', 'mT5', 'Moderno', 'pred_ida', 'pred_volta'),
    ]

    for arquivo, mod, tipo, c_ida, c_volta in configs:
        path_arq = os.path.join(RESULTS_DIR, arquivo)
        if os.path.exists(path_arq):
            df = pd.read_csv(path_arq)
            col_pt_ref = 'pt_moderno_clean' if tipo == 'Moderno' else 'source_text'
            
            lista_res.append(calc_met(df, c_ida, 'tupi_clean', mod, 'PT->Tupi', tipo))
            lista_res.append(calc_met(df, c_volta, col_pt_ref, mod, 'Tupi->PT', tipo))

    df_final = pd.DataFrame([x for x in lista_res if x is not None])
    if not df_final.empty:
        df_final = df_final.sort_values(by=['Modelo', 'Dados', 'Direção'])
        print("\nTABELA DE RESULTADOS FINAIS (ZERO-SHOT)")
        print(df_final.to_string(index=False))
        df_final.to_csv(os.path.join(RESULTS_DIR, 'RELATORIO_FINAL_ZEROSHOT.csv'), index=False)
    else:
        print("Nenhum resultado gerado para a tabela final.")

if __name__ == "__main__":
    main()