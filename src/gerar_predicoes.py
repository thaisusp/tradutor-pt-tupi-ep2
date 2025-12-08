import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from tqdm import tqdm
import evaluate
import os


def main():
    # Configuração de caminhos

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

    # Caminhos de Entrada 
    DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
    ARQUIVOS_ENTRADA = {
        "Antigo": os.path.join(DATA_DIR, "test.csv"),
        "Moderno": os.path.join(DATA_DIR, "test_modern.csv")
    }

    # Caminho do Modelo - Escolha o modelo desejado (results/models/nllb_finetuned_antigo/final ou results/models/nllb_finetuned_moderno/final)
    MODEL_DIR = os.path.join(PROJECT_ROOT, "results", "models", "nllb_finetuned_antigo/final")

    # Caminhos de Saída 
    OUT_PREDS = os.path.join(PROJECT_ROOT, "results", "predictions", "few_shot")
    OUT_METRICS = os.path.join(PROJECT_ROOT, "results", "metrics")

    # Cria pastas se não existirem
    os.makedirs(OUT_PREDS, exist_ok=True)
    os.makedirs(OUT_METRICS, exist_ok=True)

    # Configuração do Arquivo
    TAG_ARQUIVO = "modelofinetuned" 
    NOME_NA_TABELA = "NLLB Fine-Tuned"

    # Configuração de Hardware
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo: {device}")

    # Carregar modelo treinado
    print(f"Buscando modelo em: {MODEL_DIR}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR).to(device)
        print("Modelo carregado com sucesso!")
    except Exception as e:
        print(f"Erro ao carregar modelo: {e}")
        print(f"Verifique se a pasta '{MODEL_DIR}' existe e contém os arquivos do modelo.")
        exit()

    # Carregar Métricas
    print("Carregando métricas...")
    bleu = evaluate.load("sacrebleu")
    chrf = evaluate.load("chrf")

    # Funções auxiliares
    def traduzir_lista(textos, src_lang, tgt_lang):
        model.eval()
        tokenizer.src_lang = src_lang
        preds = []
        batch_size = 16 
        
        for i in tqdm(range(0, len(textos), batch_size), desc=f"Traduzindo {src_lang}->{tgt_lang}"):
            batch = textos[i : i + batch_size]
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
            bos_id = tokenizer.convert_tokens_to_ids(tgt_lang)
            
            with torch.no_grad():
                gen = model.generate(**inputs, forced_bos_token_id=bos_id, max_length=128)
            
            preds.extend(tokenizer.batch_decode(gen, skip_special_tokens=True))
        return preds

    def calcular_metricas(preds, refs, direcao, dataset_tipo):
        preds = [str(x) if pd.notna(x) else "" for x in preds]
        refs = [[str(x) if pd.notna(x) else ""] for x in refs]
        
        return {
            "Modelo": NOME_NA_TABELA,
            "Dataset": dataset_tipo,
            "Direção": direcao,
            "BLEU": round(bleu.compute(predictions=preds, references=refs)['score'], 2),
            "chrF1": round(chrf.compute(predictions=preds, references=refs, beta=1)['score'], 2),
            "chrF3": round(chrf.compute(predictions=preds, references=refs, beta=3)['score'], 2)
        }

    # Execução
    resultados_finais = []

    for tipo, caminho_csv in ARQUIVOS_ENTRADA.items():
        if not os.path.exists(caminho_csv):
            print(f" Aviso: Arquivo {tipo} não encontrado em {caminho_csv}. Pulando.")
            continue
            
        print(f"\n Processando Dataset: {tipo.upper()}")
        
        try:
            df = pd.read_csv(caminho_csv)
        except:
            df = pd.read_csv(caminho_csv, sep=';')
            
        # Identificar Colunas
        col_pt = None
        if 'pt_moderno_clean' in df.columns: col_pt = 'pt_moderno_clean'
        elif 'modern_source_text' in df.columns: col_pt = 'modern_source_text'
        elif 'source_text' in df.columns: col_pt = 'source_text'
        else: col_pt = df.columns[0]
        
        col_tupi = None
        if 'tupi_clean' in df.columns: col_tupi = 'tupi_clean'
        elif 'target_text' in df.columns: col_tupi = 'target_text'
        else: col_tupi = df.columns[1]
        
        print(f"   Colunas: PT='{col_pt}' | Tupi='{col_tupi}'")
        df = df.dropna(subset=[col_pt, col_tupi])
        
        # PT -> Tupi (Ida) 
        print(f"     Gerando Ida (PT -> Tupi)...")
        preds_ida = traduzir_lista(df[col_pt].tolist(), "por_Latn", "grn_Latn")
        
        metrics_ida = calcular_metricas(preds_ida, df[col_tupi].tolist(), "PT -> Tupi", tipo)
        resultados_finais.append(metrics_ida)
        
        # Salvar CSV Ida
        nome_arq_ida = f"traducoes_{tipo.lower()}_ida_{TAG_ARQUIVO}.csv"
        caminho_ida = os.path.join(OUT_PREDS, nome_arq_ida)
        pd.DataFrame({'source': df[col_pt], 'target': df[col_tupi], 'prediction': preds_ida}).to_csv(caminho_ida, index=False)
        print(f"      Salvo em: {caminho_ida}")

        # Tupi -> PT 
        print(f"   ⬅Gerando Volta (Tupi -> PT)...")
        preds_volta = traduzir_lista(df[col_tupi].tolist(), "grn_Latn", "por_Latn")
        
        metrics_volta = calcular_metricas(preds_volta, df[col_pt].tolist(), "Tupi -> PT", tipo)
        resultados_finais.append(metrics_volta)
        
        # Salvar CSV Volta
        nome_arq_volta = f"traducoes_{tipo.lower()}_volta_{TAG_ARQUIVO}.csv"
        caminho_volta = os.path.join(OUT_PREDS, nome_arq_volta)
        pd.DataFrame({'source': df[col_tupi], 'target': df[col_pt], 'prediction': preds_volta}).to_csv(caminho_volta, index=False)
        print(f"      Salvo em: {caminho_volta}")

    # Salvar tudo
    if resultados_finais:
        df_final = pd.DataFrame(resultados_finais)
        df_final = df_final.sort_values(by=['Dataset', 'Direção'])
        
        print(f"\nTABELA DE RESULTADOS: {NOME_NA_TABELA} ")
        print(df_final.to_string(index=False))
        
        arquivo_final = os.path.join(OUT_METRICS, f"RELATORIO_FINAL_NLLB_{TAG_ARQUIVO.upper()}.csv")
        df_final.to_csv(arquivo_final, index=False)
        print(f"\nRelatório salvo com sucesso em: {arquivo_final}")
    else:
        print("Nenhum resultado gerado.")

if __name__ == "__main__":
    main()