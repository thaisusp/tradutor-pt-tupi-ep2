import os
import sys

from numpy._core.numeric import False_

# Garante que o Python encontre a pasta src
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Importa os seus scripts (Certifique-se de ter renomeado zero-shot para zero_shot)
from src import zero_shot
from src import treino
from src import gerar_predicoes
from src import plots

# Defina como True ou False o que deseja rodar

EXECUTAR_ZERO_SHOT = True       # Roda mBART, NLLB Base e mT5
EXECUTAR_TREINO    = False      # Roda o Fine-Tuning do NLLB (Demorado)
EXECUTAR_PREDICOES = False      # Gera traduções com o modelo treinado (Few-Shot)
EXECUTAR_GRAFICOS  = True       # Gera os gráficos finais a partir dos CSVs

# ==============================================================================
# EXECUÇÃO DO PIPELINE
# ==============================================================================

def main():
    print("===  INICIANDO PIPELINE DE TRADUÇÃO (EP2)  ===\n")
    
    # Caminhos base para facilitar (ajuste se seus scripts usarem outros)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    RESULTS_DIR = os.path.join(BASE_DIR, "results")
    METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")
    PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")

    # ZERO-SHOT (Baselines) 
    if EXECUTAR_ZERO_SHOT:
        print("\nExecutando Experimentos Zero-Shot")
        if hasattr(zero_shot, 'main'):
            zero_shot.main()
        else:
            print(" O script zero_shot.py foi executado na importação.")

    # TREINAMENTO (Fine-Tuning) 
    if EXECUTAR_TREINO:
        print("\nExecutando Fine-Tuning (NLLB)")
        if hasattr(treino, 'treinar_nllb'):
            treino.treinar_nllb()
        elif hasattr(treino, 'main'):
            treino.main()
        else:
            print("Erro ao executar treinamento")

    # PREDIÇÕES (Modelo Treinado) 
    if EXECUTAR_PREDICOES:
        print("\nGerando Predições do Modelo Treinado")
        # Chama a função principal do seu script gerar_predicoes.py
        if hasattr(gerar_predicoes, 'main'):
            gerar_predicoes.main()
        else:
            print("Erro ao tentar executar o script gerar_predicoes.py")

    # GRÁFICOS E ANÁLISE 
    if EXECUTAR_GRAFICOS:
        print("\nGerando Gráficos de Métricas")
        
        # Define onde está o CSV final (gerado pelos passos anteriores)
        # Ajuste este nome se o seu zero-shot ou treino salvarem com outro nome
        arquivo_metricas = os.path.join(METRICS_DIR, "RELATORIO_FINAL_ZEROSHOT.csv") 
        # Se tiver um relatório consolidado geral, aponte para ele aqui.
        
        if os.path.exists(arquivo_metricas):
            if hasattr(plots, 'gerar_graficos'):
                plots.gerar_graficos(arquivo_metricas, PLOTS_DIR)
            else:
                print("Erro: Função 'gerar_graficos' não encontrada em plots.py")
        else:
            print(f"Arquivo de métricas não encontrado em: {arquivo_metricas}")
            print("Verifique se os passos anteriores geraram o CSV corretamente.")

    print("\n=== PROCESSO FINALIZADO ===")

if __name__ == "__main__":
    main()