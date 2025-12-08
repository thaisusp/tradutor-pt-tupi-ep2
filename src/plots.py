import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def gerar_graficos(csv_path, output_dir):
    """
    Gera gráficos comparativos a partir do CSV de métricas.
    """
    # Verifica se o arquivo existe
    if not os.path.exists(csv_path):
        print(f"⚠️ Arquivo de métricas não encontrado: {csv_path}")
        return

    print(f"📊 Gerando gráficos a partir de: {csv_path}")
    
    # Carrega os dados
    df = pd.read_csv(csv_path)
    
    # Configuração de estilo do Seaborn
    sns.set_theme(style="whitegrid")
    
    # Define as métricas que queremos plotar
    metricas = ['BLEU', 'chrF3'] # Adicione 'chrF1' se quiser também
    
    # Garante que a pasta de saída existe
    os.makedirs(output_dir, exist_ok=True)

    for metrica in metricas:
        if metrica not in df.columns:
            continue
            
        # Criação da Figura
        plt.figure(figsize=(12, 6))
        
        # Gráfico de Barras Agrupado (Catplot)
        # X = Modelo (mBART, NLLB, etc)
        # Y = Score da métrica
        # Hue = Dados (Antigo vs Moderno)
        # Col = Direção (PT->Tupi vs Tupi->PT)
        
        g = sns.catplot(
            data=df, 
            kind="bar",
            x="Modelo", 
            y=metrica, 
            hue="Dados", 
            col="Direção",
            height=5, 
            aspect=1.2,
            palette="viridis",
            errorbar=None
        )
        
        # Ajustes visuais
        g.despine(left=True)
        g.set_axis_labels("Modelo", f"Pontuação {metrica}")
        g.legend.set_title("Dataset")
        
        # Título superior
        g.fig.subplots_adjust(top=0.85)
        g.fig.suptitle(f'Comparação de {metrica} por Modelo e Direção', fontsize=16)
        
        # Salvar o gráfico
        nome_arquivo = f"grafico_comparativo_{metrica.lower()}.png"
        caminho_salvar = os.path.join(output_dir, nome_arquivo)
        
        plt.savefig(caminho_salvar, dpi=300, bbox_inches='tight')
        plt.close() # Fecha para liberar memória
        
        print(f"   ✅ Gráfico salvo: {caminho_salvar}")

# Bloco para teste individual (se rodar python src/plots.py direto)
if __name__ == "__main__":
    # Caminhos relativos para teste
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_file = os.path.join(base_dir, "results", "metrics", "RELATORIO_FINAL_ZEROSHOT.csv")
    out_folder = os.path.join(base_dir, "results", "plots")
    
    gerar_graficos(csv_file, out_folder)