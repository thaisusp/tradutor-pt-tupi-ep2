<<<<<<< HEAD
# Tradução Automática de Baixo Recurso: Português ↔ Tupi Antigo

Repositório oficial do **Exercício Programa 2 (EP2)** da disciplina **MAC0508 - Introdução ao Processamento de Língua Natural (USP)**. Este projeto avalia o desempenho de LLMs (*Large Language Models*) na tradução entre Português e Tupi Antigo, explorando regimes *Zero-Shot* e *Fine-Tuning*.

## Autores
* **Gustavo Ribeiro Bernardo** (NUSP: 14577174)
* **Thaís Martins de Sousa** (NUSP: 14608786)

## Estrutura do Projeto

├── data/
│   ├── processed/
│   │   ├── test.csv                       # Conjunto de teste (Original/Arcaico)
│   │   ├── test_modern.csv               # Conjunto de teste (Português Modernizado)
│   │   └── train.csv                      # Conjunto de treinamento
│   │   └── val.csv
│   │   └── train_modern.csv    
│   │   └── val_modern.csv                        # Conjunto de validação
│   └── raw/
│   │   ├── Cópia de portugues-guarani-tupi antigo.xlsx                      # Tupi Antigo
│   │   └── tupiantigo_portugues_moderno.csv                 # Português Moderno
│                
├── results/
│   ├── RELATORIO_FINAL_ZEROSHOT.csv   # Matriz comparativa consolidada
│   └── analise_qualitativa.csv        # Exemplos de erros e acertos
├── README.md
├── requirements.txt
├── main.py
├── src/
│   ├── zero_shot.py             # Zero-shot: mBART, NLLB, mT5
│   ├── treino.py                # Fine-tuning NLLB (few-shot)
│   ├── gerar_predicoes.py       # Predições + métricas do modelo treinado
│   └── plots.py                 # Geração de gráficos

# Modelos e Estratégias

Modelo	Arquitetura	Estratégia Zero-Shot
mBART-50	Encoder-Decoder	Placeholder pt_XX (Forcing)
NLLB-200	Encoder-Decoder	Proxy grn_Latn (Guarani)
mT5-small	Text-to-Text	Prompt "translate Portuguese to Tupi..."

# Metodologia

Dados: Córpus paralelo Português-Tupi (Rezende, 2025). Inclui versão em Português Moderno (via GPT-4o) para reduzir a perplexidade dos modelos.

# Métricas:

BLEU: Precisão de n-gramas.

chrF3: F-score de caracteres com peso na cobertura (recall), ideal para a morfologia aglutinativa do Tupi.

# Como Executar
Instalação:

Bash
pip install -r requirements.txt
Reprodução:

Execute os notebooks na ordem numérica.

Para o Fine-Tuning, utilize GPU (T4 ou superior).
=======
# tradutor-pt-tupi-ep2
End-to-end NLP pipeline for PT–Old Tupi translation: preprocessing, zero-shot baselines, fine-tuning, metrics and analysis.

## **🚧 This project is still under active development 🚧**
Features, experiments, and documentation will be added soon
>>>>>>> 9a145e143c535279e367763e508a2d6a301b91cc
