#  GPU Training Optimization 

### Objetivo

Este repositório apresenta um experimento de fine-tuning no dataset GoEmotions, com foco em otimização do treino e eficiência de dados em uma GPU limitada.

O experimento atual foi feito a partir de um treino semelhante ao do modelo [pinheiro-roberta-goemotions](https://github.com/matheusdatasci/analise-de-sentimentos-nlp), porém injentando três vezes mais 
quantidades de dados do que no primeiro modelo citado, ao passo que permitisse fazer a duração de ambos os treinos ser a mesma utilizando a mesma GPU T4.

### Experimentos

#### Experimento 1 – Baseline

- Dataset: ~50k exemplos
- Épocas: 4
- Learning rate: 2e-5
- Batch size: 16
- Max tokens: 128
- GPU: T4
- Otimizações: nenhuma específica além do AdamW padrão
- Tempo de treino: ~50 minutos

#### Experimento 2 - Treinamento otimizado

- Dataset: ~170k exemplos
- Épocas: 5
- Learning rate: 2e-5
- Batch size: 32
- Max tokens: 64
- GPU: T4
- Otimizações: AdamW fused; torch.compile; AMP (Automatic Mixed Precision)
- Tempo de treino: ~50 minutos

## Observações e Insights

Durante o experimento percebi alguns pontos importantes:

- No primeiro modelo eu estava usando max tokens maiores do que o necessário (128) para frases médias de ~30 caracteres, então reduzir para 64 ajudou a economizar memória e acelerar o treino.

- Usei AdamW fused, que combina operações para reduzir overhead e melhorar desempenho.

- Ativei torch.compile, que otimiza o grafo do PyTorch para execução mais rápida.

- Usei AMP (Automatic Mixed Precision) para aproveitar float16 sem perder precisão, acelerando ainda mais o treinamento.

Com essas mudanças, mesmo com ~3x mais dados, consegui manter o mesmo tempo de treino da primeira versão, mostrando que ajustes de otimização podem trazer ganhos significativos sem precisar de hardware extra.


## Caso queira usar a segunda versão do modelo

```python
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# Carregar tokenizer
tokenizer =  AutoTokenizer.from_pretrained("roberta-base")

# Map de id2label
id2label = {
    0: 'admiration', 1: 'amusement', 2: 'anger', 3: 'annoyance', 4: 'approval',
    5: 'caring', 6: 'confusion', 7: 'curiosity', 8: 'desire', 9: 'disappointment',
    10: 'disapproval', 11: 'disgust', 12: 'embarrassment', 13: 'excitement', 14: 'fear',
    15: 'gratitude', 16: 'grief', 17: 'joy', 18: 'love', 19: 'nervousness', 20: 'optimism',
    21: 'pride', 22: 'realization', 23: 'relief', 24: 'remorse', 25: 'sadness', 26: 'surprise', 27: 'neutral'
}

# Carregar modelo
modelo = AutoModelForSequenceClassification.from_pretrained("pinheiroxs/pinheiro-roberta-goemotions-v2",id2label=id2label)

# Criar pipeline
classifier = pipeline("text-classification", model=modelo, tokenizer=tokenizer, top_k=None)
```
```python
# Exemplo de uso
texto = "I am very happy today!"
resultado = classifier(texto)[0]

# Formatar resultados
resultado_ordenado = sorted(resultado, key=lambda x: x['score'], reverse=True)
for r in resultado_ordenado:
    print(f"{r['label']:15} : {r['score']*100:.2f}%")
```
