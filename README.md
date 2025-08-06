# python-distributed-llm

Python Distributed LLM
📌 Visão Geral

Este projeto é um estudo sobre execução distribuída de modelos de linguagem usando Python e PyTorch, com foco em rodar modelos grandes em máquinas com pouca memória RAM.

A ideia principal é dividir as camadas do modelo em múltiplos workers, que podem estar em processos ou máquinas diferentes, e processar sequencialmente as partes do modelo via rede (TCP/IP).

Foram realizados testes bem-sucedidos com os modelos:

   - microsoft/Phi-3-mini-instruct

   - microsoft/Phi-3.5-mini-instruct

---

🏗 Arquitetura

O projeto segue um modelo Manager → Workers:

#### Manager (Master)

- Recebe o *prompt* de entrada.

- Tokeniza usando `AutoTokenizer`.

- Orquestra a execução chamando cada worker na sequência correta.

- Não carrega nenhuma camada do modelo na memória (somente embeddings, config e tokenizer).

#### Workers

- Cada worker carrega **apenas um subconjunto das camadas** do modelo.

- Executam o *forward* local das camadas que possuem.

- Retornam os *hidden states* processados para o próximo worker via rede.

📡 Comunicação

   - A comunicação é feita via **sockets TCP/IP**.

   - Serialização usando `pickle`.

   - O protocolo inclui prefixo de tamanho (4 bytes) para controle do fluxo de dados.

---

🎯 Motivação

Modelos de linguagem grandes normalmente precisam de **vários GB de RAM** para rodar.
Este projeto demonstra que é possível distribuir a execução em múltiplos nós modestos (como **Orange Pi 4GB**), processando partes do modelo de forma sequencial.

Isso permite:

   - Usar hardware barato para estudar modelos grandes.

   - Executar localmente modelos que não caberiam em um único dispositivo.


📂 Estrutura do Projeto

```bash
src/
├── gpt2/                     # Versão experimental para GPT-2
│   ├── master/                # Scripts do manager
│   ├── worker1/               # Scripts do primeiro worker
│   └── worker2/               # Scripts do segundo worker
├── phi3/                      # Versão para modelos Phi-3 / Phi-3.5
│   ├── manager.py             # Manager (master)
│   ├── partial_forward_client.py
│   ├── partial_forward_server.py
│   ├── phi3_partial/          # Implementação de forward parcial no modelo
│   ├── worker_1.py            # Primeiro worker
│   ├── worker_2.py            # Segundo worker
│   └── test/                  # Testes e exemplos
```

🚀 Como Executar

1️⃣ **Instalar dependências**
```bash
poetry install
```

2️⃣ **Iniciar os Workers**

Em terminais separados, rode:
```bash
poetry run python src/phi3/worker_1.py
poetry run python src/phi3/worker_2.py
```

3️⃣ **Iniciar o Manager**
```bash
poetry run python src/phi3/manager.py
```

🔧 **Configuração**

Modelo: definido em `MODEL_PATH` no código.

Workers: definidos na lista `WORKERS` no `manager.py`.

Portas: ajustadas diretamente nos scripts dos workers.

Divisão de camadas: controlada via parâmetros `handle_section_index` e `total_sections` no carregamento `from_pretrained_partial`.

📜 **Licença**

Este projeto é apenas para estudo e uso pessoal.
Licença: MIT
