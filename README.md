# python-distributed-llm

A ideia de criar um sistema distribuído para execução de modelos LLM (como LLaMA ou Phi-3) dividido entre máquinas com poucos recursos, com um gerenciador coordenando os workers, tem várias vantagens importantes, tanto técnicas quanto estratégicas.

✅ Vantagens principais

🧩 1. Aproveitamento de hardware modesto

    Máquinas fracas isoladas não conseguem rodar o modelo inteiro, mas cooperando podem executar partes.

    Reduz a dependência de uma única máquina potente (com muita RAM e CPU/GPU).

💰 2. Custo reduzido

    Evita comprar hardware caro.

    Você pode usar mini-PCs, Raspberry Pi, Orange Pi, VPSs baratos, PCs antigos, etc.

    Usa o que já tem — ótima ideia para home labs ou ambientes educacionais.

🧠 3. Entendimento profundo de como um LLM funciona

    Obriga a entender o pipeline interno de inferência, camadas, tensores, e otimizações.

    Excelente aprendizado para quem quer trabalhar com sistemas distribuídos e LLMs.

🚀 4. Escalabilidade horizontal

    Quando quiser mais performance, adicione mais nós ao cluster.

    Pode fazer sharding mais fino (mais camadas por nó) ou balancear carga por token, etc.

🔄 5. Flexibilidade arquitetural

    Pode:

        Substituir partes do modelo,

        Trocar o backend (PyTorch, GGUF, etc.),

        Integrar cache, prefetch, compressão de tensores, etc.

🧪 6. Testbed para inovação

    Este MVP vira uma base para:

        Explorar novos protocolos (ZeroMQ, gRPC),

        Testar compressão de dados em tempo real,

        Rodar modelos heterogêneos (parte no CPU, parte em GPU),

        Usar quantização sob demanda (ex: ativação em int8, atenção em fp32).

🧰 7. Controle total

    Diferente de frameworks fechados (como DeepSpeed, vLLM), você controla cada etapa.

    Pode customizar tudo: agendamento, balanceamento, cache de atenção, etc.

📡 8. Independência de cloud / open-source por completo

    Pode funcionar offline ou em redes internas (LAN).

    Ideal para aplicações sensíveis (privacidade, edge computing, defesa, indústria).

🛠️ Use cases viáveis

    Chatbots embarcados, com o modelo rodando parcialmente em cada nó.

    Clusters de edge computing, onde cada nó é simples (IoT, SBCs).

    Pesquisas sobre inferência distribuída.

    Educação em sistemas paralelos e IA.


Teste de execução de inferência de modo distribuido

```bash
cd worker1 && uvicorn main:app --host 0.0.0.0 --port 8001
cd worker2 && uvicorn main:app --host 0.0.0.0 --port 8002
cd master && python main.py
```

criado dia 2025-05-05

##
Implementação com safetensors exige a criação de todas as etapas do Transformer (attention, MLP, residuals, etc) (Pausado para estudos).