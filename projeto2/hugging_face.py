from transformers.pipelines import pipeline

chatbot = pipeline("text2text-generation", model="google/flan-t5-small")

print("Chatbot IA (Digite 'sair' para encerrar)")

while True:
    pergunta = input("Você: ")
    if pergunta.lower() == "sair":
        break

    prompt = f"Answer the question clearly and precisely: {pergunta}"

    resposta = chatbot(prompt, max_new_tokens=150, do_sample=False, temperature=0.3)
    print("Bot:", resposta[0]['generated_text']) #type: ignore

# o chat-gpt precisa de uma conta paga parafuncionar e, infelizmente, o hugging face não apresenta um bomo desempenho para respostas
