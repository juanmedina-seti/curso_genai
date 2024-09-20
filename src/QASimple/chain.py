from langchain_core.prompts import  PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv


# Configuración del modelo
llm = ChatGoogleGenerativeAI(model='gemini-pro',temperature=0)

# Plantillas de prompt
prompt_template = """
Eres un asistente muy útil. El usuario te hará preguntas, y debes responderlas con precisión y claridad.
Pregunta: {user_input}
Respuesta:
"""

# Crear la cadena de LLM
prompt = PromptTemplate(template=prompt_template, input_variables=["user_input"])
llm_chain = prompt | llm

def get_response(user_input):
    response = llm_chain.invoke({"user_input": user_input})
    return response.content
def main():
    user_input = ""
    while True:
        user_input=input("Ingrese la pregunta (/q para finalizar)")
        if user_input.startswith("/q"):
            break
        response = llm_chain.invoke({"user_input": user_input})
        print(response)
        print(response.content)


if __name__ == "__main__":
    main()