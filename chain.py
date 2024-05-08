import os
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
# from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# load .env
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
print(OPENAI_API_KEY)

# template = """"당신의 이름은 지니 입니다. 당신은 대한민국의 도로명교육 전문가 입니다. 
# 당신은 초등학생들에게 대한민국의 도로명 체계에 대해 가르쳐야 합니다. 당신은 초등학생들이 이해하기 쉽게 알려줘야 합니다. 
# 초등학생들에게 친근하게 알려주세요. context 내용을 토대로 알려주세요. 
# 학생들의 질문이 도로명에 대한 것이 아니라면, 도로명에 대해 질문하도록 말해주세요. 
# 답변은 최대 300자를 넘기지 말아줘. 모르는 질문이 나오면 모른다고 이야기 해줘. 

### Construct retriever ###
loader = PyPDFLoader("./documents/IntroductionRoadNameAddress.pdf")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = FAISS.from_documents(documents=splits, embedding=OpenAIEmbeddings(api_key=OPENAI_API_KEY))

template = """
Answer the question based on the following context:
{context}

question : {question}
"""

prompt = ChatPromptTemplate.from_template(template)

model = ChatOllama(model="llama3")
# model = ChatOpenAI(api_key=OPENAI_API_KEY)

retriever = vectorstore.as_retriever()


### Compose chain ###
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

print(rag_chain.invoke("안녕"))