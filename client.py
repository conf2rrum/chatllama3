from langserve import RemoteRunnable

llm = RemoteRunnable("http://localhost:8000/ask/")

print(llm.invoke("안녕"))
