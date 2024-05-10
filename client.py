from langserve import RemoteRunnable

llama3 = RemoteRunnable("http://0.0.0.0:8000/ask/")

# llama3.invoke({"question": "안녕"})

for token in llama3.stream({"question": "안녕"}):
    print(token, end="")