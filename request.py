import requests

rsp = requests.get("http://localhost:8000")
print("rsppppp content bytes: ",rsp.content)

