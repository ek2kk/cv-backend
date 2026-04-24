from search import build_index, search

build_index("data")

request = "Какой линкедин у тебя?"

results = search(request, k=3)

for r in results:
    print(r["score"], r["title"], r["file"])
    print(r["text"][:500])
    print("---")
