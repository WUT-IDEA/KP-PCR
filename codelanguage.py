import Algorithmia
apiKey = '{{Esther}}'
client = Algorithmia.client(apiKey)
input = "/* Simple JavaScript example */\n\n// add two numbers\nfunction add(n, m) {\n  return n + m;\n}"
client = Algorithmia.client('Esther')
algo = client.algo('PetiteProgrammer/ProgrammingLanguageIdentification/0.1.3')
algo.set_options(timeout=300) # optional
print(algo.pipe(input).result)
