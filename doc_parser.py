import pymupdf
doc = pymupdf.open("data/Климович Илья 5ПИ Лабораторная 2.pdf")
for page in doc:
  text = page.get_text()
  print(text)
