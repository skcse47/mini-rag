from rag.document_loader import load_pdf


pages = load_pdf(
    "data/Employee-Handbook.pdf"
)

for page in pages:

    print("\n====================")
    print(f"PAGE: {page['page']}")
    print("====================")

    print(page["text"])