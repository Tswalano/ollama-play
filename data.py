import chromadb

# Initialize Chroma client
chroma_client = chromadb.Client()

# Get or create a collection called "documents"
collection = chroma_client.get_or_create_collection(name="documents")

# Define the documents to upsert
documents = [
    "What is your return policy? Our return policy allows customers to return items within 30 days of purchase. Items must be in original condition with the receipt. Returns are processed within 7 business days.",
    "How long does shipping take? Standard shipping takes 5-7 business days. Expedited shipping options are available for faster delivery. International shipping times may vary depending on location.",
    "What payment methods do you accept? We accept major credit cards (Visa, MasterCard, American Express), PayPal, and Apple Pay. Payment can be made securely on our website during checkout.",
    "Can I change my order after it has been placed? Once an order is placed, it is processed immediately. However, you may be able to cancel or modify your order within 30 minutes of purchase by contacting customer support.",
    "Do you offer gift wrapping? Yes, we offer gift wrapping for an additional fee. You can select this option during checkout."
]

# Upsert the documents into the collection
collection.upsert(
    documents=documents,
    ids=["id1", "id2", "id3", "id4", "id5"]  # Unique IDs for each document
)

import chromadb

# Initialize Chroma client
chroma_client = chromadb.Client()

# Retrieve your collection
collection = chroma_client.get_collection("documents")

# Get all documents
docs = collection.get()["documents"]

# Print out the documents
for doc in docs:
    print(doc)

