from app.services.embeddings import embed_batch


def main():
    sample_texts = ["What is e-KYC?", "This is a test chunk of policy text."]
    embeddings = embed_batch(sample_texts)
    print(f"Got {len(embeddings)} embeddings")
    print(f"Dimension of first embedding: {len(embeddings[0])}")
    print(f"First 5 values: {embeddings[0][:5]}")


if __name__ == "__main__":
    main()