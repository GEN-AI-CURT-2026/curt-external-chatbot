import os
import time
import hashlib
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

DATA_DIR        = Path(__file__).parent.parent / "data"
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIM   = 3072         
CHUNK_SIZE      = 700
CHUNK_OVERLAP   = 100
INDEX_NAME      = os.getenv("PINECONE_INDEX_NAME", "external-chatbot")
NAMESPACE       = "curt_docs"  
BATCH_SIZE      = 100      


class PineconeBuilder:
    def __init__(self):
        self._validate_env()

        self.embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
        )

        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    def _validate_env(self):
        required = ["OPENAI_API_KEY", "PINECONE_API_KEY", "PINECONE_INDEX_NAME"]
        missing = [k for k in required if not os.getenv(k)]
        if missing:
            raise EnvironmentError(f"Missing env vars: {', '.join(missing)}")

    def _get_or_create_index(self):
        existing = [idx.name for idx in self.pc.list_indexes()]

        if INDEX_NAME not in existing:
            print(f"  Creating new index '{INDEX_NAME}' (dim={EMBEDDING_DIM})...")
            self.pc.create_index(
                name=INDEX_NAME,
                dimension=EMBEDDING_DIM,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud=os.getenv("PINECONE_CLOUD", "aws"),
                    region=os.getenv("PINECONE_ENVIRONMENT", "us-east-1"),
                ),
            )

            while not self.pc.describe_index(INDEX_NAME).status["ready"]:
                print("  Waiting for index to become ready...")
                time.sleep(2)
            print(f"  ✓ Index '{INDEX_NAME}' created\n")
        else:
            print(f"  ✓ Index '{INDEX_NAME}' already exists\n")

        return self.pc.Index(INDEX_NAME)

    # ── Document loading ────────────────────────────────────────────────────
    def load_documents(self):
        print("Step 1: Loading documents...")
        docs = []
        
        for glob, label in [("**/*.txt", "text"), ("**/*.md", "markdown")]:
            try:
                loader = DirectoryLoader(
                    str(DATA_DIR),
                    glob=glob,
                    loader_cls=TextLoader,
                    loader_kwargs={"encoding": "utf-8"},
                )
                loaded = loader.load()
                docs.extend(loaded)
                print(f"  Loaded {len(loaded)} {label} files")
            except Exception as e:
                print(f"  Note ({label}): {e}")

        if not docs:
            raise FileNotFoundError("No .md or .txt files found in /data/")

        total_chars = sum(len(d.page_content) for d in docs)
        print(f"\n  Total documents : {len(docs)}")
        print(f"  Total characters: {total_chars:,}\n")
        return docs

    # ── Chunking ────────────────────────────────────────────────────────────
    def chunk_documents(self, documents):
        print("Step 2: Chunking documents...")
        chunks = self.text_splitter.split_documents(documents)

        counts: dict[str, int] = {}
        for c in chunks:
            src = Path(c.metadata.get("source", "unknown")).name
            counts[src] = counts.get(src, 0) + 1

        print(f"  ✓ {len(chunks)} chunks created")
        for src, n in sorted(counts.items()):
            print(f"      {src}: {n} chunks")
        print()
        return chunks

    # ── Upsert to Pinecone ──────────────────────────────────────────────────
    def upsert_chunks(self, index, chunks):
        """Embed chunks in batches and upsert to Pinecone."""
        print("Step 3: Embedding & upserting to Pinecone...")

        total = len(chunks)
        upserted = 0

        for start in range(0, total, BATCH_SIZE):
            batch = chunks[start : start + BATCH_SIZE]
            texts = [c.page_content for c in batch]

            # Embed the batch
            vectors = self.embeddings.embed_documents(texts)

            # Build Pinecone records using tuple syntax (safe & explicit)
            records = []
            for i, (chunk, vec) in enumerate(zip(batch, vectors)):
                source_name = Path(chunk.metadata.get("source", "unknown")).name
                
                # FIX: Create a unique, deterministic ID to prevent collision updates
                text_hash = hashlib.md5(chunk.page_content.encode('utf-8')).hexdigest()[:12]
                chunk_id = f"{source_name.replace('.', '_')}_{start + i}_{text_hash}"
                
                metadata = {
                    "text": chunk.page_content,
                    "source": source_name,
                    "chunk_index": start + i,
                }
                # Structured as: (id, vector_values, metadata)
                records.append((chunk_id, vec, metadata))

            index.upsert(vectors=records, namespace=NAMESPACE)
            upserted += len(records)
            print(f"  Upserted {upserted}/{total} chunks", end="\r")

        print(f"\n  ✓ All {upserted} chunks stored in namespace '{NAMESPACE}'\n")

    # ── Smoke test ──────────────────────────────────────────────────────────
    def verify(self, index):
        print("Step 4: Verifying index...")
        stats = index.describe_index_stats()
        
        # FIX: Safe dictionary traversal across SDK versions
        ns_data = stats.namespaces.get(NAMESPACE, {})
        vector_count = ns_data.get("vector_count", 0) if isinstance(ns_data, dict) else getattr(ns_data, "vector_count", 0)
        print(f"  Vectors in namespace '{NAMESPACE}': {vector_count}")

        test_vec = self.embeddings.embed_query("What is CURT?")
        results = index.query(
            vector=test_vec,
            top_k=3,
            namespace=NAMESPACE,
            include_metadata=True,
        )
        print(f"  Similarity search returned {len(results.matches)} matches")
        if results.matches:
            top = results.matches[0]
            print(f"  Top match  — score: {top.score:.4f} | source: {top.metadata.get('source')}")
            print(f"  Preview    — {top.metadata.get('text', '')[:120]}...\n")

    # ── Main ────────────────────────────────────────────────────────────────
    def build(self):

        index  = self._get_or_create_index()
        docs   = self.load_documents()
        chunks = self.chunk_documents(docs)
        self.upsert_chunks(index, chunks)
        self.verify(index)
        print(f"  Index      : {INDEX_NAME}")
        print(f"  Namespace  : {NAMESPACE}")
        print(f"  Chunks     : {len(chunks)}")
        print(f"  Embed model: {EMBEDDING_MODEL}")
        print(f"  Dimensions : {EMBEDDING_DIM}\n")


def main():
    DATA_DIR.mkdir(exist_ok=True)
    builder = PineconeBuilder()
    builder.build()


if __name__ == "__main__":
    main()