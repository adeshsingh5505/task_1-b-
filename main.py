import os
import json
import pdfplumber
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# -------- Configuration --------
PDF_DIR = "./data"
MODEL_PATH = "./models/all-MiniLM-L6-v2"
OUTPUT_FILE = "output.json"
NUM_TOP_SECTIONS = 5

# -------- Load Persona & Job --------
def load_file_text(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read().strip()

persona = load_file_text(os.path.join(PDF_DIR, "persona.txt"))
job = load_file_text(os.path.join(PDF_DIR, "job.txt"))
persona_job_query = f"{persona}. Task: {job}"

# -------- Load Model --------
model = SentenceTransformer(MODEL_PATH)

# -------- Extract Sections from PDFs --------
def extract_sections(pdf_path):
    sections = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                sections.append({
                    "document": os.path.basename(pdf_path),
                    "page_number": i + 1,
                    "text": text.strip()
                })
    return sections

# -------- Rank Sections --------
def rank_sections(sections, query):
    texts = [s["text"] for s in sections]
    section_embeddings = model.encode(texts)
    query_embedding = model.encode([query])
    scores = cosine_similarity(query_embedding, section_embeddings)[0]
    
    for i, sec in enumerate(sections):
        sec["score"] = scores[i]
    
    return sorted(sections, key=lambda x: x["score"], reverse=True)

# -------- Build Output JSON --------
def build_output(input_files, persona, job, ranked):
    output = {
        "metadata": {
            "input_documents": input_files,
            "persona": persona,
            "job_to_be_done": job,
            "processing_timestamp": datetime.now().isoformat()
        },
        "extracted_sections": [],
        "subsection_analysis": []
    }

    for rank, section in enumerate(ranked[:NUM_TOP_SECTIONS], 1):
        output["extracted_sections"].append({
            "document": section["document"],
            "section_title": section["text"].split("\n")[0][:100],
            "importance_rank": rank,
            "page_number": section["page_number"]
        })

        output["subsection_analysis"].append({
            "document": section["document"],
            "refined_text": section["text"][:1500],  # Customize as needed
            "page_number": section["page_number"]
        })

    return output

# -------- Main Pipeline --------
def main():
    pdf_files = [f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")]
    input_paths = [os.path.join(PDF_DIR, f) for f in pdf_files]
    
    all_sections = []
    for pdf_path in input_paths:
        all_sections.extend(extract_sections(pdf_path))

    ranked_sections = rank_sections(all_sections, persona_job_query)
    output_json = build_output(pdf_files, persona, job, ranked_sections)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output_json, f, indent=4)

    print(f"[✓] JSON saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
