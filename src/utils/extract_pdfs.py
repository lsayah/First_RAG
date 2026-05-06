# ============================================================================
# 0. IMPORTS & PATH
# ============================================================================

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import PyPDF2
from config import PDF_DIR, DOCUMENTS_DIR


# ============================================================================
# 1. EXTRACTION DES PDFs (PDFs → fichiers texte)
# ============================================================================

def extract_all_pdfs():
    """Extrait tous les PDFs de Sources/ en fichiers .txt dans documents/"""
    
    PDF_DIR.mkdir(exist_ok=True)
    DOCUMENTS_DIR.mkdir(exist_ok=True)
    
    print("\n📄 Extraction PDFs...\n")
    
    success_count = 0
    for pdf_file in sorted(PDF_DIR.glob("*.pdf")):
        print(f"  {pdf_file.name}...", end=" ")
        
        try:
            # Extrait le texte du PDF (TOUTES les pages)
            with open(pdf_file, 'rb') as f:
                pdf = PyPDF2.PdfReader(f)
                text = "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
            
            # Crée le fichier texte
            out_file = DOCUMENTS_DIR / pdf_file.stem
            out_file = out_file.with_suffix('.txt')
            
            with open(out_file, 'w', encoding='utf-8') as f:
                f.write(text)
            
            print(f"✅ ({len(text)//1000}KB)")
            success_count += 1
        except Exception as e:
            print(f"❌ {e}")
    
    print(f"\n✨ {success_count} fichiers créés\n")
    return success_count > 0


if __name__ == "__main__":
    extract_all_pdfs()

