# ============================================================================
# 0. IMPORT & CONFIGURATION
# ============================================================================

import requests
from pathlib import Path
from bs4 import BeautifulSoup
from config import DOCUMENTS_DIR

# ============================================================================
# 1. MiFID II
# ============================================================================

def fetch_mifid_ii():
    url = "https://eur-lex.europa.eu/legal-content/FR/TXT/?uri=CELEX:32014L0065"
    try:
        response = requests.get(url, timeout=10)
        response.encoding = 'utf-8'
        soup = BeautifulSoup(response.text, 'html.parser')
        
        text_content = soup.find('div', {'class': 'oj-doc-body'})
        content = text_content.get_text(separator='\n') if text_content else soup.get_text(separator='\n')
        
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        content = '\n'.join(lines[:2000])
        
        mifid_content = f"""[SOURCE: EUR-Lex]
[URL: https://eur-lex.europa.eu/legal-content/FR/TXT/?uri=CELEX:32014L0065]
[DOCUMENT: Directive MiFID II (2014/65/UE)]

{content}"""
        
        with open(DOCUMENTS_DIR / "mifid_ii.txt", "w", encoding='utf-8') as f:
            f.write(mifid_content)
        return True
    except Exception as e:
        print(f"❌ MiFID II: {e}")
        return False


# ============================================================================
# 2. Bâle III
# ============================================================================

def fetch_basel_iii():
    url = "https://www.banque-france.fr/stabilite-financiere/prudence/normes-prudentielles"
    try:
        response = requests.get(url, timeout=10)
        response.encoding = 'utf-8'
        soup = BeautifulSoup(response.text, 'html.parser')
        
        content_div = soup.find('main')
        content = content_div.get_text(separator='\n') if content_div else soup.get_text(separator='\n')
        
        lines = [line.strip() for line in content.split('\n') if line.strip() and len(line) > 10]
        content = '\n'.join(lines[:1500])
        
        basel_content = f"""[SOURCE: Banque de France]
[URL: https://www.banque-france.fr/stabilite-financiere/prudence/normes-prudentielles]
[DOCUMENT: Normes Prudentielles Bâle III]

{content}"""
        
        with open(DOCUMENTS_DIR / "basel_iii.txt", "w", encoding='utf-8') as f:
            f.write(basel_content)
        return True
    except Exception as e:
        print(f"❌ Bâle III: {e}")
        return False


# ============================================================================
# 3. AMF Guide
# ============================================================================

def fetch_amf_guide():
    url = "https://www.amf-france.org/sites/default/files/private/2024-01/GUIDE%20DE%20L_INVESTISSEUR_2024.pdf"
    try:
        response = requests.get(url, timeout=10)
        try:
            import PyPDF2
            import io
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(response.content))
            content = "\n".join(page.extract_text() for page in pdf_reader.pages[:10])
        except:
            content = "[PDF not extracted - consult official source]"
        
        amf_content = f"""[SOURCE: AMF (Autorité des Marchés Financiers)]
[URL: https://www.amf-france.org/]
[DOCUMENT: Guide de l'Investisseur 2024]

{content}"""
        
        with open(DOCUMENTS_DIR / "amf_guide.txt", "w", encoding='utf-8') as f:
            f.write(amf_content)
        return True
    except Exception as e:
        print(f"❌ AMF: {e}")
        return False

