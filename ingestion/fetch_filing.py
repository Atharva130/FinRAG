# ingestion/fetch_filing.py
from edgar import Company, set_identity
import os

# SEC EDGAR requires you to identify yourself (name + email) in every request.
# This is NOT an API key — just a compliance header. Put a real email.
set_identity("Atharva.rahateatharva13@gmail.com")

def fetch_latest_10k(ticker: str, save_dir: str = "data/raw"):
    os.makedirs(save_dir, exist_ok=True)

    company = Company(ticker)
    filings = company.get_filings(form="10-K")
    filing = filings.latest()

    print(f"Found filing: {filing}")
    print(f"Filing date: {filing.filing_date}")
    print(f"Accession number: {filing.accession_no}")

    # Save raw HTML
    html_content = filing.html()
    html_path = os.path.join(save_dir, f"{ticker}_10K.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    # Save plain text version too (edgartools extracts this for us)
    text_content = filing.text()
    text_path = os.path.join(save_dir, f"{ticker}_10K.txt")
    with open(text_path, "w", encoding="utf-8") as f:
        f.write(text_content)

    print(f"Saved HTML to: {html_path}")
    print(f"Saved text to: {text_path}")
    print(f"Text length: {len(text_content)} characters")

if __name__ == "__main__":
    fetch_latest_10k("NVDA")