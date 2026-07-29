import pytest
from data_pipeline import (
    heuristic_filter,
    exact_dedup_paragraphs,
    exact_dedup_documents,
    fuzzy_dedup,
    scrub_pii,
    quality_filter,
)

def test_heuristic_filter_removes_short():
    text = "Short.\n\nGood long paragraph that has enough words and content to be kept.\n\nTiny."
    filtered = heuristic_filter(text)
    assert "Short." not in filtered
    assert "Tiny." not in filtered
    assert "Good long paragraph" in filtered

def test_heuristic_filter_preserves_prose():
    text = "This is a wonderful piece of prose that is long enough to be preserved. It contains multiple words and is clearly a proper sentence.\n\nAnother very good sentence that should be kept around for our excellent dataset."
    filtered = heuristic_filter(text)
    assert "prose that is long enough" in filtered
    assert "Another very good sentence" in filtered

def test_exact_dedup_paragraphs():
    paras = [
        "This is paragraph one.",
        "This is paragraph two.",
        "This is paragraph one.",
    ]
    deduped = exact_dedup_paragraphs(paras)
    assert len(deduped) == 2
    assert deduped.count("This is paragraph one.") == 1

def test_exact_dedup_documents():
    docs = [
        "Doc A content",
        "Doc B content",
        "Doc A content",
    ]
    deduped = exact_dedup_documents(docs)
    assert len(deduped) == 2
    assert deduped.count("Doc A content") == 1

def test_fuzzy_dedup():
    paras = [
        "This is a relatively long paragraph that is entirely unique and contains specific content about ducks.",
        "This is a relatively long paragraph that is entirely unique and contains specific content about gees.",
        "Completely different text that has nothing to do with birds or any of the above content whatsoever.",
    ]
    deduped = fuzzy_dedup(paras, threshold=0.8)
    assert len(deduped) == 2
    assert "Completely different text" in deduped[1]

def test_scrub_pii_email():
    text = "Contact me at test@example.com for more info."
    scrubbed = scrub_pii(text)
    assert "test@example.com" not in scrubbed
    assert "[EMAIL]" in scrubbed

def test_scrub_pii_phone():
    text = "Call me maybe at 555-123-4567 today."
    scrubbed = scrub_pii(text)
    assert "555-123-4567" not in scrubbed
    assert "[PHONE]" in scrubbed

def test_scrub_pii_ip():
    text = "Server IP is 192.168.1.1 right here."
    scrubbed = scrub_pii(text)
    assert "192.168.1.1" not in scrubbed
    assert "[IP_ADDRESS]" in scrubbed

def test_scrub_pii_credit_card():
    text = "My card is 1234-5678-9012-3456 do not steal."
    scrubbed = scrub_pii(text)
    assert "1234-5678-9012-3456" not in scrubbed
    assert "[CREDIT_CARD]" in scrubbed

def test_quality_filter():
    paras = [
        "Short.",
        "This paragraph is long enough. It has enough words to bypass the short check. The vocabulary is varied and it seems like a normal piece of text. We will see if it works.",
        "Repetitive word word word word word word word word word word word word word word word word word word word word word.",
    ]
    filtered = quality_filter(paras)
    assert len(filtered) == 1
    assert "The vocabulary is varied" in filtered[0]

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
