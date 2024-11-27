class TextProcessor:
    def __init__(self, nlp):
        self.nlp = nlp

    def summarize_with_spacy(self, text, max_sentences=3):
        if isinstance(text, dict):
            text = str(text)
        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents]

        word_freq = {}
        for token in doc:
            if not token.is_stop and not token.is_punct:
                word_freq[token.text] = word_freq.get(token.text, 0) + 1

        sentence_scores = {}
        for sent in sentences:
            score = sum(word_freq.get(word, 0) for word in sent.split())
            sentence_scores[sent] = score

        summary_sentences = sorted(
            sentence_scores.items(), key=lambda x: x[1], reverse=True
        )[:max_sentences]

        return " ".join(sent[0] for sent in summary_sentences)

    def extract_nouns(self, text):
        if isinstance(text, dict):
            text = str(text)
        doc = self.nlp(text)
        nouns = [token.text for token in doc if token.pos_ in ["NOUN", "PROPN"]]
        return list(set(nouns))
