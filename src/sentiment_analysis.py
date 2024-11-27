from textblob import TextBlob


class SentimentAnalyzer:
    def __init__(self, nlp):
        self.nlp = nlp

    async def analyze_sentiment(self, texts):
        combined_text = " ".join(texts)
        blob = TextBlob(combined_text)
        doc = self.nlp(combined_text)

        entity_sentiments = {}
        for ent in doc.ents:
            entity_blob = TextBlob(ent.text)
            entity_sentiments[ent.text] = entity_blob.sentiment.polarity

        return {
            "polarity": blob.sentiment.polarity,
            "subjectivity": blob.sentiment.subjectivity,
            "entities": entity_sentiments,
            "confidence": 1 - blob.sentiment.subjectivity,
        }
