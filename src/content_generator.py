from cgitb import text
from datetime import datetime
from openai import OpenAI


class ContentGenerator:
    def __init__(self, config, openai_client, text_processor):
        self.config = config
        self.openai_client = openai_client
        self.text_processor = text_processor

    def generate_prompt(self, content):
        summary = self.text_processor.summarize_with_spacy(content, max_sentences=3)
        nouns = self.text_processor.extract_nouns(summary)
        key_phrases = ", ".join(nouns[:7])

        prompt = f"""
        Create a visual scene incorporating these financial concepts:
        
        Market Context: {summary}
        Key Concepts: {key_phrases}
        Elements: {', '.join(nouns)}

        Describe a striking visual composition that:
        1. Features dramatic lighting, perspective, and depth
        2. Incorporates financial symbols and market indicators
        3. Uses professional color schemes and textures
        4. Creates a powerful mood and atmosphere
        
        Provide only the visual description, focused on artistic elements.
        """

        response = self.openai_client.chat.completions.create(
            model="llama-3.2-3b-preview",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert art director specializing in financial visualization.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=80,
            temperature=0.5,
        )
        return response.choices[0].message.content.strip()

    async def generate_content(self, ticker_data, sentiment, news):
        prompt = f"""
        Write a detailed analysis of {ticker_data['info']['shortName']}:
        
        Market Data:
        - Price: ${ticker_data['price']:.2f}
        - Change: {ticker_data['change']:.2%}
        
        Sentiment:
        - Overall: {sentiment['polarity']:.2f}
        - Confidence: {sentiment['confidence']:.2f}
        
        Recent Headlines:
        {' | '.join(article['title'] for article in news[:3])}
        
        Format in Markdown with technical analysis and outlook.
        """

        response = self.openai_client.chat.completions.create(
            model="llama-3.2-3b-preview",
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional financial analyst.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=2000,
            temperature=0,
        )

        content = response.choices[0].message.content
        return {
            "content": content,
            "title": f"Analysis: {ticker_data['info']['shortName']}",
            "timestamp": datetime.now().isoformat(),
            "sentiment": sentiment,
            "prompt": self.generate_prompt(content),
        }
