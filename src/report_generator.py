import os
from datetime import datetime


class ReportGenerator:
    def __init__(self, config, logger, image_generator):
        self.config = config
        self.logger = logger
        self.image_generator = image_generator

    async def save_report(self, category, ticker, content):
        output_dir = f"output/{category}"
        posts_dir = f"{output_dir}/posts"
        images_dir = f"{output_dir}/images"
        os.makedirs(posts_dir, exist_ok=True)
        os.makedirs(images_dir, exist_ok=True)

        try:
            image, image_path = self.image_generator.generate_image(
                prompt=content["prompt"], output_dir=images_dir
            )
            content["image_path"] = os.path.relpath(image_path, start=posts_dir)
        except Exception as e:
            self.logger.error("image_generation_failed", error=str(e))
            content["image_path"] = None

        filename = f"{ticker}_{datetime.now().strftime('%Y%m%d%H%M%S')}.md"
        path = os.path.join(posts_dir, filename)

        with open(path, "w") as f:
            f.write(f"# {content['title']}\n\n")
            f.write(f"**Timestamp:** {content['timestamp']}\n\n")

            if content.get("image_path"):
                f.write("## Market Analysis Visualization\n\n")
                f.write(f"![Market Analysis for {ticker}]({content['image_path']})\n\n")

            f.write(f"**Content:**\n\n{content['content']}\n\n")
            f.write("**Sentiment Analysis:**\n\n")
            f.write(f"- Polarity: {content['sentiment']['polarity']:.2f}\n")
            f.write(f"- Confidence: {content['sentiment']['confidence']:.2f}\n\n")
            f.write(f"**Generated Prompt:**\n\n{content['prompt']}\n")
