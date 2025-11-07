
from sentence_transformers import SentenceTransformer, util

class OsbornClassifier:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.osborn_prompts = {
            "Adapt": "change to make it suitable for a new use or purpose",
            "Modify": "change the form or quality of something",
            "Magnify": "make something greater, larger, or stronger",
            "Minify": "make something smaller, lighter, or less",
            "Substitute": "use something new in place of something else",
            "Rearrange": "change the order or layout of components",
            "Reverse": "turn something upside-down or backwards",
            "Combine": "join or merge two or more things into one"
        }
        self.prompt_embeddings = {k: self.model.encode(v) for k, v in self.osborn_prompts.items()}

    def classify_idea(self, idea_text):
        if not idea_text:
            return "Unclassified"
            
        idea_embedding = self.model.encode(idea_text)
        
        max_sim = -1
        best_category = "Unclassified"
        
        for category, prompt_embedding in self.prompt_embeddings.items():
            sim = util.cos_sim(idea_embedding, prompt_embedding).item()
            if sim > max_sim:
                max_sim = sim
                best_category = category
        
        return best_category
