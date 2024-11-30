import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch

class LLMModel:
    def __init__(self, model_name='all-MiniLM-L6-v2', batch_size=16):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = SentenceTransformer(model_name).to(self.device)
        self.batch_size = batch_size
        print(f"Initialized model: {model_name} on device: {self.device}")

    def preprocess_data(self, row):
        skills = ', '.join(eval(row['skills'])) if 'skills' in row and row['skills'] else "Unknown"
        return f"{row['job_description']} Skills: {skills}"

    def preprocess_dataset(self, df):
        print("Preprocessing dataset...")
        df['processed_text'] = df.apply(self.preprocess_data, axis=1)
        print("Dataset preprocessing complete.")
        return df

    def batch_extract_features(self, text_list):
        print("Extracting features in batches...")
        features = []
        for i in range(0, len(text_list), self.batch_size):
            batch = text_list[i:i + self.batch_size]
            batch_features = self.model.encode(batch, device=self.device)
            features.extend(batch_features)
        print("Feature extraction complete.")
        return features

    def generate_features(self, df):
        print("Generating features for the dataset...")
        df['features'] = self.batch_extract_features(df['processed_text'].tolist())
        print("Feature generation complete.")
        return df

    def recommend_projects(self, user_query, df, top_n=5):
        print(f"Generating recommendations for query: '{user_query}'")
        user_features = self.model.encode([user_query], device=self.device)[0]
        df['similarity'] = df['features'].apply(lambda x: cosine_similarity([user_features], [x])[0][0])
        recommendations = df.sort_values(by='similarity', ascending=False).head(top_n)
        print("Recommendations generated.")
        return recommendations

    def save_model(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
        self.model.save(directory)
        print(f"Model saved successfully to {directory}")

    @classmethod
    def load_model(cls, directory, batch_size=16):
        instance = cls(batch_size=batch_size)
        instance.model = SentenceTransformer(directory).to(instance.device)
        print(f"Model loaded successfully from {directory}")
        return instance

if __name__ == "__main__":
    llm_model = LLMModel()
    df = pd.read_csv(r'C:\Users\Admin\Desktop\LLM_For_h2h\projects_dataset.csv')
    df_subset = df.copy().head(2500)
    df_subset = llm_model.preprocess_dataset(df_subset)
    df_subset = llm_model.generate_features(df_subset)
    llm_model.save_model('saved_llm_model')
    user_query = "Looking for a project related to data analysis and machine learning."
    recommendations = llm_model.recommend_projects(user_query, df_subset, top_n=5)
    print("Top Recommendations:")
    print(recommendations[['job_description', 'similarity']])