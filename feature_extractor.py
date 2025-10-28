from pathlib import Path
from typing import Dict
import pandas as pd
from conllu_parser import CoNLLUReader
from linguistic_features import AdvancedLinguisticFeatureExtractor

class LanguageFeatureExtractor:
    def __init__(self, max_tokens = 5000):
        self.conllu_reader = CoNLLUReader()
        self.max_tokens = max_tokens
        self.linguistic_extractor = AdvancedLinguisticFeatureExtractor(max_tokens=max_tokens)
    
    def extract_from_file(self, file_path, language_name):
        sentences = self.conllu_reader.read_conllu(file_path, max_tokens=self.max_tokens)
        features = self.linguistic_extractor.extract_from_sentences(sentences, language_name)
        return features
    
    def extract_from_directory(self, directory_path, language_mapping):
        directory = Path(directory_path)
        all_features = []
        
        all_feature_names = set()
        
        for file_path in directory.glob('*.conllu'):
            if language_mapping and file_path.name in language_mapping:
                lang_name = language_mapping[file_path.name]
            else:
                lang_name = file_path.stem
            
            print(f"Processing {file_path.name} ({lang_name})...")
            
            features = self.extract_from_file(str(file_path), lang_name)
            all_features.append(features)
            all_feature_names.update(features.keys())
        
        aligned_features = []
        for features in all_features:
            aligned_feature = {}
            for feature_name in all_feature_names:
                aligned_feature[feature_name] = features.get(feature_name, 0.0)
            aligned_features.append(aligned_feature)
        
        df = pd.DataFrame(aligned_features)
        if 'language' in df.columns:
            cols = ['language'] + [col for col in df.columns if col != 'language']
            df = df[cols]
        
        return df