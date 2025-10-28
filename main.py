from feature_extractor import LanguageFeatureExtractor
from pathlib import Path
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def normalize_features(df: pd.DataFrame, exclude_cols):
    if not exclude_cols:
        exclude_cols = ['language']
    
    if df is None or df.empty:
        if df is None:
            return pd.DataFrame()
        else:
            return df.copy()
    
    if any(col in df.columns for col in exclude_cols):
        metadata = df[exclude_cols]
    else:
        metadata = None
        
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    if not feature_cols:
        return df.copy()
    
    features_numeric = (
        df[feature_cols]
        .apply(pd.to_numeric, errors='coerce')
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )
    
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(features_numeric)
    
    df_normalized = pd.DataFrame(normalized_features, columns=feature_cols, index=df.index)
    
    if metadata is not None:
        for col in exclude_cols:
            if col in df.columns:
                df_normalized[col] = df[col].values
    
    return df_normalized

if __name__ == "__main__":
    extractor = LanguageFeatureExtractor(max_tokens=100000)
    
    language_mapping = {
        'english.conllu': 'English',
        'spanish.conllu': 'Spanish',
        'french.conllu': 'French',
        'hindi.conllu': 'Hindi',
        'arabic.conllu': 'Arabic',
        'chinese.conllu': 'Mandarin',
        'vietnamese.conllu': 'Vietnamese',
        'thai.conllu': 'Thai',
        'german.conllu': 'German',
        'sanskrit.conllu': 'Sanskrit',
        'tamil.conllu': 'Tamil',
        'telugu.conllu': 'Telugu',
        'marathi.conllu': 'Marathi',
        'swedish.conllu': 'Swedish',
        'japanese.conllu': 'Japanese',
        'korean.conllu': 'Korean',
        'turkish.conllu': 'Turkish',
        'uzbek.conllu': 'Uzbek',
        'kazakh.conllu': 'Kazakh',
        'sindhi.conllu': 'Sindhi',
        'malayalam.conllu': 'Malayalam',
        'portugal.conllu': 'Portugal',
        'finnish.conllu': 'Finnish',
        'punjabi.conllu': 'Punjabi',
        "indonesian.conllu": 'Indonesian',
        'romanian.conllu': 'Romanian',
        'russian.conllu': 'Russian'

    }

    data_dir = 'data'
    if not Path(data_dir).exists():
        print(f"Warning: data directory '{data_dir}' not found. Skipping extraction.")
        df = None
    else:
        print("Found directory")
        df = extractor.extract_from_directory(data_dir, language_mapping)

    if df is None or getattr(df, 'empty', True):
        print("No features extracted. Ensure the data directory contains .conllu files.")
        sys.exit(0)
    df.to_csv('language_features_raw.csv', index=False)

    df_normalized = normalize_features(df, exclude_cols=['language'])

    if df_normalized is not None and not df_normalized.empty:
        df_normalized.to_csv('language_features_normalized.csv', index=False)
            
    print(f"\nProcessed {len(df)} languages")
    print(f"Total features: {len(df.columns) - 1}")
    print("\nSample features:")
    print(df)
