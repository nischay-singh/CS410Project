import re
import numpy as np
import unicodedata
from collections import Counter
from typing import Dict, List


class AdvancedLinguisticFeatureExtractor:

    def __init__(self, max_tokens: int = 5000):
        self.max_tokens = max_tokens
    
    def is_word(self, token):
        word_pos_tags = {'NOUN', 'VERB', 'ADJ', 'ADV', 'PRON', 'PROPN', 'NUM', 'DET'}
        return token.get('upos') in word_pos_tags
    
    def is_punctuation(self, token):
        return token.get('upos') == 'PUNCT'
    
    def contains_letter(self, text):
        if not text or text == '_':
            return False
        return any(unicodedata.category(char).startswith('L') for char in text)
        
    def extract_from_sentences(self, sentences, language_name):
        
        features = {}
        
        features.update(self.extract_basic_statistics(sentences))
        features.update(self.extract_morphological_features(sentences))
        features.update(self.extract_syntactic_features(sentences))
        features.update(self.extract_word_order_features(sentences))
        features.update(self.extract_lexical_features(sentences))
        features.update(self.extract_dependency_features(sentences))
        features.update(self.extract_character_features(sentences))
        
        if language_name:
            features['language'] = language_name
            
        return features
    
    def extract_basic_statistics(self, sentences):
        all_tokens = []
        all_words = []
        punctuation_count = 0
        
        for sentence in sentences:
            for token in sentence:
                if token['form'] and token['form'] != '_':
                    all_tokens.append(token['form'])
                    
                    if self.is_word(token):
                        all_words.append(token['form'])
                    elif self.is_punctuation(token):
                        punctuation_count += 1
        
        if not all_tokens:
            return {
                'avg_sentence_length': 0.0,
                'std_sentence_length': 0.0,
                'avg_word_length': 0.0,
                'std_word_length': 0.0,
                'word_ratio': 0.0,
                'punctuation_ratio': 0.0,
                'sentence_count': 0,
                'total_tokens': 0
            }
        
        sentence_lengths = [len(sentence) for sentence in sentences if sentence]
        word_lengths = [len(word) for word in all_words]
        
        return {
            'avg_sentence_length': float(np.mean(sentence_lengths)) if sentence_lengths else 0.0,
            'std_sentence_length': float(np.std(sentence_lengths)) if sentence_lengths else 0.0,
            'avg_word_length': float(np.mean(word_lengths)) if word_lengths else 0.0,
            'std_word_length': float(np.std(word_lengths)) if word_lengths else 0.0,
            'word_ratio': len(all_words) / len(all_tokens) if all_tokens else 0.0,
            'punctuation_ratio': punctuation_count / len(all_tokens) if all_tokens else 0.0,
            # 'sentence_count': len(sentences),
            # 'total_tokens': len(all_tokens)
        }
    
    def extract_morphological_features(self, sentences):
        total_words = 0
        total_word_length = 0
        total_char_length = 0
        morph_tags = Counter()
        case_tags = Counter()
        tense_tags = Counter()
        mood_tags = Counter()
        
        for sentence in sentences:
            for token in sentence:
                if self.is_word(token):
                    word = token['form']
                    total_words += 1
                    total_word_length += len(word)
                    
                    total_char_length += len(word)
                    
                    feats = token.get('feats', '')
                    if feats and feats != '_':
                        for feat in feats.split('|'):
                            morph_tags[feat] += 1
                            if feat.startswith('Case='):
                                case_tags[feat.split('=', 1)[1]] += 1
                            if feat.startswith('Tense='):
                                tense_tags[feat.split('=', 1)[1]] += 1
                            if feat.startswith('Mood='):
                                mood_tags[feat.split('=', 1)[1]] += 1
        
        if total_words == 0:
            return {
                'avg_morpheme_length': 0.0,
                'morphological_complexity': 0.0,
                'morph_entropy': 0.0,
                'case_entropy': 0.0,
                'tense_entropy': 0.0,
                'mood_entropy': 0.0
            }
        
        avg_morpheme_length = total_char_length / total_words
        def entropy(counter):
            if not counter:
                return 0.0
            probs = np.array(list(counter.values()), dtype=float)
            probs = probs / probs.sum()
            return float(-np.sum(probs * np.log2(probs + 1e-12)))
        
        morph_entropy = entropy(morph_tags)
        case_entropy = entropy(case_tags)
        tense_entropy = entropy(tense_tags)
        mood_entropy = entropy(mood_tags)
        
        return {
            'avg_morpheme_length': avg_morpheme_length,
            'morphological_complexity': avg_morpheme_length / 5.0,
            'morph_entropy': morph_entropy,
            'case_entropy': case_entropy,
            'tense_entropy': tense_entropy,
            'mood_entropy': mood_entropy
        }
    
    def extract_syntactic_features(self, sentences):
        pos_counts = Counter()
        total_tokens = 0
        
        for sentence in sentences:
            for token in sentence:
                if token['upos'] and token['upos'] != '_':
                    pos_counts[token['upos']] += 1
                    total_tokens += 1
        
        if total_tokens == 0:
            base_features = {f'pos_{tag}': 0.0 for tag in ['NOUN', 'VERB', 'ADJ', 'ADV', 'PRON', 'DET', 'ADP', 'CONJ']}
            base_features.update({
                'noun_verb_ratio': 0.0,
                'adj_noun_ratio': 0.0,
                'syntactic_diversity': 0.0
            })
            return base_features
        
        pos_ratios = {}
        for tag in ['NOUN', 'VERB', 'ADJ', 'ADV', 'PRON', 'DET', 'ADP', 'CONJ']:
            pos_ratios[f'pos_{tag}'] = pos_counts.get(tag, 0) / total_tokens
        
        noun_count = pos_counts.get('NOUN', 0)
        verb_count = max(1, pos_counts.get('VERB', 1))
        
        pos_ratios['noun_verb_ratio'] = noun_count / verb_count
        pos_ratios['adj_noun_ratio'] = pos_counts.get('ADJ', 0) / max(1, noun_count)
        pos_ratios['syntactic_diversity'] = len(pos_counts) / total_tokens
        
        return pos_ratios
    
    def extract_word_order_features(self, sentences):
        svo_patterns = 0
        sov_patterns = 0
        vso_patterns = 0
        total_valid_sentences = 0
        verb_final = 0
        verb_initial = 0
        subj_before_verb = 0
        obj_before_verb = 0
        
        for sentence in sentences:
            if len(sentence) < 3:
                continue
            
            pos_sequence = [token['upos'] for token in sentence 
                          if token['upos'] and token['upos'] != '_']
            
            if len(pos_sequence) < 3:
                continue
            
            has_subj = any(pos in pos_sequence for pos in ['PRON', 'NOUN', 'PROPN'])
            has_verb = 'VERB' in pos_sequence
            has_obj = 'NOUN' in pos_sequence or 'PROPN' in pos_sequence
            
            if not (has_subj and has_verb and has_obj):
                continue
            
            total_valid_sentences += 1
            
            subj_idx = next((i for i, pos in enumerate(pos_sequence) 
                           if pos in ['PRON', 'NOUN', 'PROPN']), -1)
            verb_idx = next((i for i, pos in enumerate(pos_sequence) 
                           if pos == 'VERB'), -1)
            obj_idx = next((i for i, pos in enumerate(pos_sequence[subj_idx+1:], subj_idx+1) 
                          if pos in ['NOUN', 'PROPN']), -1)
            
            if subj_idx >= 0 and verb_idx >= 0 and obj_idx >= 0:
                if subj_idx < verb_idx < obj_idx:
                    svo_patterns += 1
                elif subj_idx < obj_idx < verb_idx:
                    sov_patterns += 1
                elif verb_idx < subj_idx < obj_idx:
                    vso_patterns += 1
                if verb_idx == len(pos_sequence) - 1:
                    verb_final += 1
                if verb_idx == 0:
                    verb_initial += 1
                if subj_idx < verb_idx:
                    subj_before_verb += 1
                if obj_idx < verb_idx:
                    obj_before_verb += 1
        
        if total_valid_sentences == 0:
            return {
                'svo_ratio': 0.0,
                'sov_ratio': 0.0,
                'vso_ratio': 0.0,
                'word_order_diversity': 0.0,
                'verb_final_ratio': 0.0,
                'verb_initial_ratio': 0.0,
                'subj_before_verb_ratio': 0.0,
                'obj_before_verb_ratio': 0.0
            }
        
        return {
            'svo_ratio': svo_patterns / total_valid_sentences,
            'sov_ratio': sov_patterns / total_valid_sentences,
            'vso_ratio': vso_patterns / total_valid_sentences,
            'word_order_diversity': len(set([svo_patterns, sov_patterns, vso_patterns])) / 3.0,
            'verb_final_ratio': verb_final / total_valid_sentences,
            'verb_initial_ratio': verb_initial / total_valid_sentences,
            'subj_before_verb_ratio': subj_before_verb / total_valid_sentences,
            'obj_before_verb_ratio': obj_before_verb / total_valid_sentences
        }
    
    def extract_lexical_features(self, sentences):
        all_words = []
        all_lemmas = []
        
        for sentence in sentences:
            for token in sentence:
                if self.is_word(token):
                    all_words.append(token['form'].lower())
                    if token['lemma'] and token['lemma'] != '_':
                        all_lemmas.append(token['lemma'].lower())
        
        if not all_words:
            return {
                'type_token_ratio': 0.0,
                'lemma_ttr': 0.0,
                'vocabulary_richness': 0.0,
                'avg_word_frequency': 0.0,
                'unique_words': 0,
                'total_words': 0
            }
        
        word_counts = Counter(all_words)
        lemma_counts = Counter(all_lemmas) if all_lemmas else Counter()
        
        ttr = len(word_counts) / len(all_words)
        lemma_ttr = len(lemma_counts) / len(all_lemmas) if all_lemmas else 0.0
        
        hapax_legomena = sum(1 for count in word_counts.values() if count == 1)
        vocabulary_richness = hapax_legomena / len(word_counts) if word_counts else 0.0
        
        avg_word_frequency = float(np.mean(list(word_counts.values()))) if word_counts else 0.0
        
        return {
            'type_token_ratio': ttr,
            'lemma_ttr': lemma_ttr,
            'vocabulary_richness': vocabulary_richness,
            'avg_word_frequency': avg_word_frequency,
            'unique_words': len(word_counts),
            'total_words': len(all_words)
        }
    
    def extract_dependency_features(self, sentences):
        dependency_lengths = []
        dependency_types = Counter()
        total_dependencies = 0
        left_arcs = 0
        right_arcs = 0
        left_verbs = 0
        right_verbs = 0
        left_nouns = 0
        right_nouns = 0
        left_adp = 0
        right_adp = 0
        amod_left = 0
        amod_right = 0
        nmod_left = 0
        nmod_right = 0
        
        for sentence in sentences:
            for token in sentence:
                if token['head'] and token['head'] != '_' and token['deprel'] and token['deprel'] != '_':
                    head_idx = int(token['head'])
                    token_idx = int(token['id'])
                    dependency_length = abs(head_idx - token_idx)
                    dependency_lengths.append(dependency_length)
                    dependency_types[token['deprel']] += 1
                    total_dependencies += 1
                    if token_idx < head_idx:
                        left_arcs += 1
                    elif token_idx > head_idx:
                        right_arcs += 1
                    upos = token.get('upos')
                    if upos == 'VERB':
                        if token_idx < head_idx:
                            left_verbs += 1
                        elif token_idx > head_idx:
                            right_verbs += 1
                    if upos == 'NOUN':
                        if token_idx < head_idx:
                            left_nouns += 1
                        elif token_idx > head_idx:
                            right_nouns += 1
                    if upos == 'ADP':
                        if token_idx < head_idx:
                            left_adp += 1
                        elif token_idx > head_idx:
                            right_adp += 1
                    if token['deprel'] == 'amod':
                        if token_idx < head_idx:
                            amod_left += 1
                        elif token_idx > head_idx:
                            amod_right += 1
                    if token['deprel'] in ('nmod', 'obl'):
                        if token_idx < head_idx:
                            nmod_left += 1
                        elif token_idx > head_idx:
                            nmod_right += 1
        
        if not dependency_lengths:
            return {
                'avg_dependency_length': 0.0,
                'std_dependency_length': 0.0,
                'dependency_diversity': 0.0,
                'long_dependency_ratio': 0.0,
                'dep_left_ratio': 0.0,
                'dep_right_ratio': 0.0,
                'dep_left_ratio_verbs': 0.0,
                'dep_left_ratio_nouns': 0.0,
                'dep_left_ratio_adp': 0.0,
                'amod_left_ratio': 0.0,
                'nmod_left_ratio': 0.0,
                'dep_len_p50': 0.0,
                'dep_len_p90': 0.0
            }
        
        avg_dep_length = float(np.mean(dependency_lengths))
        std_dep_length = float(np.std(dependency_lengths))
        dependency_diversity = len(dependency_types) / total_dependencies if total_dependencies > 0 else 0.0
        long_dependency_ratio = sum(1 for length in dependency_lengths if length > 3) / len(dependency_lengths)
        dep_left_ratio = left_arcs / total_dependencies if total_dependencies else 0.0
        dep_right_ratio = right_arcs / total_dependencies if total_dependencies else 0.0
        dep_left_ratio_verbs = left_verbs / max(1, left_verbs + right_verbs)
        dep_left_ratio_nouns = left_nouns / max(1, left_nouns + right_nouns)
        dep_left_ratio_adp = left_adp / max(1, left_adp + right_adp)
        amod_left_ratio = amod_left / max(1, amod_left + amod_right)
        nmod_left_ratio = nmod_left / max(1, nmod_left + nmod_right)
        dep_len_p50 = float(np.percentile(dependency_lengths, 50))
        dep_len_p90 = float(np.percentile(dependency_lengths, 90))
        
        return {
            'avg_dependency_length': avg_dep_length,
            'std_dependency_length': std_dep_length,
            'dependency_diversity': dependency_diversity,
            'long_dependency_ratio': long_dependency_ratio,
            'dep_left_ratio': dep_left_ratio,
            'dep_right_ratio': dep_right_ratio,
            'dep_left_ratio_verbs': dep_left_ratio_verbs,
            'dep_left_ratio_nouns': dep_left_ratio_nouns,
            'dep_left_ratio_adp': dep_left_ratio_adp,
            'amod_left_ratio': amod_left_ratio,
            'nmod_left_ratio': nmod_left_ratio,
            'dep_len_p50': dep_len_p50,
            'dep_len_p90': dep_len_p90,
        }
    
    def extract_character_features(self, sentences):
        raw_text = ' '.join(
            token['form']
            for sentence in sentences
            for token in sentence
            if token['form'] and token['form'] != '_'
        )
        if not raw_text:
            return {
                'unique_char_ratio': 0.0,
                'space_ratio': 0.0,
                'digit_ratio': 0.0,
                'character_diversity': 0.0,
                'letter_ratio': 0.0,
            }
        text = unicodedata.normalize('NFKC', raw_text)
        total_chars = len(text)
        spaces = text.count(' ')
        digits = sum(1 for c in text if c.isdigit())
        letters = [c for c in text if unicodedata.category(c).startswith('L')]
        unique_letters = len(set(letters))
        letter_count = len(letters)
        letter_ratio = letter_count / total_chars if total_chars else 0.0
        unique_letter_ratio = unique_letters / letter_count if letter_count else 0.0
        return {
            'unique_char_ratio': unique_letter_ratio,
            'space_ratio': spaces / total_chars,
            'digit_ratio': digits / total_chars,
            'character_diversity': unique_letter_ratio,
            'letter_ratio': letter_ratio,
        }