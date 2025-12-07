from typing import Dict, List


class CoNLLUReader:
    def read_conllu(self, file_path, max_tokens):
        sentences = []
        current_sentence = []
        token_count = 0
        ended_due_to_limit = False
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                
                if not line or line.startswith('#'):
                    if current_sentence and not line:
                        sentences.append(current_sentence)
                        current_sentence = []
                        if max_tokens is not None and token_count >= max_tokens:
                            ended_due_to_limit = True
                            break
                    continue
                
                fields = line.split('\t')
                if len(fields) >= 10 and '-' not in fields[0] and '.' not in fields[0]:
                    token = {
                        'id': fields[0],
                        'form': fields[1],
                        'lemma': fields[2],
                        'upos': fields[3],
                        'xpos': fields[4],
                        'feats': fields[5],
                        'head': fields[6],
                        'deprel': fields[7],
                        'deps': fields[8],
                        'misc': fields[9]
                    }
                    current_sentence.append(token)
                    token_count += 1
                    if max_tokens is not None and token_count >= max_tokens:
                        sentences.append(current_sentence)
                        ended_due_to_limit = True
                        break
        
        if current_sentence and not ended_due_to_limit:
            sentences.append(current_sentence)
        
        return sentences
    
    def extract_text(self, sentences: List[List[Dict]]):
        text_parts = []
        for sentence in sentences:
            words = []
            for token in sentence:
                words.append(token['form'])
            text_parts.append(' '.join(words))
        return ' '.join(text_parts)