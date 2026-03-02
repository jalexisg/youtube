import re
import os
import json
from typing import List, Dict, Optional
from collections import Counter

# NLTK will be initialized via a helper if needed, but we try to import it globally
try:
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.corpus import stopwords
except ImportError:
    nltk = None
    sent_tokenize = None
    word_tokenize = None
    stopwords = None

def setup_nltk():
    """Configura las dependencias de NLTK"""
    if nltk is None:
        return
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt_tab', quiet=True)
    except Exception as e:
        print(f"⚠️  Advertencia: Error configurando NLTK: {e}")

def clean_text(text: str) -> str:
    """Limpia y normaliza el texto transcrito"""
    # Eliminar espacios extra
    text = re.sub(r'\s+', ' ', text.strip())
    # Corregir puntuación común
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)
    text = re.sub(r'([.,!?;:])\s*([a-zA-Z])', r'\1 \2', text)
    return text

def extract_keywords(text: str, num_keywords: int = 10) -> List[str]:
    """Extrae palabras clave del texto"""
    if not word_tokenize:
        return []
    try:
        words = word_tokenize(text.lower())
        try:
            stop_words_es = set(stopwords.words('spanish'))
            stop_words_en = set(stopwords.words('english'))
            stop_words = stop_words_es.union(stop_words_en)
        except:
            stop_words = set()
        
        filtered_words = [
            word for word in words 
            if word.isalpha() and len(word) > 3 and word not in stop_words
        ]
        word_freq = Counter(filtered_words)
        return [word for word, freq in word_freq.most_common(num_keywords)]
    except Exception as e:
        print(f"⚠️  Error extrayendo palabras clave: {e}")
        return []

def create_extractive_summary(text: str, num_sentences: int = 5) -> str:
    """Crea un resumen extractivo basado en las oraciones más importantes"""
    if not sent_tokenize:
        return text[:1000]
    try:
        sentences = sent_tokenize(text)
        if len(sentences) <= num_sentences:
            return text
        
        words = word_tokenize(text.lower())
        try:
            stop_words_es = set(stopwords.words('spanish'))
            stop_words_en = set(stopwords.words('english'))
            stop_words = stop_words_es.union(stop_words_en)
        except:
            stop_words = set()
        
        filtered_words = [word for word in words if word.isalpha() and word not in stop_words]
        word_freq = Counter(filtered_words)
        
        sentence_scores = {}
        for sentence in sentences:
            sentence_words = word_tokenize(sentence.lower())
            score = sum(word_freq[word] for word in sentence_words if word in word_freq)
            word_count = sum(1 for word in sentence_words if word.isalpha())
            if word_count > 0:
                sentence_scores[sentence] = score / word_count
        
        best_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)
        summary_sentences = [sent for sent, score in best_sentences[:num_sentences]]
        
        ordered_summary = [sent for sent in sentences if sent in summary_sentences]
        return ' '.join(ordered_summary)
    except Exception as e:
        print(f"⚠️  Error creando resumen extractivo: {e}")
        return text.split('. ')[0] + '.'

def create_topic_summary(text: str) -> Dict[str, str]:
    """Crea un resumen organizado por temas principales"""
    try:
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        if not paragraphs:
            sentences = sent_tokenize(text) if sent_tokenize else text.split('. ')
            paragraphs = [' '.join(sentences[i:i+4]) for i in range(0, len(sentences), 4)]
        
        topics = {}
        for i, paragraph in enumerate(paragraphs[:5]):
            keywords = extract_keywords(paragraph, num_keywords=3)
            topic_name = f"Tema {i+1}: {', '.join(keywords[:2])}" if keywords else f"Tema {i+1}"
            topics[topic_name] = create_extractive_summary(paragraph, num_sentences=2)
        return topics
    except Exception as e:
        print(f"⚠️  Error creando resumen por temas: {e}")
        return {"Resumen general": create_extractive_summary(text)}

def generate_social_descriptions(text: str, model: str = "mistralai/Mistral-7B-Instruct-v0.2", hf_token: Optional[str] = None) -> Dict[str, str]:
    """Genera 3 opciones de descripción para Reels/Shorts usando Hugging Face."""
    if not hf_token:
        hf_token = os.environ.get("HF_TOKEN")
    
    if not hf_token:
        return {}
            
    try:
        from openai import OpenAI
        client = OpenAI(base_url="https://router.huggingface.co/v1", api_key=hf_token)
        
        prompt = f"""
        Basado en la siguiente transcripción de un video, genera exactamente 3 opciones de texto muy breve (máximo 2 líneas por opción) para usar como presentación de un Reel o Short.
        
        REGLAS CRÍTICAS:
        1. NO uses emojis de ningún tipo.
        2. NO uses asteriscos (*) ni formatos de negrita/cursiva. Solo texto plano.
        3. Sé extremadamente conciso. Máximo 2 líneas por opción.
        4. El tono debe ser DESCRIPTIVO e INTRODUCTORIO, no puramente afirmativo.
        5. El texto debe estar listo para copiar y pegar directamente.
        
        Transcripción: "{text[:2000]}"
        
        Sigue este formato exacto en español:
        Opción 1 (Filosofía): [Texto]
        Opción 2 (Lección): [Texto]
        Opción 3 (Aprendizaje): [Texto]
        """
        
        response_obj = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=600
        )
        response = response_obj.choices[0].message.content or ""
        
        options = {}
        # Simple parsing
        for key, pattern in [("filosofia", "1"), ("leccion", "2"), ("aprendizaje", "3")]:
            match = re.search(fr'Opci[óo]n {pattern}.*?\:(.*?)(?=Opci[óo]n|$)', response, re.S | re.I)
            if match:
                options[key] = match.group(1).strip()
        
        return options
    except Exception as e:
        print(f"⚠️ Error generando descripciones sociales: {e}")
        return {}
