import re
import string
import pandas as pd
from collections import Counter

class IndonesianTextPreprocessor:
    """Preprocessing khusus untuk teks bahasa Indonesia dengan normalisasi ejaan"""
    
    def __init__(self):
        # Stopwords bahasa Indonesia yang umum
        self.stopwords = {
            'dan', 'atau', 'yang', 'di', 'ke', 'dari', 'pada', 'dengan', 'untuk', 
            'dalam', 'adalah', 'akan', 'ada', 'juga', 'tidak', 'ini', 'itu', 
            'atau', 'saja', 'hanya', 'bisa', 'dapat', 'sudah', 'telah', 'sedang',
            'kemudian', 'lalu', 'setelah', 'sambil', 'hingga', 'sampai', 'agar',
            'supaya', 'karena', 'sebab', 'oleh', 'secara', 'seperti', 'ibarat'
        }
        
        # Kata-kata umum dalam resep yang bisa dihapus
        self.recipe_stopwords = {
            'buah', 'lembar', 'siung', 'butir', 'biji', 'potong', 'iris', 'cincang',
            'parut', 'haluskan', 'aduk', 'masukkan', 'tambahkan', 'tuang', 'panaskan',
            'goreng', 'rebus', 'kukus', 'bakar', 'panggang', 'secukupnya', 'seperlunya', 'aduk2'
        }
        
        # Dictionary untuk normalisasi ejaan yang salah/tidak baku
        self.spelling_normalization = {
            # Telur variations
            'telor': 'telur',
            'telor ayam': 'telur ayam',
            'telor bebek': 'telur bebek',
            'telor puyuh': 'telur puyuh',
            'telor dadar': 'telur dadar',
            'telor mata sapi': 'telur mata sapi',
            
            # Tempe/Tempeh
            'tempe': 'tempeh',
            'tempe goreng': 'tempeh goreng',
            'tempe bacem': 'tempeh bacem',
            
            # Cabai/Cabe variations
            'cabe': 'cabai',
            'cabe rawit': 'cabai rawit',
            'cabe merah': 'cabai merah',
            'cabe hijau': 'cabai hijau',
            'cabe keriting': 'cabai keriting',
            'lombok': 'cabai',
            'lombok rawit': 'cabai rawit',
            'lombok merah': 'cabai merah',
            'lombok hijau': 'cabai hijau',
            
            # Tomat variations
            'tomat': 'tomat',  # sudah benar tapi untuk konsistensi
            'tamad': 'tomat',
            'tomad': 'tomat',
            
            # Bawang variations
            'bamer': 'bawang merah',
            'baput': 'bawang putih',
            'bawang bombai': 'bawang bombay',
            'bawang bombay': 'bawang bombay',
            
            # Daging variations
            'daging sapi': 'daging sapi',
            'daging ayam': 'daging ayam',
            'daging kambing': 'daging kambing',
            'daging domba': 'daging domba',
            
            # Ayam variations
            'ayam kampung': 'ayam kampung',
            'ayam broiler': 'ayam broiler',
            'ayam potong': 'ayam potong',
            
            # Ikan variations
            'ikan bandeng': 'ikan bandeng',
            'ikan lele': 'ikan lele',
            'ikan nila': 'ikan nila',
            'ikan patin': 'ikan patin',
            'ikan salmon': 'ikan salmon',
            'ikan tuna': 'ikan tuna',
            'ikan kakap': 'ikan kakap',
            'ikan kembung': 'ikan kembung',
            'ikan tongkol': 'ikan tongkol',
            'ikan teri': 'ikan teri',
            'ikan asin': 'ikan asin',
            
            # Sayuran variations
            'kangkung': 'kangkung',
            'bayam': 'bayam',
            'sawi': 'sawi',
            'selada': 'selada',
            'kubis': 'kubis',
            'kol': 'kubis',
            'wortel': 'wortel',
            'kentang': 'kentang',
            'ubi': 'ubi',
            'singkong': 'singkong',
            'ketela': 'singkong',
            'talas': 'talas',
            
            # Bumbu dan rempah
            'jahe': 'jahe',
            'kunyit': 'kunyit',
            'kunir': 'kunyit',
            'kencur': 'kencur',
            'lengkuas': 'lengkuas',
            'laos': 'lengkuas',
            'sereh': 'serai',
            'serai': 'serai',
            'daun jeruk': 'daun jeruk',
            'daun salam': 'daun salam',
            'daun pandan': 'daun pandan',
            'kemiri': 'kemiri',
            'ketumbar': 'ketumbar',
            'jinten': 'jintan',
            'jintan': 'jintan',
            'merica': 'merica',
            'lada': 'merica',
            'pala': 'pala',
            'cengkeh': 'cengkeh',
            'cengkih': 'cengkeh',
            'kayu manis': 'kayu manis',
            'kapulaga': 'kapulaga',
            'asam jawa': 'asam jawa',
            'asem': 'asam jawa',
            'garam': 'garam',
            'gula': 'gula',
            'gula pasir': 'gula pasir',
            'gula merah': 'gula merah',
            'gula aren': 'gula aren',
            'gula jawa': 'gula jawa',
            
            # Minyak dan lemak
            'minyak goreng': 'minyak goreng',
            'minyak kelapa': 'minyak kelapa',
            'minyak sawit': 'minyak sawit',
            'mentega': 'mentega',
            'margarin': 'margarin',
            'butter': 'mentega',
            
            # Santan dan susu
            'santan': 'santan',
            'santan kental': 'santan kental',
            'santan cair': 'santan cair',
            'susu': 'susu',
            'susu sapi': 'susu sapi',
            'susu kambing': 'susu kambing',
            'susu kental manis': 'susu kental manis',
            'susu evaporasi': 'susu evaporasi',
            
            # Kacang-kacangan
            'kacang tanah': 'kacang tanah',
            'kacang mete': 'kacang mete',
            'kacang almond': 'kacang almond',
            'kacang hijau': 'kacang hijau',
            'kacang merah': 'kacang merah',
            'kacang kedelai': 'kacang kedelai',
            'tahu': 'tahu',
            
            # Buah-buahan
            'pisang': 'pisang',
            'apel': 'apel',
            'jeruk': 'jeruk',
            'mangga': 'mangga',
            'pepaya': 'pepaya',
            'semangka': 'semangka',
            'melon': 'melon',
            'anggur': 'anggur',
            'nanas': 'nanas',
            'kelapa': 'kelapa',
            'kelapa parut': 'kelapa parut',
            'kelapa muda': 'kelapa muda',
            
            # Pasta dan bumbu siap pakai
            'sambal': 'sambal',
            'sambal oelek': 'sambal oelek',
            'sambal terasi': 'sambal terasi',
            'petis': 'petis',
            'terasi': 'terasi',
            'kecap': 'kecap',
            'kecap manis': 'kecap manis',
            'kecap asin': 'kecap asin',
            
            # Common typos atau singkatan
            'msg': 'msg',
            'micin': 'msg',
            'vetsin': 'msg',
            'penyedap rasa': 'msg',
            'royco': 'penyedap rasa',
            'masako': 'penyedap rasa',
        }
    
    def normalize_spelling(self, text):
        """Normalisasi ejaan yang salah atau tidak baku"""
        if pd.isna(text) or text == '':
            return ''
        
        text = str(text).lower().strip()
        
        # Normalisasi phrase lengkap terlebih dahulu (2-3 kata)
        for wrong_phrase, correct_phrase in self.spelling_normalization.items():
            if len(wrong_phrase.split()) > 1:
                text = text.replace(wrong_phrase, correct_phrase)
        
        # Kemudian normalisasi kata per kata
        words = text.split()
        normalized_words = []
        
        for word in words:
            # Hilangkan tanda baca dari kata
            clean_word = word.strip('.,!?()[]{}"\'-')
            # Cek apakah ada normalisasi untuk kata ini
            normalized_word = self.spelling_normalization.get(clean_word, clean_word)
            normalized_words.append(normalized_word)
        
        return ' '.join(normalized_words)
    
    def clean_text(self, text):
        """Membersihkan teks bahasa Indonesia dengan normalisasi ejaan"""
        if pd.isna(text) or text == '':
            return ''
        
        # Normalisasi ejaan terlebih dahulu
        text = self.normalize_spelling(text)
        
        # Convert to lowercase (sudah dilakukan di normalize_spelling)
        text = str(text).lower()
        
        # Remove numbers and punctuation
        text = re.sub(r'\d+', '', text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove stopwords
        words = text.split()
        words = [word for word in words if word not in self.stopwords and word not in self.recipe_stopwords]
        
        return ' '.join(words)
    
    def extract_keywords(self, text, max_words=10):
        """Ekstrak kata kunci penting dari teks"""
        cleaned = self.clean_text(text)
        words = cleaned.split()
        
        # Ambil kata-kata yang paling sering muncul
        word_freq = Counter(words)
        return [word for word, freq in word_freq.most_common(max_words)]
    
    def process_ingredients(self, ingredient_text):
        """Khusus untuk memproses ingredients dengan tetap mempertahankan informasi penting"""
        if pd.isna(ingredient_text) or ingredient_text == '':
            return ''
        
        # Normalisasi ejaan
        normalized = self.normalize_spelling(ingredient_text)
        
        # Bersihkan tapi jangan hapus stopwords recipe yang penting untuk ingredients
        text = str(normalized).lower()
        text = re.sub(r'\d+', '', text)  # hapus angka
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = ' '.join(text.split())  # normalize whitespace
        
        # Hanya hapus stopwords umum, bukan recipe stopwords
        words = text.split()
        words = [word for word in words if word not in self.stopwords]
        
        return ' '.join(words)