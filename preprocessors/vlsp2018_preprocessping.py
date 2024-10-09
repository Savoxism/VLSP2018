import re
import csv
from tqdm import tqdm
from datasets import load_dataset


class SentimentMapping:
    INDEX_TO_SENTIMENT = {
        0: None,
        1: 'positive',
        2: 'negative',
        3: 'neutral',
    }
    
    INDEX_TO_ONEHOT = {
        0: [1, 0, 0, 0],
        1: [0, 1, 0, 0],
        2: [0, 0, 1, 0],
        3: [0, 0, 0, 1],
    }
    
    SENTIMENT_TO_INDEX = {
        None: 0,
        'positive': 1,
        'negative': 2,
        'neutral': 3,
    }
    
class VLSP2018Parser:
    def __init__(self, *file_paths):
        self.files = {key: path for key, path in zip(['train', 'val', 'test'], file_paths) if path}
        self.reviews = {key: [] for key in self.files}
        self.aspect_categories = set()
        self._parse_files()
        
    def _parse_files(self):
        for dataset, path in self.files.items():
            with open(path, 'r', encoding='utf-8') as f:
                for block in f.read().strip().split('\n\n'):
                    lines = block.split('\n')
                    sentiments = re.findall(r'\{([^,]+)#([^,]+), ([^}]+)\}', lines[2].strip())
                    review_data = {f'{a.strip()}#{c.strip()}': SentimentMapping.SENTIMENT_TO_INDEX[p.strip()] 
                                   for a, c, p in sentiments}
                    self.aspect_categories.update(review_data.keys())
                    self.reviews[dataset].append((lines[1].strip(), review_data))
        
        self.aspect_categories = sorted(self.aspect_categories)
    
    def txt_to_csv(self):
        for dataset, path in self.files.items():
            with open(path.replace('.txt', '.csv'), 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Review'] + self.aspect_categories)
                for review, data in self.reviews[dataset]:
                    writer.writerow([review] + [data.get(cat, 0) for cat in self.aspect_categories])
                    
if __name__ == "__main__":
    # hotel_file = 'datasets/vlsp2018_hotel/train.txt'
    # VLSP2018Parser(hotel_file).txt_to_csv()
    
    restaurant_file = "datasets/vlsp2018_restaurant/train.txt"
    VLSP2018Parser(restaurant_file).txt_to_csv()  
    
    