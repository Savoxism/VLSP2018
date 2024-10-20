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
    
class VLSP2018Loader:
    @staticmethod
    def load(train_csv_path, val_csv_path, test_csv_path):
        dataset_paths = {'train': train_csv_path, 'val': val_csv_path, 'test': test_csv_path}
        raw_datasets = load_dataset('csv', data_files={k: v for k, v in dataset_paths.items() if v})
        return raw_datasets
    
    @staticmethod
    def preprocess_and_tokenize(text_data, preprocessor, tokenizer, batch_size, max_length):
        # print('[INFO] Preprocessing and tokenizing text data...')
        def transform_each_batch(batch):
            preprocessed_batch = preprocessor.process_batch(batch)
            return tokenizer(preprocessed_batch, max_length=max_length, padding='max_length', truncation=True)
        
        if type(text_data) == str: return transform_each_batch([text_data])
        return text_data.map(
            lambda reviews: transform_each_batch(reviews['Review']), 
            batched=True, batch_size=batch_size
        ).remove_columns('Review')
    
    @staticmethod
    def labels_to_flatten_onehot(datasets):
        # print('[INFO] Transforming "Aspect#Categoy,Polarity" labels to flattened one-hot encoding...')
        model_input_names = ['input_ids', 'token_type_ids', 'attention_mask']
        label_columns = [col for col in datasets['train'].column_names if col not in ['Review', *model_input_names]]
        
        def transform_each_review(review): 
            review['FlattenOneHotLabels'] = sum([
                SentimentMapping.INDEX_TO_ONEHOT[review[aspect_category]] # Get one-hot encoding
                for aspect_category in label_columns
            ], []) # Need to be flattened to match the model's output shape
            return review 
        return datasets.map(transform_each_review, num_proc=8).select_columns(['FlattenOneHotLabels', *model_input_names])
        
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
                    
# if __name__ == "__main__":
#     # Hotel Domain
#     hotel_train_path = '../datasets/vlsp2018_hotel/1-VLSP2018-SA-Hotel-train.txt'
#     hotel_dev_path = '../datasets/vlsp2018_hotel/2-VLSP2018-SA-Hotel-dev.txt'
#     hotel_test_path = '../datasets/vlsp2018_hotel/3-VLSP2018-SA-Hotel-test.txt'
#     vlsp_hotel_parser = VLSP2018Parser(hotel_train_path, hotel_dev_path, hotel_test_path)
#     vlsp_hotel_parser.txt_to_csv()
    

    