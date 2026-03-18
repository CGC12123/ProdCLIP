import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor
from PIL import Image
import logging
from config import config
from utils.logger import get_dataset_logger


logger = get_dataset_logger()


class MultimodalDataset(Dataset):
    """Dataset class for multimodal retrieval"""

    def __init__(self, dataframe, processor, transform=None, use_category=False):
        """
        Args:
            dataframe: DataFrame containing image and description
            processor: CLIP processor for tokenization and image preprocessing
            transform: Optional transforms to apply to images
            use_category: Whether to return category information (for hard negative training)
        """
        self.dataframe = dataframe.reset_index(drop=True)
        self.processor = processor
        self.transform = transform
        self.use_category = use_category

        # Verify all image paths exist
        missing_paths = []
        for idx, row in self.dataframe.iterrows():
            img_path = os.path.join(config.data.image_dir, row['image'])
            if not os.path.exists(img_path):
                missing_paths.append(img_path)

        if missing_paths:
            logger.warning(f"Missing {len(missing_paths)} image paths: {missing_paths[:5]}...")

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        img_path = os.path.join(config.data.image_dir, row['image'])

        # Load and preprocess image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            logger.error(f"Failed to load image {img_path}: {str(e)}")
            # Return a dummy image if loading fails
            image = Image.new('RGB', (224, 224), color='black')

        # Preprocess image using CLIP processor
        inputs = self.processor(
            text=row['description'],  # caption
            images=image,
            return_tensors="pt",
            padding=True,  # Ensure padding is applied
            truncation=True,
            max_length=config.model.max_length
        )

        # Clean caption text
        clean_caption = row['description'].strip().lower()

        result = {
            'pixel_values': inputs['pixel_values'][0],
            'input_ids': inputs['input_ids'][0],
            'attention_mask': inputs['attention_mask'][0],
            'caption': clean_caption,
            'image_path': row['image'],
            'display_name': row.get('display name', ''),
            'category': row.get('category', '')
        }

        # Include category information if requested
        if self.use_category:
            result['category'] = row.get('category', 'unknown')

        return result


def load_data(data_path: str = None):
    """
    Load the CSV data and return the DataFrame

    Args:
        data_path: Path to the CSV file (uses config default if None)

    Returns:
        DataFrame containing the data
    """
    if data_path is None:
        data_path = config.data.data_path

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} samples from {data_path}")

    # Validate required columns
    required_columns = ['image', 'description']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in data")

    # Clean captions
    df['description'] = df['description'].apply(lambda x: str(x).strip().lower() if pd.notna(x) else "")

    return df


def split_data(df, splits=None):
    """
    Split the data into train, val, test sets

    Args:
        df: Input DataFrame
        splits: Dictionary with split ratios (default from config)

    Returns:
        Dictionary containing train, val, test DataFrames
    """
    if splits is None:
        splits = config.data.splits

    from sklearn.model_selection import train_test_split

    # Calculate split sizes
    total_samples = len(df)
    train_ratio = splits['train']
    val_ratio = splits['val']
    test_ratio = splits['test']

    # First split: separate train from (val + test)
    train_df, temp_df = train_test_split(
        df,
        test_size=(val_ratio + test_ratio),
        random_state=42,
        shuffle=True
    )

    # Second split: separate val from test
    # Calculate the proportion for val among the remaining (val + test) portion
    if val_ratio + test_ratio > 0:
        val_proportion = val_ratio / (val_ratio + test_ratio)
    else:
        val_proportion = 0.5

    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - val_proportion),
        random_state=42,
        shuffle=True
    )

    logger.info(f"Split data: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    return {
        'train': train_df,
        'val': val_df,
        'test': test_df
    }


def custom_collate_fn(batch):
    """
    Custom collate function to handle variable-length sequences properly
    """
    # Find max lengths for padding
    max_seq_len = max([item['input_ids'].size(0) for item in batch])

    # Pad sequences to max length in the batch
    padded_pixel_values = torch.stack([item['pixel_values'] for item in batch])
    padded_input_ids = []
    padded_attention_masks = []

    for item in batch:
        input_ids = item['input_ids']
        att_mask = item['attention_mask']

        # Pad to max length in batch
        input_pad = torch.cat([
            input_ids,
            torch.zeros(max_seq_len - input_ids.size(0), dtype=input_ids.dtype)
        ], dim=0)
        padded_input_ids.append(input_pad)

        atten_pad = torch.cat([
            att_mask,
            torch.zeros(max_seq_len - att_mask.size(0), dtype=att_mask.dtype)
        ], dim=0)
        padded_attention_masks.append(atten_pad)

    # Stack the padded tensors
    padded_input_ids = torch.stack(padded_input_ids)
    padded_attention_masks = torch.stack(padded_attention_masks)

    # Other fields that don't need stacking
    captions = [item['caption'] for item in batch]
    image_paths = [item['image_path'] for item in batch]
    display_names = [item['display_name'] for item in batch]
    categories = [item['category'] for item in batch]

    return {
        'pixel_values': padded_pixel_values,
        'input_ids': padded_input_ids,
        'attention_mask': padded_attention_masks,
        'caption': captions,
        'image_path': image_paths,
        'display_name': display_names,
        'category': categories
    }


def create_dataloader(dataset, shuffle=True, drop_last=False):
    """
    Create a DataLoader for the given dataset

    Args:
        dataset: Dataset object
        shuffle: Whether to shuffle the data
        drop_last: Whether to drop the last incomplete batch

    Returns:
        DataLoader object
    """
    dataloader = DataLoader(
        dataset,
        batch_size=config.model.batch_size,
        shuffle=shuffle,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues
        pin_memory=torch.cuda.is_available(),
        drop_last=drop_last,
        collate_fn=custom_collate_fn  # Use custom collate function
    )

    return dataloader


def get_data_loaders(processor: CLIPProcessor = None, use_category=False):
    """
    Create and return data loaders for train, val, and test sets

    Args:
        processor: CLIP processor (will be created if None)
        use_category: Whether to return category information (for hard negative training)

    Returns:
        Dictionary containing train, val, test data loaders
    """
    if processor is None:
        from transformers import CLIPProcessor
        processor = CLIPProcessor.from_pretrained(config.model.model_name, use_fast=False)

    # Load and split data
    df = load_data()
    splits = split_data(df)

    # Create datasets for each split
    datasets = {}
    dataloaders = {}

    for split_name, split_df in splits.items():
        dataset = MultimodalDataset(split_df, processor, use_category=use_category)
        datasets[split_name] = dataset

        shuffle = (split_name == 'train')  # Only shuffle training data
        dataloaders[split_name] = create_dataloader(dataset, shuffle=shuffle)

    return dataloaders