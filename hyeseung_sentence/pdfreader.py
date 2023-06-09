import pdfplumber
import os
import re
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
import torch


class PDFDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data[index]
        inputs = self.tokenizer.encode_plus(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_token_type_ids=True,
            return_attention_mask=True,
        )

        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "token_type_ids": inputs["token_type_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
        }


def get_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        extracted_text = ""
        for page in pdf.pages:
            extracted_text += page.extract_text().replace("\n", " ")
    return extracted_text


def preprocess_text(text):
    preprocessed_text = re.sub(r'\n+', ' ', text).strip()
    return preprocessed_text


def main():
    pdf_path = "YOUR_PDF_PATH_HERE"
    raw_text = get_text_from_pdf(pdf_path)
    preprocessed_text = preprocess_text(raw_text)

    texts = [preprocessed_text]

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased")

    MAX_LENGTH = 128
    BATCH_SIZE = 8

    dataset = PDFDataset(texts, tokenizer, MAX_LENGTH)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE)


if __name__ == "__main__":
    main()
