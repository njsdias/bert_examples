import src.config as config
import torch

class BERTDataset:
    def __init__(self, review, target):

        # review: list with review text
        self.review = review
        # list of zeros,ones (positive, negative)
        self.target = target
        # token: words / sentences

        self.tokenizer = config.TOKENIZER
        self.max_len = config.MAX_LEN

    def __len__(self):
        """
        :return: the total length of a review
        """
        return len(self.review)

    def __getitem__(self, item):
        """
        Takes an item and returns the item from our dataset
        :return:
        """
        review = str(self.review)

        # review the weird spaces
        review = " ".join(review.split())

        # can encode two strings at the time
        inputs = self.tokenizer.encode_plus(review,  # first text
                                            None,    # second text
                                            add_special_tokens=True,  # CLS token
                                            max_length=self.max_len # length of token
                                            )

        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]

        # Padding section
        # we will fill with zeros the positions that
        # until reach the length more than padding_length
        # to put the tensor all with the same size
        padding_length = self.max_len - len(ids)
        ids = ids + ([0] * padding_length)
        mask = mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)

        return {
            'ids': torch.tensor(ids, dtype=torch.float),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.target[item], dtype=torch.float)
        }



