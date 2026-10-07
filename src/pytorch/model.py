import torch
import torch.nn as nn


class NewsClassifier(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim=128,
        hidden_dim=128,
        num_classes=4,
        dropout=0.3
    ):
        super().__init__()

        # Word Embedding
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0
        )

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True
        )

        # Attention layer
        self.attention = nn.Linear(
            hidden_dim * 2,
            1
        )

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Final classification layer
        self.fc = nn.Linear(
            hidden_dim * 2,
            num_classes
        )

    def forward(self, x):

        # x shape:
        # [batch_size, sequence_length]

        embedded = self.embedding(x)

        # embedded shape:
        # [batch_size, sequence_length, embedding_dim]

        lstm_output, _ = self.lstm(embedded)

        # lstm_output shape:
        # [batch_size, sequence_length, hidden_dim * 2]

        attention_scores = self.attention(lstm_output)

        # attention_scores shape:
        # [batch_size, sequence_length, 1]

        attention_weights = torch.softmax(
            attention_scores,
            dim=1
        )

        # Weighted sum of LSTM outputs
        context_vector = torch.sum(
            attention_weights * lstm_output,
            dim=1
        )

        context_vector = self.dropout(context_vector)

        output = self.fc(context_vector)

        return output