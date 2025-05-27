"""
author: zhangjunyang
time: 20240723
"""
from torch.nn import TransformerEncoder, TransformerEncoderLayer, Linear
from torch.nn import Module
from torch import nn
import numpy as np
from torch import FloatTensor, LongTensor


def get_model_mask(input_dim, output_dim, dim_feedforward, embedding_dim, n_head, n_layer, class_num, device,
                   ckpt_path=None):
    epispy_predictor_mask = EpispyPredictorMask(input_dim, output_dim, dim_feedforward, embedding_dim, n_head, n_layer,
                                                class_num,
                                                ckpt_path)
    epispy_predictor_mask = epispy_predictor_mask.to(device)
    return epispy_predictor_mask


class EpispyPredictorMask(nn.Module):
    def __init__(self, input_dim, output_dim, dim_feedforward, embedding_dim, n_head, n_layer, class_num,
                 ckpt_path=None, is_mlp=False):
        super(EpispyPredictorMask, self).__init__()
        # 暂时不做模型加载
        assert ckpt_path is None

        self.eeg_bert_mask = EpispyBERTMask(
            input_dim=input_dim,
            out_dim=output_dim,
            embedding_dim=embedding_dim,
            dim_feedforward=dim_feedforward,
            n_head=n_head,
            n_layer=n_layer,
            batch_first=True)
        # MLP layers
        if is_mlp:
            self.mlp = nn.Linear(246 * output_dim, class_num)
        else:
            self.mlp = nn.Linear(output_dim, class_num)

    def forward(self, input_data, is_mlp=False):
        embeddings = self.eeg_bert_mask(input_data)
        # Assume embeddings are in shape (batch_size, sequence_length, d_model)
        # and we use the first representation (at position 0) of each sequence for prediction
        if is_mlp:
            bz, _, _, = input_data.shape
            embeddings = embeddings.reshape((bz, -1))
            output = self.mlp(embeddings)
        else:
            output = self.mlp(embeddings)
        return output


class EpispyBERTMask(Module):
    def __init__(self, input_dim, out_dim, embedding_dim, n_head, n_layer, dim_feedforward, batch_first, *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.embedding_layer = Linear(input_dim, embedding_dim)
        self.projection_layer = Linear(embedding_dim, out_dim)

        transformer_encoder_layer = TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            batch_first=batch_first
        )
        self.transformer_encoder = TransformerEncoder(
            transformer_encoder_layer,
            num_layers=n_layer
        )

    def forward(self, input_data):
        embedding = self.embedding_layer(input_data)
        hidden_state = self.transformer_encoder(embedding)
        output = self.projection_layer(hidden_state)
        return output

    def get_hidden_state(self, input_data, padding_mask=None):
        embedding = self.embedding_layer(input_data)
        hidden_state = self.transformer_encoder(embedding, src_key_padding_mask=padding_mask)
        return hidden_state


def model_eval_mask(model, loader, batch_size, label_type, device, is_mlp=False):
    label_list, predict_list, metric_list = [], [], []
    for batch_idx, data in enumerate(loader, 0):
        input_src, label = data
        assert 0 < len(label) <= batch_size
        input_src = FloatTensor(input_src).to(device)
        predict = model(input_src)
        predict = predict.detach().to('cpu').numpy()
        if is_mlp:
            predict = predict
        else:
            predict = predict[:, 0, :]

        label = label.numpy()
        if label_type == 'continuous':
            label_list.append(label)
            predict_list.append(predict)
            ae = np.absolute((predict - label))
            metric_list.append(np.average(ae))
        else:
            assert label_type == 'discrete'
            predict_idx = np.argmax(predict, axis=1)
            match = predict_idx == label
            metric_list.append(np.average(match))
    metric = np.average(metric_list)
    return metric


def loss_calculate_mask(batch, device, model, loss, label_type, is_mlp=False):
    input_src, label = batch
    input_src = FloatTensor(input_src).to(device)
    predict = model(input_src)
    if is_mlp:
        predict = predict
    else:
        predict = predict[:, 0, :]
    if label_type == 'continuous':
        label = FloatTensor(label).to(device)
    else:
        label = LongTensor(label).to(device)
    loss = loss(predict, label).mean()
    return loss