from msilib.sequence import tables
import os
import csv
import torch
import torchaudio
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from torch.utils.data import Dataset
# from torch.optim import optim
SAMPLE_RATE = 22050
charset = ' abcdefghijklmnopqrstuvwxyz,.'
YMAX = 150

ret = []
with open("data\LJSpeech-1.1\metadata.csv", newline='',encoding='gb18030',errors='ignore') as csvfile:
    reader = csv.reader(csvfile, delimiter='|')
    for i in reader:
            # print(i[1])
        idx = i[0]
        content = [charset.index(c)+1 for c in i[1].lower() if c in charset]
            # print(len(content))
        if len(content) < YMAX:
            ret.append((f"data\LJSpeech-1.1\wavs\{idx}.wav", content))

# print(ret[0])
mel_transform = torchaudio.transforms.MelSpectrogram(SAMPLE_RATE,
 n_fft=1024, win_length=1024,hop_length=256,n_mels=80)

def load_example(x):
    waveform, sample_rate = torchaudio.load(x, normalize = True)
    assert(sample_rate == SAMPLE_RATE)
    mel_specgram = mel_transform(waveform)
    return mel_specgram[0].T

dic={}
class LJSpeech(Dataset):
    def __init__(self):
        self.meta = ret

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        if idx not in dic:
            x,y = self.meta[idx]
            dic[idx] = load_example(x),y
        return dic[idx]


class Rec(nn.Module):
    def __init__(self):
        super().__init__()
        self.prepare = nn.Sequential(
            nn.Linear(80, 128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU())
        self.encoder = nn.GRU(128,128, batch_first = False)
        self.decode = nn.Linear(128,len(charset))
    
    def forward(self,x):
        x = self.prepare(x)
        x = nn.functional.relu(self.encoder(x)[0])
        x = self.decode(x)
        return torch.nn.functional.log_softmax(x,dim=2)



def collate_fn(batch_data):
    """
    自定义 batch 内各个数据条目的组织方式
    :param data: 元组，第一个元素：句子序列数据，第二个元素：长度 第2维：句子标签
    :return: 填充后的句子列表、实际长度的列表、以及label列表
    """
    # batch_data 为一个batch的数据组成的列表，data中某一元素的形式如下
    # (tensor([1, 2, 3, 5]), 4, 0)
    # 后续将填充好的序列数据输入到RNN模型时需要使用pack_padded_sequence函数
    # pack_padded_sequence函数要求要按照序列的长度倒序排列
    batch_data.sort(key=lambda xi: len(xi[0]), reverse=True)
    data_length = [xi[0].shape[0] for xi in batch_data]
    target_length = [len(xi[1]) for xi in batch_data]
    
    sent_seq = [xi[0] for xi in batch_data]
    label = [xi[1]+[0]*(YMAX-len(xi[1])) for xi in batch_data]
    label = torch.LongTensor(label)
    padded_sent_seq = torch.nn.utils.rnn.pad_sequence(sent_seq, batch_first=False, padding_value=0)
    return padded_sent_seq, label[:, :max(target_length)], data_length, target_length


def get_dataloader(batch_size):
    dset = LJSpeech()
    trainloader = torch.utils.data.DataLoader(dset,
    batch_size = batch_size, collate_fn = collate_fn, shuffle = False, num_workers=0,drop_last = False)
    return dset, trainloader

    
def train():
    batch_size = 32
    dset, trainloader = get_dataloader(batch_size)
    ctc_loss = nn.CTCLoss(reduction='mean').cuda()
    model = Rec().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
    t = tqdm(trainloader, total = len(dset)//batch_size)
    for data in t:
        print(len(data))
    # val = load_example('data/LJ037-0171.wav')
    
    # for epoch in range(100):
    #      t = tqdm(trainloader, total = len(dset)//batch_size)
    #     for data in trainloader:
    #         print(type(data))

            
if __name__ == "__main__":
    print(1)    
    train()
    print(1)





        




