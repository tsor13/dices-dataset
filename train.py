import pandas as pd
import numpy as np
df1 = pd.read_csv("350/diverse_safety_adversarial_dialog_350.csv")
df2 = pd.read_csv("990/diverse_safety_adversarial_dialog_990.csv")
df2["item_id"] = df2["item_id"] + df1["item_id"].max() + 1
# combine both dfs together
dfs = [
    df1, df2
]
df = pd.concat(dfs)
df

# %%
# aggregate by item_id and rater_, value counting the  'degree_of_harm'
agg1 = df.groupby(['item_id', 'rater_race']).agg({'degree_of_harm': 'value_counts'}).unstack().fillna(0)
agg1.columns = agg1.columns.droplevel(0)
agg1.columns = [str(col) for col in agg1.columns]
# merge down so all same level
agg1 = agg1.reset_index()

data = pd.merge(agg1, df[['item_id', 'context', 'response']].drop_duplicates(), on='item_id', how='inner')
# harm is Benign, Debatable, Moderate, Extreme from cols
data['harm'] = data.apply(lambda row: np.array([row['Benign'], row['Debatable'], row['Moderate'], row['Extreme']]), axis=1)
# normalize
data['harm'] = data['harm'].apply(lambda x: x / np.sum(x))
# delete Benign, Debatable, Moderate, Extreme
data = data.drop(columns=['Benign', 'Debatable', 'Moderate', 'Extreme'])
# write prompt which is:
# User demographic: <rater_> \t Context: <context>
data['prompt'] = 'Rater demographic: ' + data['rater_race'] + '\tContext: ' + data['context'] + '\tResponse: ' + data['response']
data

# %%
import numpy as np
# randomly split data into train and test
train_perc = 0.8
item_ids = data.index.get_level_values(0).unique()
np.random.seed(0)
train_ids = np.random.choice(item_ids, int(len(item_ids) * train_perc), replace=False)
print(train_ids)
train_data = data[data['item_id'].isin(train_ids)]
test_data = data[~data['item_id'].isin(train_ids)]
train_data

# %%
test_data

# %%
from transformers import AutoTokenizer, AutoModelForSequenceClassification, T5EncoderModel
# initialize with smallest t5 model
model_name = 'google/flan-t5-small'
tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForSequenceClassification.from_pretrained(model_name)
model = T5EncoderModel.from_pretrained(model_name)

# %%
# do 4-way classification on the context
from torch import nn
import torch


# define the model
class Classifier(nn.Module):
    def __init__(self, model, num_labels=4):
        super(Classifier, self).__init__()
        self.model = model
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(512, num_labels)
    def forward(self, input_ids, attention_mask):
        # print(input_ids.shape, attention_mask.shape)
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # last_hidden_state = outputs.encoder_last_hidden_state
        last_hidden_state = outputs.last_hidden_state
        cls_token = last_hidden_state[:, 0, :]
        cls_token = self.dropout(cls_token)
        logits = self.classifier(cls_token)
        # to probabilities
        logits = torch.log_softmax(logits, dim=1)
        return logits

# %%
# initialize the model
classifier = Classifier(model)
# run a test forward pass
# input_ids = tokenizer(train_data['prompt'].tolist()[0:2], padding=True, truncation=True, return_tensors='pt')['input_ids']
# logits = classifier(input_ids, attention_mask=input_ids != tokenizer.pad_token_id)

# train the model
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

# initialize the optimizer
optimizer = AdamW(classifier.parameters(), lr=1e-5)
# cross entropy loss
loss_fn = nn.KLDivLoss()
# make a dataloader which goes from prompt to harm
tokenizer.pad_token = tokenizer.eos_token
# tokenize dataset
# train_dataloader = DataLoader(train_data[['prompt', 'harm']].to_dict('records'), batch_size=8, shuffle=True)
train_dataloader = DataLoader(train_data[['prompt', 'harm']].to_dict('records'), batch_size=5, shuffle=True)

# train the model
classifier.train()
num_epochs = 3
loop = tqdm(total=num_epochs*len(train_dataloader))
for epoch in range(num_epochs):
    # for batch in tqdm(train_dataloader):
    for batch in train_dataloader:
        optimizer.zero_grad()
        # tokenize prompt
        inputs = tokenizer(batch['prompt'], padding=True, truncation=True, return_tensors='pt')
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        labels = batch['harm']
        model(input_ids=input_ids, attention_mask=attention_mask)
        # print(input_ids.shape)
        logits = classifier(input_ids, attention_mask)
        loss = loss_fn(logits, labels)
        loss.backward()
        loop.set_description(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')
        loop.update(1)
        optimizer.step()

# %%
from transformers import AutoTokenizer, AutoModelForSequenceClassification, T5EncoderModel
# initialize with smallest t5 model
model_name = 'google/flan-t5-small'
tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForSequenceClassification.from_pretrained(model_name)
model = T5EncoderModel.from_pretrained(model_name)

output = model(input_ids)
# output = model(input_ids[2:4,0:10])
# output.encoder_last_hidden_state.shape

# %%
output.last_hidden_state.shape


