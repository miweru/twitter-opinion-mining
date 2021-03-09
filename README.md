# twitter-opinion-mining
Repository for Using Deep Learning for Opinion Mining and Sentiment Classification on Tweets

## Manual Annotation of Tweets
We provide a PyQT6 App *qannotator* under qannotator/
You need to replace the String **BASE_FILE** by the path to a NDJSON File to tweets, as downloaded by https://github.com/twintproject/twint like:
`twint -s "Keyword" -o tweets.ndjson --json --lang en`

This includes skmultiflow to automatically suggest annotions based on your previous annotations. You can add additional labels and the annotated tweets will be appended to the ndjson File.

# Training Classifier
We splitted this into multiple Units:

1. Define and Pretrain Embeddings
2. Create Training and Pretraining Arrays
3. Train Multi-Task Model with Keras
4. Predict Labels of new Tweets

We recommend using *nvidia-docker*  (https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) like:
`nvidia-docker run  -it -p 8888:8888 -v "$(pwd)"/twitter-opionion-mining:/tf tensorflow/tensorflow:latest-gpu-jupyter`


## Pretraining Embeddings
Under **opinion-mining** you find a Dataset which can load annotated Tweets created by annotator and create training Data with it or load downloaded but not annotated tweets to pretrain unsuperviced Embeddings like LDA and the Character Embedding model.

Certain Embedding Models were included like:

1. CharEmbedding. Creates One-Hot-Vectors on Characters. You can set a maximum *seq_len* with a padding and truncating value and a *min_count*, charaters with less frequency will be ignored.
2. BERTweet. Loads the https://huggingface.co/vinai/bertweet-base Embeddins and applies the Normalizer on them. Returns Pooled Output. Includes Batching Mechanism.
3. BERTtweetCONV. Same as BERTweet but returns Output Sequence.
4. ELECTRA. Using the https://huggingface.co/google/electra-base-discriminator and returns output sequence.
5. LDAembedding: Can be pretrained on larger Dataset.

The Pretrainable models will load themselves from the ./embedding-models folder when starting. Pretrained Models will automatically be downloaded by *transformers*.

```
from dataset import TweetLabelDS
from embeddings import CharEmbedding, BERTweet, LDAembedding, BERTweetCONV, ELECTRA
from smart_open import open
import json

embeddings = {"char":CharEmbedding(), "bert":BERTweet(), "lda":LDAembedding(), "bert_conv":BERTweetCONV(), "electra":ELECTRA()}
tweets = [l for l in (json.loads(l) for l in open("pretrain-tweets.ndjson")) if l["language"]=="en"]
ds = TweetLabelDS(tweets=tweets, embeddings=embeddings)
ds.unsupervised_embedding_pretrain()
```

## Create Training and Pretraining Arrays

We are pretraining the Model on 3 Additional Tweet Sentiment-Classification Datasets:

1. Twitter Sentiment Corpus: https://github.com/FurkanArslan/twitter-text-classification (5394 Tweets in multiple Languages)
2. Sanders Analytics Tweet Dataset: http://boston.lti.cs.cmu.edu/classes/95-865-K/HW/HW3/
3. Kaggles Tweet Sentiment Extraction Dataset: https://www.kaggle.com/c/tweet-sentiment-extraction
4. Twitter US Airline Sentiment: https://www.kaggle.com/crowdflower/twitter-airline-sentiment

Additional Datasets could be gathered from SemEval and similar shared Tasks.

Then you may convert the other datasets into the ndjson format. Because of licencing reasons the dataset is not included here.

You can extract and generate Training Data from your annotated Tweets like that.
```
from dataset import TweetLabelDS
from embeddings import CharEmbedding, BERTweet, LDAembedding, BERTweetCONV, ELECTRA
import json

tweets = json.load(open("tweets.ndjson"))
atweets={}
for _, ts in tweets.items():
    for t in ts:        
        if "labels" in t and len(t["labels"])>1:
            if t["id"] not in atweets:
                atweets[t["id"]]=[]
            atweets[t["id"]].append(t)
tweets_for_training = list(atweets.values())


embeddings = {"char":CharEmbedding(), "bert":BERTweet(), "lda":LDAembedding(),"bert_conv":BERTweetCONV(), "electra":ELECTRA()}

ds = TweetLabelDS(tweets=tweets, embeddings=embeddings)
ds.save_train_data("train_tweets") 
```

# Training

You see a exemplary possibility for pretraining under opinion-mining/Training Example.ipynb

![alt text](https://github.com/miweru/twitter-opinion-mining/raw/main/model.png "Opinion Mining Classifier Architecture")

# Using inference

For Inference you can use TweetLabelPredictor. The Cutoff Value determines when a label will be assigned.
Assumes that the trained *embedding-models* are in the same folder.

```
from preprocessing.inference import TweetLabelPredictor
import json
comparison = [json.loads(l) for l in open("tweets_to_annotate.ndjson")]

model = TweetLabelPredictor()
comparison_ann = model.enrich_datast(comparison, cutoff=0.2)
```
