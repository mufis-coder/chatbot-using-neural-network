# Chatbot Intent - Neural Network

The chatbot project uses the intent concept with a neural network model

- How to train

```py
python training.py
```

- Run chatbot

```py
python chatbot.py
```

- Custom dataset, create file ```dataset\intents.json```. Inside ```intents.json``` you can fill:
  
```txt
{"intents": [
    {
        "tag": "{tag1}",
        "patterns": ["{question-1}", "{question-2}", "{question-3}"],
        "responses": ["{answer-1}", "{answer-2}", "{answer-3}"],
        "extend": "{Additional answer that doesn't need to be paraphrased GPT-03}"
    },
]}
```
