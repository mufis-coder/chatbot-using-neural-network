# Chatbot Intent - Neural Network and using GPT-03 for customization of answers

The chatbot project uses the intent concept with a neural network model and using GPT-03 for customization of answers

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

## TODO

- [x] Using GPT-03 for customization of answers
- [ ] GPT offline: [https://huggingface.co/docs/transformers/model_doc/gpt_neo](https://huggingface.co/docs/transformers/model_doc/gpt_neo)
- [ ] Text classification using BERT
- [ ] Question And Answering With BERT: [https://towardsdatascience.com/question-and-answering-with-bert-6ef89a78dac](https://towardsdatascience.com/question-and-answering-with-bert-6ef89a78dac)
- [ ] Link GDrive: [https://drive.google.com/drive/folders/1zN8qxN4FWSb_JZ7-cmmlqkaihvxSfCUE](https://drive.google.com/drive/folders/1zN8qxN4FWSb_JZ7-cmmlqkaihvxSfCUE)
