import transformers
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from datasets import load_dataset, load_metric
import numpy as np
import csv
from csv import reader
import json
from rouge import Rouge
import nltk
from torch import nn
from transformers.modeling_tf_utils import get_initializer

metric = Rouge(metrics=["rouge-n"], max_n=2, apply_best=True, apply_avg=False)

tokenizer = AutoTokenizer.from_pretrained("/scratch/oe7/rp3665/models/clinicalbert")
model = TFAutoModelForSequenceClassification.from_pretrained("/scratch/oe7/rp3665/models/clinicalbert", num_labels=2)

model.classifier.activation = tf.keras.activations.sigmoid


def create_training_dataset(csv_name, data_file_name, data_num):  
  csv_file = open(csv_name, 'w')
  data_file = open(data_file_name, 'r')

  data_load = json.load(data_file)
  csv_write = csv.writer(csv_file)
  csv_write.writerow(["question", "snippet", "rouge-2", "label"])

  rouge_score_get = Rouge(metrics=["rouge-n"], max_n=2)
    
  rows = []
  top_n = 5
  question_num = 0
  for question_set in data_load["questions"]:
    if question_num <= data_num:
      sub_rows = []
      question = question_set["body"]
      top = []
      top_id = {}
      snip_num = 0
      chosen = 0
      for snippet in question_set["snippets"]:
          snip_text = snippet["text"]
          rouge_score = rouge_score_get.get_scores(question, snip_text)
          rouge_2_f = rouge_score["rouge-2"]["f"]
          top.append(rouge_2_f)
          sub_rows.append([question, snip_text, rouge_2_f])
        
      top_indx = sorted(range(len(top)), key=lambda i: top[i], reverse=True)[:5]

      for indx in top_indx:
        sub_rows[indx].append(1)

      for row in sub_rows:
        if len(row) < 4:
          row.append(0)

      rows.append(sub_rows)
    question_num += 1

  for final_row in rows:
    csv_write.writerows(final_row)

  csv_file.close()
  data_file.close()

def create_validation_dataset(csv_name, data_file_name, data_num):  
  csv_file = open(csv_name, 'w')
  data_file = open(data_file_name, 'r')

  data_load = json.load(data_file)
  csv_write = csv.writer(csv_file)
  csv_write.writerow(["question", "snippet", "rouge-2", "label"])

  rouge_score_get = Rouge(metrics=["rouge-n"], max_n=2)
    
  rows = []
  top_n = 5
  question_num = 0
  for question_set in data_load["questions"]:
    if question_num > data_num and question_num <= data_num*2:
      sub_rows = []
      question = question_set["body"]
      top = []
      top_id = {}
      snip_num = 0
      chosen = 0
      for snippet in question_set["snippets"]:
          snip_text = snippet["text"]
          rouge_score = rouge_score_get.get_scores(question, snip_text)
          rouge_2_f = rouge_score["rouge-2"]["f"]
          top.append(rouge_2_f)
          sub_rows.append([question, snip_text, rouge_2_f])
        
      top_indx = sorted(range(len(top)), key=lambda i: top[i], reverse=True)[:5]

      for indx in top_indx:
        sub_rows[indx].append(1)

      for row in sub_rows:
        if len(row) < 4:
          row.append(0)

      rows.append(sub_rows)
    question_num += 1

  for final_row in rows:
    csv_write.writerows(final_row)

  csv_file.close()
  data_file.close()

def create_test_dataset(csv_name, data_file_name):  
  csv_file = open(csv_name, 'w')
  data_file = open(data_file_name, 'r')

  data_load = json.load(data_file)
  csv_write = csv.writer(csv_file)
  csv_write.writerow(["question", "snippet", "rouge-2", "label"])

  rouge_score_get = Rouge(metrics=["rouge-n"], max_n=2)
    
  rows = []
  top_n = 5
  for question_set in data_load["questions"]:
    sub_rows = []
    question = question_set["body"]
    top = []
    top_id = {}
    snip_num = 0
    chosen = 0
    for snippet in question_set["snippets"]:
        snip_text = snippet["text"]
        rouge_score = rouge_score_get.get_scores(question, snip_text)
        rouge_2_f = rouge_score["rouge-2"]["f"]
        top.append(rouge_2_f)
        sub_rows.append([question, snip_text, rouge_2_f])
      
    top_indx = sorted(range(len(top)), key=lambda i: top[i], reverse=True)[:5]

    for indx in top_indx:
      sub_rows[indx].append(1)

    for row in sub_rows:
      if len(row) < 4:
        row.append(0)

    rows.append(sub_rows)

  for final_row in rows:
    csv_write.writerows(final_row)

  csv_file.close()
  data_file.close()

def create_ideal_answer_training(data_file_name, data_num):  
  data_file = open(data_file_name, 'r')

  data_load = json.load(data_file)

    
  question_answers = []
  question_num = 0
  for question_set in data_load["questions"]:
    if question_num <= data_num:
      question_answers.append(question_set["ideal_answer"])
    question_num += 1

  data_file.close()
  return question_answers

def create_ideal_answer_validation(data_file_name, data_num):  
  data_file = open(data_file_name, 'r')

  data_load = json.load(data_file)

    
  question_answers = []
  question_num = 0
  for question_set in data_load["questions"]:
    if question_num > data_num and question_num <= data_num*2:
      question_answers.append(question_set["ideal_answer"])
    question_num += 1

  data_file.close()
  return question_answers



dataset_split_num = 1871
train_path = '/scratch/oe7/rp3665/datasets/BioASQ-training9b/training9b.json'
validation_path = '/scratch/oe7/rp3665/datasets/BioASQ-training9b/training9b.json'
test_path = "/scratch/oe7/rp3665/datasets/Task9BGoldenEnriched/Task9BGoldenEnriched/9B3_golden.json"


datasets = load_dataset('csv', data_files={"train": '/scratch/oe7/rp3665/datasets/batch_3/training_data.csv', 
                                           "validation": "/scratch/oe7/rp3665/datasets/batch_3/validation_data.csv",
                                           "test": "/scratch/oe7/rp3665/datasets/batch_3/test_data.csv"
                                           })


def tokenize_dataset(dataset):
  encoded = tokenizer(
      dataset["question"],
      dataset["snippet"],
      padding=True,
      truncation=True,
      return_tensors="np",
  )
  return encoded.data

tok_data = {
    split: tokenize_dataset(datasets[split]) for split in datasets.keys()
}

from tensorflow.keras.optimizers.schedules import PolynomialDecay

bat_size = 8
epoch_num = 1

train_steps = (len(tok_data["train"]["input_ids"]) // bat_size) * epoch_num
lr_scheduler = PolynomialDecay(
    initial_learning_rate=1e-4,
    end_learning_rate=0.,
    decay_steps=train_steps
    )

from tensorflow.keras.optimizers import Adam

opt = Adam(learning_rate=lr_scheduler)

from tensorflow.keras.losses import SparseCategoricalCrossentropy, MeanSquaredError
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras import Input


from tqdm.keras import TqdmCallback

model.compile(optimizer=opt, loss=SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])
# model.compile(optimizer=opt, loss=MeanSquaredError(), metrics=["accuracy"])

model.fit(
    tok_data["train"],
    np.array(datasets["train"]["label"]),
    validation_data=(
        tok_data["validation"],
        np.array(datasets["validation"]["label"])
    ),
    validation_batch_size=bat_size,
    batch_size=bat_size,
    verbose=0,
    callbacks=[TqdmCallback(verbose=2)],
    epochs=epoch_num,
    # steps_per_epoch=1 # for debugging
)

import sys
np.set_printoptions(threshold=sys.maxsize)

def transform(in_string):
  tok_string = tokenizer(
      in_string,
      padding=True,
      truncation=True,
      return_tensors="np")
  return np.average(model(
    input_ids=tok_string["input_ids"],
    attention_mask=tok_string["attention_mask"],
    token_type_ids=tok_string["token_type_ids"],
    training=None, 
    output_hidden_states=True
    )["hidden_states"][0], 1)
	
def get_top_n_cos(dataset, top_n):
  curr_top_n_indicies = []
  summaries = []
  q_tracker = dataset["question"][0]
  final_question = transform(dataset["question"][0])
  compare = tf.keras.layers.Dot(axes=1, normalize=True)
  idx = 0

  for question, snippet in zip(dataset["question"], dataset["snippet"]):
    if question == q_tracker:
      if len(curr_top_n_indicies) < top_n:
        curr_top_n_indicies.append(dataset["snippet"].index(snippet))
      else:
        lowest = 0
        for search_idx, val in enumerate(curr_top_n_indicies):
          if compare([final_question, transform(dataset["snippet"][val])]) > compare([final_question, transform(dataset["snippet"][curr_top_n_indicies[lowest]])]):
            lowest = search_idx
        final_snippet = transform(snippet)
        if compare([final_question, final_snippet]) > compare([final_question, transform(dataset["snippet"][curr_top_n_indicies[lowest]])]):
            curr_top_n_indicies[lowest] = idx
    else:
      to_append = ""
      while len(curr_top_n_indicies) != 0:
        lowest_idx = 0
        for idx_idx, item in enumerate(curr_top_n_indicies):
          curr_item_score = compare([final_question, transform(dataset["snippet"][item])])
          lowest_item_score = compare([final_question, transform(dataset["snippet"][curr_top_n_indicies[lowest_idx]])])
          if curr_item_score > lowest_item_score:
            lowest_idx = idx_idx
        to_append = to_append + dataset["snippet"][curr_top_n_indicies[lowest_idx]] + " "
        curr_top_n_indicies.pop(lowest_idx)
      summaries.append(to_append)
      curr_top_n_indicies.clear()
      q_tracker = question
      final_question = transform(question)
      curr_top_n_indicies.append(dataset["snippet"].index(snippet))
    idx += 1
  to_append = ""
  while len(curr_top_n_indicies) != 0:
    lowest_idx = 0
    for idx_idx, item in enumerate(curr_top_n_indicies):
      curr_item_score = compare([final_question, transform(dataset["snippet"][item])])
      lowest_item_score = compare([final_question, transform(dataset["snippet"][curr_top_n_indicies[lowest_idx]])])
      if curr_item_score > lowest_item_score:
        lowest_idx = idx_idx
    to_append = to_append + dataset["snippet"][curr_top_n_indicies[lowest_idx]] + " "
    curr_top_n_indicies.pop(lowest_idx)
  summaries.append(to_append)
  return summaries


test_hypothesis = get_top_n_cos(datasets["test"], 2)
ideal_test = create_ideal_answer_training(test_path, 100)
results = metric.get_scores(test_hypothesis, ideal_test)
# for hypo, actual in zip(test_hypothesis, ideal_test):
  # scores = [metric.get_scores(hypo, answer)["rouge-2"]["f"] for answer in actual]
  # results.append(max(scores))
# results = np.average(results)
print(results)

def get_test_results_json(initial_json_file, results_json_name, results):
  data_load = json.load(open(initial_json_file, "r"))
  results_json = open(results_json_name, "w")

  for question, result in zip(data_load["questions"], results):
    if question["type"] == "yesno":
      question["exact_answer"] = "yes" 
      question["ideal_answer"] = result
    elif question["type"] != "summary":
      question["exact_answer"] = "" 
      question["ideal_answer"] = result
    elif question["type"] == "summary":
      question["ideal_answer"] = result
  
  json.dump(obj=data_load, fp=results_json, indent=4)
  results_json.close()

get_test_results_json(test_path, "/scratch/oe7/rp3665/results/test_results.json", test_hypothesis)
