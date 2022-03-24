from rouge import Rouge
import pandas as pd

def rouge_metric(df):
  y_true = list(df['TRUE_SUMMARY'])
  y_pred = list(df['PREDICT_SUMMARY'])

  rouge = Rouge()
  score = [rouge.get_scores(y_true[i], y_pred[i], avg=True)['rouge-1']['f'] for i in range(len(y_true))]
  df['SCORE'] = score
  
  return df
