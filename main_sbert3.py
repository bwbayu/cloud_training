import pandas as pd
from src.pipelines.SBERT_pipeline import SBERTPipeline
import logging
import torch
import os

df = pd.read_csv("data/aes_dataset_5k_clean.csv")
df = df[df['dataset'] == 'analisis_essay'][['reference_answer', 'answer', 'score', 'normalized_score', 'dataset', 'dataset_num']]
print(df.info())
df.head()

# Check if the first file exists
df_result = None
if os.path.exists("experiments/results/results_sbert1.csv"):
    df_result = pd.read_csv("experiments/results/results_sbert1.csv")
    print(df_result['config_id'].iloc[-1])
else:
    print("File 'results_sbert1.csv' does not exist.")

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
batch_sizes = [16]
learning_rates = [1e-5, 2e-5, 5e-5, 1e-4]
warm_ups = [0.0, 0.3]
idx = (df_result['config_id'].iloc[-1] + 1) if df_result is not None and not df_result.empty else 0  # index untuk setiap kombinasi
ROOT_DIR = os.getcwd()

for batch_size in batch_sizes:
    for lr in learning_rates:
        for warm_up in warm_ups:
            results = []
            results_epoch = []
            df_result1 = None
            # Check if the second file exists
            if os.path.exists("experiments/results/results_epoch_sbert1.csv"):
                df_result1 = pd.read_csv("experiments/results/results_epoch_sbert1.csv")
                print(max(df_result1['valid_pearson']))
            else:
                print("File 'results_epoch_sbert1.csv' does not exist.")

            # set up hyperparamter
            config = {
                "df": df,
                # "model_name": "indobenchmark/indobert-lite-base-p2",
                "model_name": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
                # "model_name": "all-MiniLM-L6-v2",
                "batch_size": batch_size,
                "learning_rate": lr,
                "epochs": 100,
                "config_id": idx,
                "best_valid_pearson": max(df_result1['valid_pearson']) if df_result1 is not None and not df_result1.empty else float("-inf"),
                "warmup_ratio": warm_up,
            }

            logging.info(
                f"Running configuration: config_id={idx}, model_name={config['model_name']}"
                f", batch_size={batch_size}, epochs={100}, learning_rate={lr}"
            )
            
            print(
                f"\nRunning configuration: config_id={idx}, model_name={config['model_name']}"
                f", batch_size={batch_size}, epochs={100}, learning_rate={lr}"
            )
            
            try:
                pipeline = SBERTPipeline(config, results, results_epoch, device)
                pipeline.training()

                # Save results
                # Dapatkan root project
                results_path = os.path.join(ROOT_DIR, "experiments/results/results_sbert1.csv")
                results_epoch_path = os.path.join(ROOT_DIR, "experiments/results/results_epoch_sbert1.csv")
                pipeline.save_csv(results, results_path)
                pipeline.save_csv(results_epoch, results_epoch_path)
            except Exception as e:
                logging.error(f"Error in config_id={idx}: {str(e)}")
                print(f"Error in config_id={idx}: {str(e)}")
                torch.cuda.empty_cache()
            finally:
                # Clear GPU memory after every configuration
                del pipeline.model
                del pipeline.optimizer
                torch.cuda.empty_cache()

            idx += 1