## File organization

Training data is available at https://archive.org/download/stackexchange/codereview.stackexchange.com.7z.

## Experiment replication steps

Step 1: Knowledge guided prompt learning.

- Input: Training data.
- Output: Knowledge guided prompt template.
- Method: GraphCodeBERT, SimCSE. 
- Source code: bert-base-cased2 and simcse in Pretrained_LMs, parser, myTokenizers.py listed in the Code section.
- Step 1.1. Text prompt reconstructs the request text into a generative task.
  - Method: SimCSE.
- Step 1.2. Code prompt refactors the code snippets in text into the form of data flow graph.
  - Method: Tree-sitter, GraphCodeBERT.
- Step 1.3. Matching with the external knowledge base from Wikipedia to get the knowledge guidance.

Step 2: Training. 

- Input: The knowledge guided prompt template (output of Step 1).
- Output: Predicted word list.
- Method: Mask Language Model (MLM).
- Source code: model.py, train.py listed in the Code section. 

Step 3: Answer engineering. 

- Input: Predicted word list.
- Output: Final labels.
- Source code: answer_engineering.py listed in the Code section.

## Code

- Pretrained_LMs comprises two pretrained models BERT and SimCSE. 
- parser comprises several parsers for converting source code written in programming languages into abstract syntax trees (AST). 
- wandb is a visualization tool designed for machine learning that can be used to track, visualize, and share experimental results.
- answer_engineering.py for getting the final labels.
- codelanguage.py for identifing programming languages.
- config.py for configuring the parameters used to set up and run the training and evaluation process of the machine learning model.
- main.py as a master script for our method.
- model.py for implementing Masked Language Model.
- myDataset.py for building the dataset.
- myTokenizers.py for preprocessing.
- test.py for testing.
- test_main.py for the complete training and evaluation process.
- train.py for training.
- utils.py for setting random seeds.

## Experiment environment

- GPU: NVIDIA Titan XP*4
- CPU: IntelR XeonR E5-2650 v4 @2.20GHz
- Memory Capacity: 64G*4
- CUDA Version: 11.0
- Python Version: 3.5
- Pytorch Version: 1.7