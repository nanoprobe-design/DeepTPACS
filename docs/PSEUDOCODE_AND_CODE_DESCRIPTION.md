# DeepTPACS code description and pseudocode

## Recommended location in the manuscript

The material below is a code-grounded description of the released implementation and may be incorporated into the **Methods** or **Supplementary Methods** under a heading such as **DeepTPACS implementation and computational workflow**.

## Short Methods text

DeepTPACS was implemented in Python using PyTorch together with DGL/DGLLife for molecular graph construction and graph neural-network operations. Molecular structures represented as SMILES were converted to bidirectional molecular graphs. For the released model, six selected atom-feature families were concatenated to form a 30-dimensional node representation, while bond descriptors formed a 13-dimensional edge representation. The optimized DeepTPACS architecture used three AttentiveFP graph-neural-network layers, a 300-dimensional learned graph representation, two attentive graph-readout update steps, and zero dropout. The Adam optimizer was used with an initial learning rate of 3.1623 × 10^-4 and L2 weight decay of 1 × 10^-6. Hyperparameters were selected using 30 Hyperopt/TPE evaluations. Model evaluation used shuffled 10-fold cross-validation with random seed 42, a maximum of 300 epochs per fold, batch size 32, early-stopping patience of 50 epochs, learning-rate reduction on plateau, exponential moving averaging (EMA), and late-stage stochastic weight averaging (SWA). The source code, released model checkpoint, model configuration, reference demo, tested environment and reference output are available at https://github.com/nanoprobe-design/DeepTPACS under the MIT License.

## Algorithm 1. Molecular graph generation

```text
Input:
    Molecular records containing SMILES strings and target lg(TPACS) values

For each molecule:
    1. Parse the SMILES string with RDKit/DGLLife.
    2. Convert the molecule to a bidirectional molecular graph.
    3. Select the first six atom-feature families according to FEATURE_ORDER.
    4. Sort the selected feature indices and concatenate their atom descriptors.
    5. Store the resulting 30-dimensional atom feature vector for each atom.
    6. Construct bond features from bond type, conjugation, ring membership,
       and stereochemistry, yielding a 13-dimensional edge representation.
    7. Store the molecular graph and regression target.

During mini-batch collation:
    8. Add graph self-loops.
    9. Batch molecular graphs with DGL.
```

## Algorithm 2. Hyperparameter optimization

```text
Input:
    Molecular graph dataset
    Candidate hyperparameter spaces
    random seed = 42

Candidate DeepTPACS parameters:
    L2 weight decay      = {0, 1e-8, 1e-6, 1e-4}
    learning rate        = {10^-2.5, 10^-3.5, 10^-1.5}
    GNN layers           = {2, 3, 4, 5}
    readout timesteps    = {2, 3, 4, 5}
    graph feature size   = {100, 200, 300}
    dropout              = {0, 0.1, 0.2}

1. Split the dataset into 90% training and 10% validation data
   using ShuffleSplit with random seed 42.
2. Initialize a Hyperopt/TPE search.
3. For trial = 1 to 30:
       a. Select one hyperparameter combination using TPE.
       b. Initialize a DeepTPACS model.
       c. Train with Adam using the selected learning rate and L2 weight decay.
       d. Evaluate the validation subset after each epoch.
       e. Apply ReduceLROnPlateau using validation MAE.
       f. Apply early stopping using validation MAE with patience = 50.
       g. Reload the best checkpoint for the trial.
       h. Calculate validation MSE.
       i. Return validation MSE as the Hyperopt objective.
4. Retain the hyperparameter indices that minimize validation MSE.
5. Resolve the retained indices to their numerical values.

Resolved released DeepTPACS configuration:
    dropout            = 0
    graph feature size = 300
    L2 weight decay    = 1e-6
    learning rate      = 3.1622776601683795e-4
    GNN layers         = 3
    readout timesteps  = 2
```

## Algorithm 3. Ten-fold DeepTPACS training and evaluation

```text
Input:
    Molecular graph dataset
    Optimized DeepTPACS hyperparameters
    random seed = 42

1. Construct KFold(n_splits = 10, shuffle = True, random_state = 42).
2. For each fold k = 1,...,10:
       a. Use nine folds as the training subset and one fold as the held-out subset.
       b. Build mini-batch data loaders (batch size = 32).
       c. Initialize a new DeepTPACS model with:
              node feature size = 30
              edge feature size = 13
              GNN layers = 3
              graph feature size = 300
              readout timesteps = 2
              dropout = 0
       d. Initialize Adam with:
              learning rate = 3.1622776601683795e-4
              weight decay = 1e-6
       e. Initialize ReduceLROnPlateau
              factor = 0.8
              patience = 5
              minimum learning rate = 1e-6.
       f. Initialize early stopping with patience = 50.
       g. Initialize EMA with decay = 0.999.
       h. Initialize SWA; start SWA after 80% of the maximum epoch count.
       i. For epoch = 1,...,300:
              i.   Use L1 loss during the first 50% of training.
              ii.  Use weighted MAE loss after the first 50% of training.
              iii. Perform forward propagation through the AttentiveFP GNN.
              iv.  Compute the loss and update model parameters with Adam.
              v.   Update EMA parameters.
              vi.  Evaluate MSE, MAE, R2 and Pearson correlation.
              vii. Step the learning-rate scheduler using held-out MAE.
              viii.Apply early stopping using held-out MAE.
              ix.  Save the current model when held-out R2 improves.
              x.   Every five epochs, evaluate the EMA model and save it
                   when its held-out R2 improves.
              xi.  After the SWA start epoch, update the SWA model.
              xii. Stop if the early-stopping criterion is met.
       j. If SWA parameters were accumulated, update normalization statistics,
          evaluate the SWA model and retain it if held-out R2 improves.
       k. Load the best saved model for the fold.
       l. Generate predictions for the held-out molecules.
       m. Save fold-level metrics and predictions.
3. Aggregate held-out predictions and fold-level metrics across all 10 folds.
```

## Algorithm 4. Released DeepTPACS inference

```text
Input:
    CSV containing query SMILES strings
    model/trained.pt
    config/DeepTPACS_best_params_index.txt

Output:
    predicted_log10_TPACS
    predicted_TPACS

1. Read the input CSV and identify the SMILES column.
2. Resolve the released model configuration from the committed parameter file.
3. Build the same atom and bond featurizers used for training.
4. For each query molecule:
       a. Parse the SMILES string.
       b. Construct a bidirectional molecular graph.
       c. Generate 30-dimensional node features.
       d. Generate 13-dimensional edge features.
5. Add self-loops and batch molecular graphs.
6. Initialize DeepTPACS with the released architecture:
       GNN layers = 3
       graph feature size = 300
       readout timesteps = 2
       dropout = 0
7. Load the trained checkpoint.
8. For each molecular graph batch:
       a. Update atom representations using AttentiveFP message passing.
       b. Perform two attentive graph-readout updates.
       c. Map the 300-dimensional molecular representation to one scalar output.
9. Report the model output as predicted log10(TPACS).
10. Calculate predicted TPACS = 10^(predicted log10(TPACS)).
11. Write all predictions to the output CSV.
```

## Algorithm 5. Classical machine-learning sample workflow

```text
Input:
    data_process/TPACS_sample_data.csv

1. Read the same public sample CSV used for the demonstration workflow.
2. Parse each SMILES string with RDKit.
3. Generate the requested molecular representation, including:
       RDKit descriptors,
       Morgan fingerprints,
       Daylight fingerprints,
       atom-pair fingerprints,
       or topological-torsion fingerprints.
4. Save the generated descriptor/fingerprint matrices in the generated ml/ workspace.
5. Run data-processing.py to construct the classical-ML input tables.
6. Optimize the selected classical-ML model with model_params_opt.py.
7. Train/evaluate the selected model with ml_train.py.
8. Save model outputs and evaluation results in the generated workspace.
```

## Code availability text

> The DeepTPACS source code, trained model checkpoint, model configuration, tested software environment, numerical reference output, and a small reference demo are publicly available at https://github.com/nanoprobe-design/DeepTPACS under the MIT License. The repository includes instructions for installation, model inference, training, and computational reproduction. A versioned DOI should be cited in the final article if the submitted code release is archived in a persistent repository.

## Manuscript-data note

The committed `TPACS_sample_data.csv` contains 100 sample molecules and is used by both the public GNN example and the classical-ML sample workflow. Classical descriptor/fingerprint matrices are regenerated from these SMILES and therefore do not need to be committed as a separate raw `ml/` dataset. If quantitative results reported in the manuscript were generated from data beyond this public sample, those additional data should be deposited or linked in the Data Availability statement.
