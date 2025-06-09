import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, LeaveOneOut
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import seaborn as sns
from types import SimpleNamespace
import pickle
import gzip
from pathlib import Path

class SequentialMovementDataset(Dataset):
    """Dataset for sequential movement classification with small sample sizes"""
    def __init__(self, features, current_labels, terminal_labels, sequence_lengths, augment=True):
        self.features = torch.FloatTensor(features)
        self.current_labels = torch.LongTensor(current_labels)
        self.terminal_labels = torch.LongTensor(terminal_labels)
        self.sequence_lengths = sequence_lengths
        self.augment = augment
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        features = self.features[idx]
        
        # Simple augmentation for small datasets
        if self.augment and self.training:
            # Add small amount of Gaussian noise
            noise = torch.randn_like(features) * 0.01
            features = features + noise
            
            # Random time shift (small)
            if features.size(0) > 10:
                shift = np.random.randint(-2, 3)
                if shift != 0:
                    features = torch.roll(features, shift, dims=0)
        
        return {
            'features': features,
            'current_label': self.current_labels[idx],
            'terminal_label': self.terminal_labels[idx],
            'seq_length': self.sequence_lengths[idx]
        }

class CompactDualLSTM(nn.Module):
    """Compact LSTM model designed for small datasets"""
    def __init__(self, input_size, hidden_size, num_current_classes, num_terminal_classes, dropout=0.3):
        super(CompactDualLSTM, self).__init__()
        
        # Smaller, more regularized architecture for limited data
        self.hidden_size = hidden_size
        
        # Single bidirectional LSTM layer to avoid overfitting
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=0,  # No dropout in LSTM for single layer
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # Shared feature extractor
        self.shared_features = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Classification heads with reduced complexity
        self.current_classifier = nn.Linear(hidden_size, num_current_classes)
        self.terminal_classifier = nn.Linear(hidden_size, num_terminal_classes)
        
        # Initialize weights properly for small datasets
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight' in name:
                    torch.nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    torch.nn.init.zeros_(param)
    
    def forward(self, x, seq_lengths=None):
        batch_size = x.size(0)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)  # [batch_size, seq_len, hidden_size*2]
        
        # Masked attention (ignore padded positions)
        if seq_lengths is not None:
            mask = torch.zeros_like(lstm_out[:, :, 0])  # [batch_size, seq_len]
            for i, length in enumerate(seq_lengths):
                mask[i, :length] = 1
            mask = mask.unsqueeze(-1)  # [batch_size, seq_len, 1]
            lstm_out = lstm_out * mask
        
        # Attention weights
        attention_scores = self.attention(lstm_out)  # [batch_size, seq_len, 1]
        attention_weights = torch.softmax(attention_scores, dim=1)
        
        # Apply attention
        attended_output = torch.sum(attention_weights * lstm_out, dim=1)  # [batch_size, hidden_size*2]
        
        # Shared feature extraction
        shared_features = self.shared_features(attended_output)
        
        # Dual classification
        current_logits = self.current_classifier(shared_features)
        terminal_logits = self.terminal_classifier(shared_features)
        
        return current_logits, terminal_logits, attention_weights

class SequentialMovementPipeline:
    """Pipeline for sequential movement classification with proper sequence understanding"""
    
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.sDats = {}
        
        # Define the movement sequence structure based on your diagram
        self.sequence_definitions = {
            'Reach': ['Reach'],
            'GraspSeq': ['Reach', 'Grasp'],
            'HandtoMouthSeq': ['Reach', 'Grasp', 'Hand to mouth'],
            'InMouthSeq': ['Reach', 'Grasp', 'Hand to mouth', 'Put in mouth'],
            'ChewSeq': ['Reach', 'Grasp', 'Hand to mouth', 'Put in mouth', 'Chew']
        }
        
        # Terminal movements for each sequence
        self.terminal_movements = {
            'Reach': '1ReachSeq',
            'GraspSeq': '2GraspSeq', 
            'HandtoMouthSeq': '3HandToMouthSeq',
            'InMouthSeq': '4PutInMouthSeq',
            'ChewSeq': '5ChewSeq'
        }
        
        # Map sDat names to current movement subcomponents
        self.sdat_to_movement = {
            'reach': '1Reach',
            'grasp': '2Grasp', 
            'hand_to_mouth': '3HandToMouth',
            'in_mouth': '4PutInMouth',
            'chew': '5Chew'  # Adding chew assuming you have 5 sDats
        }
        
    def load_data(self):
        """Load all sDat files"""
        print("Loading sDat files...")
        
        sdat_files = ["reach", "grasp", "hand_to_mouth", "in_mouth"]
        
        for sub in sdat_files:
            file_path = f"{self.data_dir}/sDat_{sub}_alignment.pkl.gz"
            if Path(file_path).exists():
                with gzip.open(file_path, "rb") as f:
                    sDat_raw = pickle.load(f)
                    sDat = SimpleNamespace(**sDat_raw)
                    self.sDats[sub] = sDat
                    
                    print(f"\n=== {sub.upper()} sDat ===")
                    print(f"Trials: {len(sDat.movementCodes)}")
                    print(f"Movement names: {sDat.movementNames}")
                    print(f"Movement codes: {np.unique(sDat.movementCodes, return_counts=True)}")
                    print(f"Features shape: {sDat.features.shape}")
                    print(f"Analysis window: {getattr(sDat, 'analysisWindow', 'Not found')}")
            else:
                print(f"Warning: {file_path} not found")
    
    def analyze_trial_distribution(self):
        """Analyze the distribution of trials across conditions"""
        print("\n=== TRIAL DISTRIBUTION ANALYSIS ===")
        
        for sub_name, sDat in self.sDats.items():
            print(f"\n{sub_name}:")
            
            # Check movement sets
            movement_sets = getattr(sDat, 'movementSets', [[]])
            if movement_sets and len(movement_sets) > 0:
                valid_movement_codes = set(movement_sets[0])
                print(f"Valid movement codes (from movementSets): {valid_movement_codes}")
                
                # Filter trials by movement sets
                trials_in_movement_set = np.isin(sDat.movementCodes, list(valid_movement_codes))
                filtered_codes = sDat.movementCodes[trials_in_movement_set]
                
                print(f"Total trials: {len(sDat.movementCodes)}")
                print(f"Trials in movement set: {len(filtered_codes)}")
                print(f"Excluded trials: {len(sDat.movementCodes) - len(filtered_codes)}")
                
                # Count trials per sequence condition (filtered)
                unique_codes, counts = np.unique(filtered_codes, return_counts=True)
            else:
                print("Warning: No movement sets found, using all trials")
                unique_codes, counts = np.unique(sDat.movementCodes, return_counts=True)
            
            print("Sequence condition distribution (after movement set filtering):")
            for code, count in zip(unique_codes, counts):
                if code < len(sDat.movementNames):
                    sequence_name = sDat.movementNames[code]
                    terminal_movement = self.terminal_movements.get(sequence_name, 'Unknown')
                    print(f"  {sequence_name} (code {code}, terminal: {terminal_movement}): {count} trials")
                else:
                    print(f"  Unknown code {code}: {count} trials")
            
            # Show excluded trials
            if movement_sets and len(movement_sets) > 0:
                excluded_codes = sDat.movementCodes[~trials_in_movement_set]
                if len(excluded_codes) > 0:
                    print("Excluded trials by movement code:")
                    unique_excluded, counts_excluded = np.unique(excluded_codes, return_counts=True)
                    for code, count in zip(unique_excluded, counts_excluded):
                        if code < len(sDat.movementNames):
                            print(f"  {sDat.movementNames[code]} (code {code}): {count} trials")
                        else:
                            print(f"  Unknown code {code}: {count} trials")
    
    def combine_sequential_data(self):
        """Combine data from all sDats with proper current and terminal labels"""
        print("\n=== COMBINING SEQUENTIAL DATA ===")
        
        all_features = []
        all_current_labels = []
        all_terminal_labels = []
        all_seq_lengths = []
        all_sequence_conditions = []
        all_subcomponent_names = []
        
        # Find maximum sequence length for padding
        max_seq_length = 0
        for sub_name, sDat in self.sDats.items():
            analysis_window = getattr(sDat, 'analysisWindow', (0, 71))
            seq_length = analysis_window[1] - analysis_window[0] + 1
            max_seq_length = max(max_seq_length, seq_length)
        
        print(f"Maximum sequence length: {max_seq_length}")
        
        # Process each sDat
        total_trials = 0
        for sub_name, sDat in self.sDats.items():
            current_movement = self.sdat_to_movement[sub_name]
            analysis_window = getattr(sDat, 'analysisWindow', (0, 71))
            seq_length = analysis_window[1] - analysis_window[0] + 1
            
            # Get movement sets for filtering
            movement_sets = getattr(sDat, 'movementSets', [[]])
            if movement_sets and len(movement_sets) > 0:
                valid_movement_codes = set(movement_sets[0])  # Use first movement set
                print(f"\nProcessing {sub_name} ({current_movement}):")
                print(f"  Analysis window: {analysis_window}, Length: {seq_length}")
                print(f"  Valid movement codes: {valid_movement_codes}")
                print(f"  Total trials before filtering: {len(sDat.movementCodes)}")
                
                # Filter trials by movement sets
                trials_in_movement_set = np.isin(sDat.movementCodes, list(valid_movement_codes))
                print(f"  Trials in movement set: {np.sum(trials_in_movement_set)}")
            else:
                print(f"\nProcessing {sub_name} ({current_movement}):")
                print(f"  Warning: No movement sets found, including all trials")
                print(f"  Analysis window: {analysis_window}, Length: {seq_length}")
                valid_movement_codes = set(range(len(sDat.movementNames)))  # Include all if no sets defined
                trials_in_movement_set = np.ones(len(sDat.movementCodes), dtype=bool)
            
            valid_trials = 0
            for trial_idx, (go_time, movement_code) in enumerate(zip(sDat.goTimes, sDat.movementCodes)):
                # Check if trial is in movement set
                if not trials_in_movement_set[trial_idx]:
                    continue
                    
                start_idx = go_time + analysis_window[0]
                end_idx = go_time + analysis_window[1] + 1
                
                if start_idx >= 0 and end_idx <= sDat.features.shape[0] and movement_code < len(sDat.movementNames):
                    # Extract features
                    trial_features = sDat.features[start_idx:end_idx]
                    
                    # Pad to max length
                    if trial_features.shape[0] < max_seq_length:
                        padding = np.zeros((max_seq_length - trial_features.shape[0], trial_features.shape[1]))
                        trial_features = np.vstack([trial_features, padding])
                    
                    # Get sequence condition and terminal movement
                    sequence_condition = sDat.movementNames[movement_code]
                    terminal_movement = self.terminal_movements.get(sequence_condition, sequence_condition)
                    
                    # Store data
                    all_features.append(trial_features)
                    all_current_labels.append(current_movement)
                    all_terminal_labels.append(terminal_movement)
                    all_seq_lengths.append(seq_length)
                    all_sequence_conditions.append(sequence_condition)
                    all_subcomponent_names.append(sub_name)
                    
                    valid_trials += 1
            
            print(f"  Valid trials: {valid_trials}")
            total_trials += valid_trials
        
        # Convert to arrays
        combined_data = {
            'features': np.array(all_features),
            'current_labels': np.array(all_current_labels),
            'terminal_labels': np.array(all_terminal_labels),
            'seq_lengths': np.array(all_seq_lengths),
            'sequence_conditions': np.array(all_sequence_conditions),
            'subcomponent_names': np.array(all_subcomponent_names)
        }
        
        print(f"\n=== COMBINED DATASET SUMMARY ===")
        print(f"Total trials: {total_trials}")
        print(f"Features shape: {combined_data['features'].shape}")
        
        print(f"\nCurrent movement distribution:")
        unique_current, counts_current = np.unique(combined_data['current_labels'], return_counts=True)
        for movement, count in zip(unique_current, counts_current):
            print(f"  {movement}: {count} trials")
        
        print(f"\nTerminal movement distribution:")
        unique_terminal, counts_terminal = np.unique(combined_data['terminal_labels'], return_counts=True)
        for movement, count in zip(unique_terminal, counts_terminal):
            print(f"  {movement}: {count} trials")
        
        print(f"\nSequence condition distribution:")
        unique_seq, counts_seq = np.unique(combined_data['sequence_conditions'], return_counts=True)
        for seq, count in zip(unique_seq, counts_seq):
            print(f"  {seq}: {count} trials")
        
        return combined_data
    
    def create_naive_bayes_model(self, combined_data):
        """Create Naive Bayes model with cross-validation for small datasets"""
        print("\n=== TRAINING NAIVE BAYES MODEL ===")
        
        # Flatten features for Naive Bayes
        X = combined_data['features'].reshape(combined_data['features'].shape[0], -1)
        y_current = combined_data['current_labels'] 
        y_terminal = combined_data['terminal_labels']
        
        # Encode labels
        le_current = LabelEncoder()
        le_terminal = LabelEncoder()
        
        y_current_encoded = le_current.fit_transform(y_current)
        y_terminal_encoded = le_terminal.fit_transform(y_terminal)
        
        # Use stratified k-fold for small datasets
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        current_predictions = np.zeros_like(y_current_encoded)
        terminal_predictions = np.zeros_like(y_terminal_encoded)
        
        print("Performing 5-fold cross-validation...")
        
        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y_current_encoded)):
            print(f"  Fold {fold + 1}/5")
            
            X_train, X_test = X[train_idx], X[test_idx]
            y_current_train, y_current_test = y_current_encoded[train_idx], y_current_encoded[test_idx]
            y_terminal_train, y_terminal_test = y_terminal_encoded[train_idx], y_terminal_encoded[test_idx]
            
            # Train models
            nb_current = GaussianNB()
            nb_terminal = GaussianNB()
            
            nb_current.fit(X_train, y_current_train)
            nb_terminal.fit(X_train, y_terminal_train)
            
            # Predict
            current_predictions[test_idx] = nb_current.predict(X_test)
            terminal_predictions[test_idx] = nb_terminal.predict(X_test)
        
        return {
            'current': {'y_true': y_current_encoded, 'y_pred': current_predictions, 'label_encoder': le_current},
            'terminal': {'y_true': y_terminal_encoded, 'y_pred': terminal_predictions, 'label_encoder': le_terminal}
        }
    
    def tune_lstm_hyperparameters(self, combined_data, epochs=150):
        """Tune LSTM hyperparameters for small datasets based on combined label accuracy"""
        print("\n=== TUNING LSTM HYPERPARAMETERS (OPTIMIZING FOR COMBINED ACCURACY) ===")
        
        # Define hyperparameter search space (focused on small datasets)
        hidden_sizes = [32, 64, 128]  # Start smaller for limited data
        learning_rates = [0.0005, 0.001, 0.002, 0.005]  # Fine-grained learning rate search
        
        # Encode labels
        le_current = LabelEncoder()
        le_terminal = LabelEncoder()
        
        y_current_encoded = le_current.fit_transform(combined_data['current_labels'])
        y_terminal_encoded = le_terminal.fit_transform(combined_data['terminal_labels'])
        
        X = combined_data['features']
        seq_lengths = combined_data['seq_lengths']
        
        # Store results for each hyperparameter combination
        tuning_results = []
        
        # Use a smaller subset for hyperparameter tuning to speed up the process
        if len(X) < 30:
            cv = LeaveOneOut()
            print("Using Leave-One-Out for hyperparameter tuning (very small dataset)")
        else:
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)  # 3-fold for speed
            print("Using 3-fold stratified CV for hyperparameter tuning")
        
        total_combinations = len(hidden_sizes) * len(learning_rates)
        combination_count = 0
        
        for hidden_size in hidden_sizes:
            for learning_rate in learning_rates:
                combination_count += 1
                print(f"\nTesting combination {combination_count}/{total_combinations}: "
                    f"hidden_size={hidden_size}, learning_rate={learning_rate}")
                
                fold_combined_accuracies = []
                fold_current_accuracies = []
                fold_terminal_accuracies = []
                
                for fold, (train_idx, val_idx) in enumerate(cv.split(X, y_current_encoded)):
                    X_train, X_val = X[train_idx], X[val_idx]
                    y_current_train, y_current_val = y_current_encoded[train_idx], y_current_encoded[val_idx]
                    y_terminal_train, y_terminal_val = y_terminal_encoded[train_idx], y_terminal_encoded[val_idx]
                    seq_train, seq_val = seq_lengths[train_idx], seq_lengths[val_idx]
                    
                    # Create datasets
                    train_dataset = SequentialMovementDataset(X_train, y_current_train, y_terminal_train, seq_train, augment=True)
                    val_dataset = SequentialMovementDataset(X_val, y_current_val, y_terminal_val, seq_val, augment=False)
                    
                    train_loader = DataLoader(train_dataset, batch_size=min(16, len(train_dataset)), shuffle=True)
                    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)
                    
                    # Initialize model
                    input_size = X.shape[2]
                    num_current_classes = len(np.unique(y_current_encoded))
                    num_terminal_classes = len(np.unique(y_terminal_encoded))
                    
                    model = CompactDualLSTM(
                        input_size=input_size,
                        hidden_size=hidden_size,
                        num_current_classes=num_current_classes,
                        num_terminal_classes=num_terminal_classes,
                        dropout=0.3
                    )
                    
                    # Training setup
                    criterion = nn.CrossEntropyLoss()
                    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.5)
                    
                    # Training loop with early stopping based on combined accuracy
                    best_combined_acc = 0.0
                    best_model_state = None
                    patience_counter = 0
                    max_patience = 30
                    
                    for epoch in range(epochs):
                        model.train()
                        train_dataset.training = True
                        
                        for batch in train_loader:
                            optimizer.zero_grad()
                            
                            current_logits, terminal_logits, _ = model(batch['features'], batch['seq_length'])
                            
                            current_loss = criterion(current_logits, batch['current_label'])
                            terminal_loss = criterion(terminal_logits, batch['terminal_label'])
                            total_loss = current_loss + terminal_loss
                            
                            total_loss.backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                            optimizer.step()
                        
                        # Validation with combined accuracy
                        model.eval()
                        train_dataset.training = False
                        val_loss = 0
                        
                        with torch.no_grad():
                            all_current_preds = []
                            all_terminal_preds = []
                            all_current_true = []
                            all_terminal_true = []
                            
                            for batch in val_loader:
                                current_logits, terminal_logits, _ = model(batch['features'], batch['seq_length'])
                                
                                # Calculate loss
                                current_loss = criterion(current_logits, batch['current_label'])
                                terminal_loss = criterion(terminal_logits, batch['terminal_label'])
                                total_loss = current_loss + terminal_loss
                                val_loss += total_loss.item()
                                
                                # Get predictions
                                current_preds = torch.argmax(current_logits, dim=1).cpu().numpy()
                                terminal_preds = torch.argmax(terminal_logits, dim=1).cpu().numpy()
                                
                                all_current_preds.extend(current_preds)
                                all_terminal_preds.extend(terminal_preds)
                                all_current_true.extend(batch['current_label'].cpu().numpy())
                                all_terminal_true.extend(batch['terminal_label'].cpu().numpy())
                            
                            # Calculate combined accuracy (both predictions must be correct)
                            combined_correct = np.array([
                                (curr_pred == curr_true) and (term_pred == term_true)
                                for curr_pred, curr_true, term_pred, term_true in 
                                zip(all_current_preds, all_current_true, all_terminal_preds, all_terminal_true)
                            ])
                            combined_acc = np.mean(combined_correct)
                            
                            # Also calculate individual accuracies for monitoring
                            current_acc = accuracy_score(all_current_true, all_current_preds)
                            terminal_acc = accuracy_score(all_terminal_true, all_terminal_preds)
                        
                        scheduler.step(val_loss)
                        
                        # Early stopping based on combined accuracy
                        if combined_acc > best_combined_acc:
                            best_combined_acc = combined_acc
                            best_model_state = model.state_dict().copy()
                            patience_counter = 0
                        else:
                            patience_counter += 1
                            if patience_counter >= max_patience:
                                break
                    
                    # Load best model and get final validation metrics
                    if best_model_state is not None:
                        model.load_state_dict(best_model_state)
                    
                    model.eval()
                    with torch.no_grad():
                        all_current_preds = []
                        all_terminal_preds = []
                        all_current_true = []
                        all_terminal_true = []
                        
                        for batch in val_loader:
                            current_logits, terminal_logits, _ = model(batch['features'], batch['seq_length'])
                            
                            current_preds = torch.argmax(current_logits, dim=1).cpu().numpy()
                            terminal_preds = torch.argmax(terminal_logits, dim=1).cpu().numpy()
                            
                            all_current_preds.extend(current_preds)
                            all_terminal_preds.extend(terminal_preds)
                            all_current_true.extend(batch['current_label'].cpu().numpy())
                            all_terminal_true.extend(batch['terminal_label'].cpu().numpy())
                        
                        # Calculate final metrics
                        combined_correct = np.array([
                            (curr_pred == curr_true) and (term_pred == term_true)
                            for curr_pred, curr_true, term_pred, term_true in 
                            zip(all_current_preds, all_current_true, all_terminal_preds, all_terminal_true)
                        ])
                        final_combined_acc = np.mean(combined_correct)
                        final_current_acc = accuracy_score(all_current_true, all_current_preds)
                        final_terminal_acc = accuracy_score(all_terminal_true, all_terminal_preds)
                        
                        fold_combined_accuracies.append(final_combined_acc)
                        fold_current_accuracies.append(final_current_acc)
                        fold_terminal_accuracies.append(final_terminal_acc)
                
                # Store results
                mean_combined_acc = np.mean(fold_combined_accuracies)
                std_combined_acc = np.std(fold_combined_accuracies)
                mean_current_acc = np.mean(fold_current_accuracies)
                mean_terminal_acc = np.mean(fold_terminal_accuracies)
                
                tuning_results.append({
                    'hidden_size': hidden_size,
                    'learning_rate': learning_rate,
                    'mean_combined_accuracy': mean_combined_acc,
                    'std_combined_accuracy': std_combined_acc,
                    'mean_current_accuracy': mean_current_acc,
                    'mean_terminal_accuracy': mean_terminal_acc,
                    'fold_combined_scores': fold_combined_accuracies,
                    'fold_current_scores': fold_current_accuracies,
                    'fold_terminal_scores': fold_terminal_accuracies
                })
                
                print(f"  Combined accuracy: {mean_combined_acc:.4f} ± {std_combined_acc:.4f}")
                print(f"  Current accuracy: {mean_current_acc:.4f}")
                print(f"  Terminal accuracy: {mean_terminal_acc:.4f}")
        
        # Find best hyperparameters based on combined accuracy
        best_result = max(tuning_results, key=lambda x: x['mean_combined_accuracy'])
        print(f"\n=== BEST HYPERPARAMETERS (BASED ON COMBINED ACCURACY) ===")
        print(f"Hidden size: {best_result['hidden_size']}")
        print(f"Learning rate: {best_result['learning_rate']}")
        print(f"Best combined accuracy: {best_result['mean_combined_accuracy']:.4f} ± {best_result['std_combined_accuracy']:.4f}")
        print(f"Corresponding current accuracy: {best_result['mean_current_accuracy']:.4f}")
        print(f"Corresponding terminal accuracy: {best_result['mean_terminal_accuracy']:.4f}")
        
        return tuning_results, best_result
    
    def plot_hyperparameter_tuning_results(self, tuning_results, save_dir):
        """Plot hyperparameter tuning results for combined accuracy optimization"""
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        fig, axes = plt.subplots(3, 2, figsize=(16, 18))
        
        # Extract data for plotting
        hidden_sizes = sorted(list(set([r['hidden_size'] for r in tuning_results])))
        learning_rates = sorted(list(set([r['learning_rate'] for r in tuning_results])))
        
        # Create heatmap data for combined accuracy
        heatmap_combined = np.zeros((len(hidden_sizes), len(learning_rates)))
        heatmap_combined_std = np.zeros((len(hidden_sizes), len(learning_rates)))
        heatmap_current = np.zeros((len(hidden_sizes), len(learning_rates)))
        heatmap_terminal = np.zeros((len(hidden_sizes), len(learning_rates)))
        
        for result in tuning_results:
            i = hidden_sizes.index(result['hidden_size'])
            j = learning_rates.index(result['learning_rate'])
            heatmap_combined[i, j] = result['mean_combined_accuracy']
            heatmap_combined_std[i, j] = result['std_combined_accuracy']
            heatmap_current[i, j] = result['mean_current_accuracy']
            heatmap_terminal[i, j] = result['mean_terminal_accuracy']
        
        # Plot 1: Heatmap of mean combined accuracies
        sns.heatmap(heatmap_combined, 
                    xticklabels=[f'{lr:.4f}' for lr in learning_rates],
                    yticklabels=[f'{hs}' for hs in hidden_sizes],
                    annot=True, fmt='.3f', cmap='viridis',
                    ax=axes[0,0])
        axes[0,0].set_title('Mean Combined Accuracy')
        axes[0,0].set_xlabel('Learning Rate')
        axes[0,0].set_ylabel('Hidden Size')
        
        # Plot 2: Heatmap of standard deviations for combined accuracy
        sns.heatmap(heatmap_combined_std,
                    xticklabels=[f'{lr:.4f}' for lr in learning_rates],
                    yticklabels=[f'{hs}' for hs in hidden_sizes],
                    annot=True, fmt='.3f', cmap='viridis',
                    ax=axes[0,1])
        axes[0,1].set_title('Standard Deviation of Combined Accuracy')
        axes[0,1].set_xlabel('Learning Rate')
        axes[0,1].set_ylabel('Hidden Size')
        
        # Plot 3: Current movement accuracy heatmap
        sns.heatmap(heatmap_current,
                    xticklabels=[f'{lr:.4f}' for lr in learning_rates],
                    yticklabels=[f'{hs}' for hs in hidden_sizes],
                    annot=True, fmt='.3f', cmap='Blues',
                    ax=axes[1,0])
        axes[1,0].set_title('Mean Current Movement Accuracy')
        axes[1,0].set_xlabel('Learning Rate')
        axes[1,0].set_ylabel('Hidden Size')
        
        # Plot 4: Terminal movement accuracy heatmap
        sns.heatmap(heatmap_terminal,
                    xticklabels=[f'{lr:.4f}' for lr in learning_rates],
                    yticklabels=[f'{hs}' for hs in hidden_sizes],
                    annot=True, fmt='.3f', cmap='Greens',
                    ax=axes[1,1])
        axes[1,1].set_title('Mean Terminal Movement Accuracy')
        axes[1,1].set_xlabel('Learning Rate')
        axes[1,1].set_ylabel('Hidden Size')
        
        # Plot 5: Combined accuracy vs Hidden Size (averaged over learning rates)
        hidden_size_means = []
        hidden_size_stds = []
        hidden_size_current = []
        hidden_size_terminal = []
        
        for hs in hidden_sizes:
            combined_accs = [r['mean_combined_accuracy'] for r in tuning_results if r['hidden_size'] == hs]
            current_accs = [r['mean_current_accuracy'] for r in tuning_results if r['hidden_size'] == hs]
            terminal_accs = [r['mean_terminal_accuracy'] for r in tuning_results if r['hidden_size'] == hs]
            
            hidden_size_means.append(np.mean(combined_accs))
            hidden_size_stds.append(np.std(combined_accs))
            hidden_size_current.append(np.mean(current_accs))
            hidden_size_terminal.append(np.mean(terminal_accs))
        
        # Plot all three metrics
        x = np.arange(len(hidden_sizes))
        width = 0.25
        
        axes[2,0].bar(x - width, hidden_size_means, width, yerr=hidden_size_stds, 
                    label='Combined', capsize=5, alpha=0.8, color='purple')
        axes[2,0].bar(x, hidden_size_current, width, 
                    label='Current', alpha=0.8, color='blue')
        axes[2,0].bar(x + width, hidden_size_terminal, width, 
                    label='Terminal', alpha=0.8, color='green')
        
        axes[2,0].set_title('Accuracy vs Hidden Size')
        axes[2,0].set_xlabel('Hidden Size')
        axes[2,0].set_ylabel('Mean Accuracy')
        axes[2,0].set_xticks(x)
        axes[2,0].set_xticklabels(hidden_sizes)
        axes[2,0].legend()
        axes[2,0].grid(True, alpha=0.3)
        
        # Plot 6: Combined accuracy vs Learning Rate (averaged over hidden sizes)
        lr_means = []
        lr_stds = []
        lr_current = []
        lr_terminal = []
        
        for lr in learning_rates:
            combined_accs = [r['mean_combined_accuracy'] for r in tuning_results if r['learning_rate'] == lr]
            current_accs = [r['mean_current_accuracy'] for r in tuning_results if r['learning_rate'] == lr]
            terminal_accs = [r['mean_terminal_accuracy'] for r in tuning_results if r['learning_rate'] == lr]
            
            lr_means.append(np.mean(combined_accs))
            lr_stds.append(np.std(combined_accs))
            lr_current.append(np.mean(current_accs))
            lr_terminal.append(np.mean(terminal_accs))
        
        x_lr = np.arange(len(learning_rates))
        
        axes[2,1].bar(x_lr - width, lr_means, width, yerr=lr_stds, 
                    label='Combined', capsize=5, alpha=0.8, color='purple')
        axes[2,1].bar(x_lr, lr_current, width, 
                    label='Current', alpha=0.8, color='blue')
        axes[2,1].bar(x_lr + width, lr_terminal, width, 
                    label='Terminal', alpha=0.8, color='green')
        
        axes[2,1].set_title('Accuracy vs Learning Rate')
        axes[2,1].set_xlabel('Learning Rate')
        axes[2,1].set_ylabel('Mean Accuracy')
        axes[2,1].set_xticks(x_lr)
        axes[2,1].set_xticklabels([f'{lr:.4f}' for lr in learning_rates])
        axes[2,1].legend()
        axes[2,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/hyperparameter_tuning_results_combined.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print detailed results table
        print("\n=== DETAILED HYPERPARAMETER TUNING RESULTS (OPTIMIZED FOR COMBINED ACCURACY) ===")
        print("Hidden Size | Learning Rate | Combined Acc | Std Combined | Current Acc | Terminal Acc")
        print("-" * 90)
        
        # Sort by combined accuracy for easier reading
        sorted_results = sorted(tuning_results, key=lambda x: x['mean_combined_accuracy'], reverse=True)
        for result in sorted_results:
            print(f"{result['hidden_size']:11d} | {result['learning_rate']:13.4f} | "
                f"{result['mean_combined_accuracy']:12.4f} | {result['std_combined_accuracy']:12.4f} | "
                f"{result['mean_current_accuracy']:11.4f} | {result['mean_terminal_accuracy']:12.4f}")
        
        # Find configurations where combined accuracy is particularly good
        print("\n=== TOP 3 CONFIGURATIONS BY COMBINED ACCURACY ===")
        for i, result in enumerate(sorted_results[:3]):
            print(f"\n{i+1}. Hidden Size: {result['hidden_size']}, Learning Rate: {result['learning_rate']}")
            print(f"   Combined: {result['mean_combined_accuracy']:.4f} ± {result['std_combined_accuracy']:.4f}")
            print(f"   Current: {result['mean_current_accuracy']:.4f}, Terminal: {result['mean_terminal_accuracy']:.4f}")
            print(f"   Fold scores: {[f'{s:.3f}' for s in result['fold_combined_scores']]}")
    


    def create_lstm_model(self, combined_data, hidden_size=64, learning_rate=0.001, epochs=200):
        """Create LSTM model optimized for small datasets with specified hyperparameters"""
        print(f"\n=== TRAINING LSTM MODEL ===")
        print(f"Using hyperparameters: hidden_size={hidden_size}, learning_rate={learning_rate}")
        
        # Encode labels
        le_current = LabelEncoder()
        le_terminal = LabelEncoder()
        
        y_current_encoded = le_current.fit_transform(combined_data['current_labels'])
        y_terminal_encoded = le_terminal.fit_transform(combined_data['terminal_labels'])
        
        X = combined_data['features']
        seq_lengths = combined_data['seq_lengths']
        
        # Use Leave-One-Out or stratified k-fold for very small datasets
        if len(X) < 50:
            print("Using Leave-One-Out cross-validation for very small dataset")
            cv = LeaveOneOut()
        else:
            print("Using 5-fold stratified cross-validation")
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        all_current_preds = []
        all_terminal_preds = []
        all_current_true = []
        all_terminal_true = []
        
        fold = 0
        for train_idx, test_idx in cv.split(X, y_current_encoded):
            fold += 1
            print(f"  Fold {fold}")
            
            X_train, X_test = X[train_idx], X[test_idx]
            y_current_train, y_current_test = y_current_encoded[train_idx], y_current_encoded[test_idx]
            y_terminal_train, y_terminal_test = y_terminal_encoded[train_idx], y_terminal_encoded[test_idx]
            seq_train, seq_test = seq_lengths[train_idx], seq_lengths[test_idx]
            
            # Create datasets with augmentation for training
            train_dataset = SequentialMovementDataset(X_train, y_current_train, y_terminal_train, seq_train, augment=True)
            test_dataset = SequentialMovementDataset(X_test, y_current_test, y_terminal_test, seq_test, augment=False)
            
            train_loader = DataLoader(train_dataset, batch_size=min(16, len(train_dataset)), shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
            
            # Initialize model with optimized hyperparameters
            input_size = X.shape[2]
            num_current_classes = len(np.unique(y_current_encoded))
            num_terminal_classes = len(np.unique(y_terminal_encoded))
            
            model = CompactDualLSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_current_classes=num_current_classes,
                num_terminal_classes=num_terminal_classes,
                dropout=0.3
            )
            
            # Training setup with optimized learning rate
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)
            
            # Training loop with early stopping
            best_loss = float('inf')
            patience_counter = 0
            max_patience = 50
            
            for epoch in range(epochs):
                model.train()
                train_dataset.training = True
                
                train_loss = 0
                for batch in train_loader:
                    optimizer.zero_grad()
                    
                    current_logits, terminal_logits, _ = model(batch['features'], batch['seq_length'])
                    
                    current_loss = criterion(current_logits, batch['current_label'])
                    terminal_loss = criterion(terminal_logits, batch['terminal_label'])
                    total_loss = current_loss + terminal_loss
                    
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    train_loss += total_loss.item()
                
                # Validation
                model.eval()
                train_dataset.training = False
                val_loss = 0
                
                with torch.no_grad():
                    for batch in test_loader:
                        current_logits, terminal_logits, _ = model(batch['features'], batch['seq_length'])
                        current_loss = criterion(current_logits, batch['current_label'])
                        terminal_loss = criterion(terminal_logits, batch['terminal_label'])
                        total_loss = current_loss + terminal_loss
                        val_loss += total_loss.item()
                
                scheduler.step(val_loss)
                
                # Early stopping
                if val_loss < best_loss:
                    best_loss = val_loss
                    patience_counter = 0
                    # Save best model state
                    best_model_state = model.state_dict().copy()
                else:
                    patience_counter += 1
                    if patience_counter >= max_patience:
                        print(f"    Early stopping at epoch {epoch}")
                        break
            
            # Load best model and get predictions
            model.load_state_dict(best_model_state)
            model.eval()
            
            with torch.no_grad():
                for batch in test_loader:
                    current_logits, terminal_logits, _ = model(batch['features'], batch['seq_length'])
                    current_preds = torch.argmax(current_logits, dim=1).cpu().numpy()
                    terminal_preds = torch.argmax(terminal_logits, dim=1).cpu().numpy()
                    
                    all_current_preds.extend(current_preds)
                    all_terminal_preds.extend(terminal_preds)
                    all_current_true.extend(batch['current_label'].cpu().numpy())
                    all_terminal_true.extend(batch['terminal_label'].cpu().numpy())
        
        return {
            'current': {'y_true': all_current_true, 'y_pred': all_current_preds, 'label_encoder': le_current},
            'terminal': {'y_true': all_terminal_true, 'y_pred': all_terminal_preds, 'label_encoder': le_terminal}
        }
    
    def plot_confusion_matrices(self, nb_results, lstm_results, save_dir, model_name="LSTM"):
        """Plot confusion matrices with proper label names including combined predictions"""
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        # Create larger figure to accommodate combined confusion matrices
        fig, axes = plt.subplots(3
                                 , 2, figsize=(15, 12), sharey="row")
        fig.subplots_adjust(bottom=0.25)
        
        # Get label names
        current_labels = nb_results['current']['label_encoder'].classes_
        terminal_labels = nb_results['terminal']['label_encoder'].classes_
        
        # === INDIVIDUAL CONFUSION MATRICES ===
        
        # Naive Bayes - Current Movement
        cm_nb_current = confusion_matrix(nb_results['current']['y_true'], nb_results['current']['y_pred'])
        cm_nb_current_norm = cm_nb_current.astype('float') / cm_nb_current.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(cm_nb_current_norm, annot=True, fmt='.2f',annot_kws={"size": 8},  cmap='Blues', ax=axes[0,0],
                   xticklabels=current_labels, yticklabels=current_labels, vmin= 0, vmax=1)
        axes[0,0].set_title('Naive Bayes - Current Movement Subcomponent', fontsize=14)
        axes[0,0].set_xlabel('Predicted', fontsize=12)
        axes[0,0].set_ylabel('Actual', fontsize=12)
        axes[0,0].set_xticklabels(axes[0,0].get_xticklabels(), rotation=15)

        
        # Naive Bayes - Terminal Movement
        cm_nb_terminal = confusion_matrix(nb_results['terminal']['y_true'], nb_results['terminal']['y_pred'])
        cm_nb_terminal_norm = cm_nb_terminal.astype('float') / cm_nb_terminal.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(cm_nb_terminal_norm, annot=True, fmt='.2f',annot_kws={"size": 8},  cmap='Greens', ax=axes[1,0], vmin= 0, vmax=1,
                   xticklabels=terminal_labels, yticklabels=terminal_labels)
        axes[1,0].set_title('Naive Bayes - Sequential Condition', fontsize=14)
        axes[1,0].set_xlabel('Predicted', fontsize=12)
        axes[1,0].set_ylabel('Actual', fontsize=12)
        axes[1,0].set_xticklabels(axes[1,0].get_xticklabels(), rotation=15)

        # LSTM - Current Movement
        cm_lstm_current = confusion_matrix(lstm_results['current']['y_true'], lstm_results['current']['y_pred'])
        cm_lstm_current_norm = cm_lstm_current.astype('float') / cm_lstm_current.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(cm_lstm_current_norm, annot=True, fmt='.2f', annot_kws={"size": 8}, cmap='Blues', ax=axes[0,1], vmin= 0, vmax=1,
                   xticklabels=current_labels, yticklabels=current_labels)
        axes[0,1].set_title(f'LSTM - Current Movement Subcomponent', fontsize=14)
        axes[0,1].set_xlabel('Predicted', fontsize=12)
        axes[0,1].set_ylabel('Actual', fontsize=12)
        axes[0,1].set_xticklabels(axes[0,0].get_xticklabels(), rotation=15)

        
        # LSTM - Terminal Movement
        cm_lstm_terminal = confusion_matrix(lstm_results['terminal']['y_true'], lstm_results['terminal']['y_pred'])
        cm_lstm_terminal_norm = cm_lstm_terminal.astype('float') / cm_lstm_terminal.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(cm_lstm_terminal_norm, annot=True, fmt='.2f', annot_kws={"size": 8}, cmap='Greens', ax=axes[1,1], vmin= 0, vmax=1,
                   xticklabels=terminal_labels, yticklabels=terminal_labels)
        axes[1,1].set_title(f'LSTM - Movement Sequence', fontsize=14)
        axes[1,1].set_xlabel('Predicted', fontsize=12)
        axes[1,1].set_ylabel('Actual', fontsize=12)
        axes[1,1].set_xticklabels(axes[1,1].get_xticklabels(), rotation=15)
        
        # === COMBINED CONFUSION MATRICES ===
        
        # Create combined labels (Current + Terminal)
        def create_combined_predictions(current_true, current_pred, terminal_true, terminal_pred, 
                                      current_encoder, terminal_encoder):
            """Create combined predictions and ground truth"""
            # Convert back to string labels
            current_true_labels = current_encoder.inverse_transform(current_true)
            current_pred_labels = current_encoder.inverse_transform(current_pred)
            terminal_true_labels = terminal_encoder.inverse_transform(terminal_true)
            terminal_pred_labels = terminal_encoder.inverse_transform(terminal_pred)
            
            # Create combined labels
            combined_true = [f"{curr}→{term}" for curr, term in zip(current_true_labels, terminal_true_labels)]
            combined_pred = [f"{curr}→{term}" for curr, term in zip(current_pred_labels, terminal_pred_labels)]
            
            return combined_true, combined_pred
        
        # Naive Bayes Combined
        nb_combined_true, nb_combined_pred = create_combined_predictions(
            nb_results['current']['y_true'], nb_results['current']['y_pred'],
            nb_results['terminal']['y_true'], nb_results['terminal']['y_pred'],
            nb_results['current']['label_encoder'], nb_results['terminal']['label_encoder']
        )
        
        # Get unique combined labels for consistent ordering
        all_combined_labels = sorted(list(set(nb_combined_true + nb_combined_pred)))
        
        cm_nb_combined = confusion_matrix(nb_combined_true, nb_combined_pred, labels=all_combined_labels)
        cm_nb_combined_norm = cm_nb_combined.astype('float') / (cm_nb_combined.sum(axis=1)[:, np.newaxis] + 1e-8)
        
        # Plot with smaller font for combined labels
        sns.heatmap(cm_nb_combined_norm, annot=True, fmt='.2f', cmap='Reds', annot_kws={"size": 8}, ax=axes[2,0], vmin= 0, vmax=1,
                   xticklabels=all_combined_labels, yticklabels=all_combined_labels)
        axes[2,0].set_title('Naive Bayes - Combined (Current→Sequence)', fontsize=14)
        axes[2,0].set_xlabel('Predicted', fontsize=12)
        axes[2,0].set_ylabel('Actual', fontsize=12)
        axes[2,0].tick_params(axis='both', which='major', labelsize=10)

        
        # LSTM Combined
        lstm_combined_true, lstm_combined_pred = create_combined_predictions(
            lstm_results['current']['y_true'], lstm_results['current']['y_pred'],
            lstm_results['terminal']['y_true'], lstm_results['terminal']['y_pred'],
            lstm_results['current']['label_encoder'], lstm_results['terminal']['label_encoder']
        )
        
        cm_lstm_combined = confusion_matrix(lstm_combined_true, lstm_combined_pred, labels=all_combined_labels)
        cm_lstm_combined_norm = cm_lstm_combined.astype('float') / (cm_lstm_combined.sum(axis=1)[:, np.newaxis] + 1e-8)
        
        sns.heatmap(cm_lstm_combined_norm, annot=True, fmt='.2f', annot_kws={"size": 8}, cmap='Reds', ax=axes[2,1], vmin= 0, vmax=1,
                   xticklabels=all_combined_labels, yticklabels=all_combined_labels)
        axes[2,1].set_title(f'LSTM - Combined (Current→Sequence)', fontsize=14)
        axes[2,1].set_xlabel('Predicted', fontsize=12)
        axes[2,1].set_ylabel('Actual', fontsize=12)
        axes[2,1].tick_params(axis='both', which='major', labelsize=10)

        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/confusion_matrices_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Calculate and print all accuracies
        nb_current_acc = np.mean(nb_results['current']['y_true'] == nb_results['current']['y_pred'])
        nb_terminal_acc = np.mean(nb_results['terminal']['y_true'] == nb_results['terminal']['y_pred'])
        lstm_current_acc = np.mean(lstm_results['current']['y_true'] == lstm_results['current']['y_pred'])
        lstm_terminal_acc = np.mean(lstm_results['terminal']['y_true'] == lstm_results['terminal']['y_pred'])
        
        # Combined accuracies (both predictions must be correct)
        nb_combined_acc = np.mean([true == pred for true, pred in zip(nb_combined_true, nb_combined_pred)])
        lstm_combined_acc = np.mean([true == pred for true, pred in zip(lstm_combined_true, lstm_combined_pred)])
        
        print(f"\n=== ACCURACY SUMMARY ===")
        print(f"Naive Bayes - Current Movement: {nb_current_acc:.3f}")
        print(f"Naive Bayes - Terminal Movement: {nb_terminal_acc:.3f}")
        print(f"Naive Bayes - Combined (Both Correct): {nb_combined_acc:.3f}")
        print(f"LSTM - Current Movement: {lstm_current_acc:.3f}")
        print(f"LSTM - Terminal Movement: {lstm_terminal_acc:.3f}")
        print(f"LSTM - Combined (Both Correct): {lstm_combined_acc:.3f}")
        
        print(f"\n=== COMBINED PREDICTION IMPROVEMENT ===")
        print(f"Combined Accuracy Improvement: {lstm_combined_acc - nb_combined_acc:+.3f}")
        print(f"Relative Improvement: {((lstm_combined_acc - nb_combined_acc) / nb_combined_acc * 100):+.1f}%")
        
        return {
            'nb_combined_accuracy': nb_combined_acc,
            'lstm_combined_accuracy': lstm_combined_acc,
            'combined_labels': all_combined_labels,
            'nb_combined_true': nb_combined_true,
            'nb_combined_pred': nb_combined_pred,
            'lstm_combined_true': lstm_combined_true,
            'lstm_combined_pred': lstm_combined_pred
        }
    
    def plot_sequence_analysis(self, combined_data, save_dir):
        """Plot sequence-specific analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Trial distribution across sequences and subcomponents
        sequence_counts = {}
        for seq in np.unique(combined_data['sequence_conditions']):
            sequence_counts[seq] = {}
            for sub in np.unique(combined_data['subcomponent_names']):
                mask = (combined_data['sequence_conditions'] == seq) & (combined_data['subcomponent_names'] == sub)
                sequence_counts[seq][sub] = np.sum(mask)
        
        # Create heatmap of trial counts
        sequences = list(sequence_counts.keys())
        subcomponents = list(sequence_counts[sequences[0]].keys())
        count_matrix = np.array([[sequence_counts[seq][sub] for sub in subcomponents] for seq in sequences])
        
        sns.heatmap(count_matrix, annot=True, fmt='d', cmap='YlOrRd', ax=axes[0,0],
                   xticklabels=subcomponents, yticklabels=sequences)
        axes[0,0].set_title('Trial Distribution: Sequences × Subcomponents')
        axes[0,0].set_xlabel('Movement Subcomponent')
        axes[0,0].set_ylabel('Sequence Condition')
        
        # 2. Feature variance across subcomponents
        feature_vars = []
        subcomp_labels = []
        for sub in np.unique(combined_data['subcomponent_names']):
            mask = combined_data['subcomponent_names'] == sub
            sub_features = combined_data['features'][mask]
            # Calculate variance across trials for each feature dimension
            var_per_feature = np.var(sub_features.reshape(sub_features.shape[0], -1), axis=0)
            feature_vars.append(np.mean(var_per_feature))
            subcomp_labels.append(sub)
        
        axes[0,1].bar(subcomp_labels, feature_vars, color='skyblue')
        axes[0,1].set_title('Average Feature Variance by Subcomponent')
        axes[0,1].set_xlabel('Movement Subcomponent')
        axes[0,1].set_ylabel('Average Feature Variance')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # 3. Sequence length distribution
        seq_length_data = []
        seq_length_labels = []
        for sub in np.unique(combined_data['subcomponent_names']):
            mask = combined_data['subcomponent_names'] == sub
            lengths = combined_data['seq_lengths'][mask]
            seq_length_data.append(lengths)
            seq_length_labels.append(sub)
        
        axes[1,0].boxplot(seq_length_data, tick_labels=seq_length_labels)
        axes[1,0].set_title('Sequence Length Distribution by Subcomponent')
        axes[1,0].set_xlabel('Movement Subcomponent')
        axes[1,0].set_ylabel('Sequence Length (time steps)')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # 4. Terminal movement prediction difficulty
        terminal_counts = {}
        for current in np.unique(combined_data['current_labels']):
            terminal_counts[current] = {}
            mask = combined_data['current_labels'] == current
            terminals = combined_data['terminal_labels'][mask]
            for terminal in np.unique(terminals):
                terminal_counts[current][terminal] = np.sum(terminals == terminal)
        
        # Create confusion-style matrix for current→terminal mapping
        current_movements = list(terminal_counts.keys())
        all_terminals = set()
        for terminals in terminal_counts.values():
            all_terminals.update(terminals.keys())
        all_terminals = sorted(list(all_terminals))
        
        mapping_matrix = np.zeros((len(current_movements), len(all_terminals)))
        for i, current in enumerate(current_movements):
            for j, terminal in enumerate(all_terminals):
                mapping_matrix[i, j] = terminal_counts[current].get(terminal, 0)
        
        # Normalize by row
        mapping_matrix = mapping_matrix / (mapping_matrix.sum(axis=1, keepdims=True) + 1e-8)
        
        sns.heatmap(mapping_matrix, annot=True, fmt='.2f', cmap='Blues', ax=axes[1,1],
                   xticklabels=all_terminals, yticklabels=current_movements)
        axes[1,1].set_title('Current → Terminal Movement Mapping')
        axes[1,1].set_xlabel('Terminal Movement')
        axes[1,1].set_ylabel('Current Movement')
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/sequence_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_model_comparison(self, nb_results, lstm_results, combined_results, save_dir, model_name="LSTM"):
        """Compare model performance across different metrics including combined predictions"""
        from sklearn.metrics import precision_recall_fscore_support, accuracy_score
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
        # Calculate metrics for both tasks and models
        models = ['Naive Bayes', model_name]
        tasks = ['Current Movement', 'Terminal Movement', 'Combined (Both Correct)']
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        results_data = {}
        for task in tasks:
            results_data[task] = {}
            for model in models:
                if task == 'Combined (Both Correct)':
                    # Combined accuracy is already calculated
                    if model == 'Naive Bayes':
                        acc = combined_results['nb_combined_accuracy']
                        # For combined predictions, calculate other metrics
                        y_true = [1 if true == pred else 0 for true, pred in 
                                 zip(combined_results['nb_combined_true'], combined_results['nb_combined_pred'])]
                        y_pred = [1] * len(y_true)  # All predictions treated as "attempted predictions"
                        prec, rec, f1, _ = precision_recall_fscore_support([1]*len(y_true), y_true, average='weighted', zero_division=0)
                    else:
                        acc = combined_results['lstm_combined_accuracy']
                        y_true = [1 if true == pred else 0 for true, pred in 
                                 zip(combined_results['lstm_combined_true'], combined_results['lstm_combined_pred'])]
                        y_pred = [1] * len(y_true)
                        prec, rec, f1, _ = precision_recall_fscore_support([1]*len(y_true), y_true, average='weighted', zero_division=0)
                    
                    # For combined task, accuracy is the main metric
                    prec = rec = f1 = acc
                
                elif model == 'Naive Bayes':
                    if task == 'Current Movement':
                        y_true, y_pred = nb_results['current']['y_true'], nb_results['current']['y_pred']
                    else:  # Terminal Movement
                        y_true, y_pred = nb_results['terminal']['y_true'], nb_results['terminal']['y_pred']
                    
                    acc = accuracy_score(y_true, y_pred)
                    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
                
                else:  # LSTM
                    if task == 'Current Movement':
                        y_true, y_pred = lstm_results['current']['y_true'], lstm_results['current']['y_pred']
                    else:  # Terminal Movement
                        y_true, y_pred = lstm_results['terminal']['y_true'], lstm_results['terminal']['y_pred']
                    
                    acc = accuracy_score(y_true, y_pred)
                    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
                
                results_data[task][model] = {
                    'Accuracy': acc,
                    'Precision': prec,
                    'Recall': rec,
                    'F1-Score': f1
                }
        
        # Plot performance for each task
        for task_idx, task in enumerate(tasks):
            task_data = [[results_data[task][model][metric] for model in models] for metric in metrics]
            x = np.arange(len(models))
            width = 0.2
            
            for i, metric in enumerate(metrics):
                axes[0, task_idx].bar(x + i*width, task_data[i], width, label=metric, alpha=0.8)
            
            axes[0, task_idx].set_xlabel('Model')
            axes[0, task_idx].set_ylabel('Score')
            axes[0, task_idx].set_title(f'{task} Performance')
            axes[0, task_idx].set_xticks(x + width * 1.5)
            axes[0, task_idx].set_xticklabels(models)
            if task_idx == 0:  # Only show legend on first subplot
                axes[0, task_idx].legend()
            axes[0, task_idx].set_ylim(0, 1)
        
        # Accuracy comparison across all tasks
        all_tasks = ['Current Movement', 'Terminal Movement', 'Combined (Both Correct)']
        nb_accs = [results_data[task]['Naive Bayes']['Accuracy'] for task in all_tasks]
        lstm_accs = [results_data[task][model_name]['Accuracy'] for task in all_tasks]
        
        x_tasks = np.arange(len(all_tasks))
        width = 0.35
        
        axes[1,0].bar(x_tasks - width/2, nb_accs, width, label='Naive Bayes', alpha=0.8, color='skyblue')
        axes[1,0].bar(x_tasks + width/2, lstm_accs, width, label="LSTM", alpha=0.8, color='lightgreen')
        
        axes[1,0].set_xlabel('Task')
        axes[1,0].set_ylabel('Accuracy')
        axes[1,0].set_title('Accuracy Comparison Across All Tasks')
        axes[1,0].set_xticks(x_tasks)
        axes[1,0].set_xticklabels(["Current Movement", "Sequential Condition", "Combined (Both Correct)"], rotation=15, ha='right')
        axes[1,0].legend()
        axes[1,0].set_ylim(0, 1)
        
        # Performance improvement (LSTM vs Naive Bayes)
        improvements = {}
        for task in all_tasks:
            for metric in metrics:
                lstm_score = results_data[task][model_name][metric]
                nb_score = results_data[task]['Naive Bayes'][metric]
                improvement = lstm_score - nb_score
                if task == "Terminal Movement":
                    improvements[f"Sequential Condition_{metric}"] = improvement
                else: improvements[f"{task}_{metric}"] = improvement

        improvement_labels = list(improvements.keys())
        improvement_values = list(improvements.values())
        colors = ['green' if v > 0 else 'red' for v in improvement_values]
        
        axes[1,1].barh(improvement_labels, improvement_values, color=colors, alpha=0.7)
        axes[1,1].set_xlabel(f'Performance Difference (LSTM - Naive Bayes)')
        axes[1,1].set_title(f'LSTM vs Naive Bayes Performance Difference')
        axes[1,1].axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # Task difficulty comparison (show how combined task is harder)
        task_difficulties = {
            'Naive Bayes': nb_accs,
            model_name: lstm_accs
        }
        
        x = np.arange(len(all_tasks))
        colors = ['lightcoral', 'lightyellow', 'lightblue']
        
        for i, task in enumerate(all_tasks):
            nb_acc = task_difficulties['Naive Bayes'][i]
            lstm_acc = task_difficulties[model_name][i]
            axes[1,2].bar(i - 0.2, nb_acc, 0.4, label='Naive Bayes' if i == 0 else "", 
                         alpha=0.8, color='skyblue')
            axes[1,2].bar(i + 0.2, lstm_acc, 0.4, label=model_name if i == 0 else "", 
                         alpha=0.8, color='lightgreen')
            
            # Add accuracy values on bars
            axes[1,2].text(i - 0.2, nb_acc + 0.01, f'{nb_acc:.3f}', ha='center', va='bottom', fontsize=9)
            axes[1,2].text(i + 0.2, lstm_acc + 0.01, f'{lstm_acc:.3f}', ha='center', va='bottom', fontsize=9)
        
        axes[1,2].set_xlabel('Task')
        axes[1,2].set_ylabel('Accuracy')
        axes[1,2].set_title('Task Difficulty Analysis')
        axes[1,2].set_xticks(x)
        axes[1,2].set_xticklabels(all_tasks, rotation=15, ha='right')
        axes[1,2].legend()
        axes[1,2].set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_per_sequence_performance(self, combined_data, nb_results, lstm_results, save_dir, model_name="LSTM", 
                                    sequence_order=["Reach", "GraspSeq", "HandtoMouthSeq", "InMouthSeq", "ChewSeq"], subcomponent_order=["reach", "grasp", "hand_to_mouth", "in_mouth"]):
        """Analyze performance for each sequence condition with custom ordering
        
        Args:
            combined_data: Combined dataset
            nb_results: Naive Bayes results
            lstm_results: LSTM results  
            save_dir: Directory to save plots
            model_name: Name of the model (default "LSTM")
            sequence_order: List specifying order of sequences for top plots (0,0 and 0,1)
            subcomponent_order: List specifying order of subcomponents for bottom left plot (1,0)
        """
        from sklearn.metrics import accuracy_score
        
        # Get sequence conditions for each prediction
        sequence_conditions = combined_data['sequence_conditions']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        
        # Performance by sequence condition
        unique_sequences = np.unique(sequence_conditions)
        
        # Apply custom ordering if provided, otherwise use natural order
        if sequence_order is not None:
            # Validate that all sequences in data are in the provided order
            missing_sequences = set(unique_sequences) - set(sequence_order)
            extra_sequences = set(sequence_order) - set(unique_sequences)
            
            if missing_sequences:
                print(f"Warning: Sequences {missing_sequences} found in data but not in sequence_order")
            if extra_sequences:
                print(f"Warning: Sequences {extra_sequences} in sequence_order but not found in data")
                
            # Use only sequences that exist in both data and order
            ordered_sequences = [seq for seq in sequence_order if seq in unique_sequences]
        else:
            ordered_sequences = sorted(unique_sequences)
        
        # Current movement performance by sequence
        nb_current_accs = []
        lstm_current_accs = []
        
        for seq in ordered_sequences:
            # Get indices for this sequence
            seq_mask = sequence_conditions == seq
            
            # Naive Bayes current
            nb_acc = accuracy_score(
                np.array(nb_results['current']['y_true'])[seq_mask],
                np.array(nb_results['current']['y_pred'])[seq_mask]
            )
            nb_current_accs.append(nb_acc)
            
            # LSTM current
            lstm_acc = accuracy_score(
                np.array(lstm_results['current']['y_true'])[seq_mask],
                np.array(lstm_results['current']['y_pred'])[seq_mask]
            )
            lstm_current_accs.append(lstm_acc)
        
        x = np.arange(len(ordered_sequences))
        width = 0.35
        
        axes[0,0].bar(x - width/2, nb_current_accs, width, label='Naive Bayes', alpha=0.8)
        axes[0,0].bar(x + width/2, lstm_current_accs, width, label=model_name, alpha=0.8)
        axes[0,0].set_xlabel('Sequential Condition')
        axes[0,0].set_ylabel('Accuracy')
        axes[0,0].set_title('Current Movement Accuracy by Sequential Condition')
        axes[0,0].set_xticks(x)
        axes[0,0].set_xticklabels(ordered_sequences, rotation=45, ha='right')
        axes[0,0].legend(loc= "lower center")
        axes[0,0].set_ylim(0, 1)
        
        # Terminal movement performance by sequence (same order as current)
        nb_terminal_accs = []
        lstm_terminal_accs = []
        
        for seq in ordered_sequences:
            seq_mask = sequence_conditions == seq
            
            # Naive Bayes terminal
            nb_acc = accuracy_score(
                np.array(nb_results['terminal']['y_true'])[seq_mask],
                np.array(nb_results['terminal']['y_pred'])[seq_mask]
            )
            nb_terminal_accs.append(nb_acc)
            
            # LSTM terminal
            lstm_acc = accuracy_score(
                np.array(lstm_results['terminal']['y_true'])[seq_mask],
                np.array(lstm_results['terminal']['y_pred'])[seq_mask]
            )
            lstm_terminal_accs.append(lstm_acc)
        
        axes[0,1].bar(x - width/2, nb_terminal_accs, width, label='Naive Bayes', alpha=0.8)
        axes[0,1].bar(x + width/2, lstm_terminal_accs, width, label=model_name, alpha=0.8)
        axes[0,1].set_xlabel('Sequential Condition')
        axes[0,1].set_ylabel('Accuracy')
        axes[0,1].set_title('Sequential Condition Classification Accuracy')
        axes[0,1].set_xticks(x)
        axes[0,1].set_xticklabels(ordered_sequences, rotation=45, ha='right')
        axes[0,1].legend()
        axes[0,1].set_ylim(0, 1)
        
        # Performance by subcomponent
        subcomponents = combined_data['subcomponent_names']
        unique_subcomponents = np.unique(subcomponents)
        
        # Apply custom ordering for subcomponents if provided
        if subcomponent_order is not None:
            # Validate subcomponent ordering
            missing_subcomponents = set(unique_subcomponents) - set(subcomponent_order)
            extra_subcomponents = set(subcomponent_order) - set(unique_subcomponents)
            
            if missing_subcomponents:
                print(f"Warning: Subcomponents {missing_subcomponents} found in data but not in subcomponent_order")
            if extra_subcomponents:
                print(f"Warning: Subcomponents {extra_subcomponents} in subcomponent_order but not found in data")
                
            # Use only subcomponents that exist in both data and order
            ordered_subcomponents = [sub for sub in subcomponent_order if sub in unique_subcomponents]
        else:
            ordered_subcomponents = sorted(unique_subcomponents)
        
        nb_sub_accs = []
        lstm_sub_accs = []
        
        for sub in ordered_subcomponents:
            sub_mask = subcomponents == sub
            
            # Current movement accuracy for this subcomponent
            nb_acc = accuracy_score(
                np.array(nb_results['current']['y_true'])[sub_mask],
                np.array(nb_results['current']['y_pred'])[sub_mask]
            )
            nb_sub_accs.append(nb_acc)
            
            lstm_acc = accuracy_score(
                np.array(lstm_results['current']['y_true'])[sub_mask],
                np.array(lstm_results['current']['y_pred'])[sub_mask]
            )
            lstm_sub_accs.append(lstm_acc)
        
        x_sub = np.arange(len(ordered_subcomponents))
        
        axes[1,0].bar(x_sub - width/2, nb_sub_accs, width, label='Naive Bayes', alpha=0.8)
        axes[1,0].bar(x_sub + width/2, lstm_sub_accs, width, label=model_name, alpha=0.8)
        axes[1,0].set_xlabel('Movement Subcomponent')
        axes[1,0].set_ylabel('Accuracy')
        axes[1,0].set_title('Current Movement Classification Accuracy')
        axes[1,0].set_xticks(x_sub)
        axes[1,0].set_xticklabels(ordered_subcomponents, rotation=45, ha='right')
        axes[1,0].legend(loc= "lower center")
        axes[1,0].set_ylim(0, 1)
        
        # Sample size vs performance (uses original sequence order for consistency with scatter plot)
        sample_sizes = []
        performance_diffs = []
        
        for seq in ordered_sequences:
            seq_mask = sequence_conditions == seq
            sample_size = np.sum(seq_mask)
            sample_sizes.append(sample_size)
            
            # Calculate performance difference (LSTM - NB) for terminal prediction
            lstm_acc = accuracy_score(
                np.array(lstm_results['terminal']['y_true'])[seq_mask],
                np.array(lstm_results['terminal']['y_pred'])[seq_mask]
            )
            nb_acc = accuracy_score(
                np.array(nb_results['terminal']['y_true'])[seq_mask],
                np.array(nb_results['terminal']['y_pred'])[seq_mask]
            )
            performance_diffs.append(lstm_acc - nb_acc)
        
        axes[1,1].scatter(sample_sizes, performance_diffs, s=100, alpha=0.7)
        for i, seq in enumerate(ordered_sequences):
            axes[1,1].annotate(seq, (sample_sizes[i], performance_diffs[i]), 
                            xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        axes[1,1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1,1].set_xlabel('Sample Size (number of trials)')
        axes[1,1].set_ylabel('Performance Difference (LSTM - NB)')
        axes[1,1].set_title(f'Sample Size vs {model_name} Advantage')
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/per_sequence_performance.png', dpi=300, bbox_inches='tight')
        plt.show()
 
    
    def run_complete_analysis(self, save_dir, optimize_hyperparameters=True, 
                            default_hidden_size=64, default_learning_rate=0.001):
        """Run the complete sequential movement analysis with optional hyperparameter tuning
        
        Args:
            save_dir: Directory to save plots and results
            optimize_hyperparameters: If True, perform hyperparameter tuning based on combined accuracy
            default_hidden_size: Hidden size to use if not optimizing (default: 64)
            default_learning_rate: Learning rate to use if not optimizing (default: 0.001)
        """
        print("=== SEQUENTIAL MOVEMENT CLASSIFICATION PIPELINE ===")
        
        # Load and analyze data
        self.load_data()
        self.analyze_trial_distribution()
        
        # Combine data with proper sequential labels
        combined_data = self.combine_sequential_data()
        
        # Train Naive Bayes model
        nb_results = self.create_naive_bayes_model(combined_data)
        
        # Handle hyperparameter optimization
        if optimize_hyperparameters:
            print("\n=== STARTING HYPERPARAMETER TUNING (OPTIMIZING FOR COMBINED ACCURACY) ===")
            print("Hyperparameters will be selected based on the accuracy of predicting")
            print("BOTH current and terminal movements correctly (combined accuracy).")
            
            tuning_results, best_hyperparams = self.tune_lstm_hyperparameters(combined_data)
            
            # Plot hyperparameter tuning results
            self.plot_hyperparameter_tuning_results(tuning_results, save_dir)
            
            # Use optimized hyperparameters
            final_hidden_size = best_hyperparams['hidden_size']
            final_learning_rate = best_hyperparams['learning_rate']
            model_name = "LSTM (Optimized for Combined)"
            
            print(f"\n=== TRAINING FINAL LSTM WITH OPTIMIZED HYPERPARAMETERS ===")
            print(f"Using hyperparameters optimized for combined accuracy:")
            print(f"  Hidden size: {final_hidden_size}")
            print(f"  Learning rate: {final_learning_rate}")
            print(f"  Expected combined accuracy: {best_hyperparams['mean_combined_accuracy']:.4f} ± {best_hyperparams['std_combined_accuracy']:.4f}")
        else:
            print("\n=== SKIPPING HYPERPARAMETER TUNING - USING DEFAULT PARAMETERS ===")
            tuning_results = None
            best_hyperparams = {
                'hidden_size': default_hidden_size,
                'learning_rate': default_learning_rate,
                'mean_combined_accuracy': None,
                'std_combined_accuracy': None,
                'mean_current_accuracy': None,
                'mean_terminal_accuracy': None
            }
            
            # Use default hyperparameters
            final_hidden_size = default_hidden_size
            final_learning_rate = default_learning_rate
            model_name = "LSTM (Default)"
            
            print(f"Using default hyperparameters: hidden_size={final_hidden_size}, learning_rate={final_learning_rate}")
        
        # Train final LSTM model
        lstm_results = self.create_lstm_model(
            combined_data, 
            hidden_size=final_hidden_size,
            learning_rate=final_learning_rate
        )
        
        # Create all visualizations with final models
        combined_results = self.plot_confusion_matrices(nb_results, lstm_results, save_dir, model_name)
        self.plot_sequence_analysis(combined_data, save_dir)
        self.plot_model_comparison(nb_results, lstm_results, combined_results, save_dir, model_name)
        self.plot_per_sequence_performance(combined_data, nb_results, lstm_results, save_dir, model_name)
        
        # Print detailed results
        print("\n=== DETAILED RESULTS ===")
        
        if optimize_hyperparameters:
            print(f"\nOptimal Hyperparameters (based on combined accuracy):")
            print(f"  Hidden Size: {best_hyperparams['hidden_size']}")
            print(f"  Learning Rate: {best_hyperparams['learning_rate']}")
            print(f"  Best CV Combined Accuracy: {best_hyperparams['mean_combined_accuracy']:.4f} ± {best_hyperparams['std_combined_accuracy']:.4f}")
            print(f"  Corresponding Current Accuracy: {best_hyperparams['mean_current_accuracy']:.4f}")
            print(f"  Corresponding Terminal Accuracy: {best_hyperparams['mean_terminal_accuracy']:.4f}")
        else:
            print(f"\nUsed Default Hyperparameters:")
            print(f"  Hidden Size: {best_hyperparams['hidden_size']}")
            print(f"  Learning Rate: {best_hyperparams['learning_rate']}")
        
        print("\nNaive Bayes - Current Movement:")
        current_labels = nb_results['current']['label_encoder'].classes_
        print(classification_report(nb_results['current']['y_true'], nb_results['current']['y_pred'], 
                                target_names=current_labels))
        
        print("\nNaive Bayes - Terminal Movement:")
        terminal_labels = nb_results['terminal']['label_encoder'].classes_
        print(classification_report(nb_results['terminal']['y_true'], nb_results['terminal']['y_pred'],
                                target_names=terminal_labels))
        
        print(f"\n{model_name} - Current Movement:")
        print(classification_report(lstm_results['current']['y_true'], lstm_results['current']['y_pred'],
                                target_names=current_labels))
        
        print(f"\n{model_name} - Terminal Movement:")
        print(classification_report(lstm_results['terminal']['y_true'], lstm_results['terminal']['y_pred'],
                                target_names=terminal_labels))
        
        # Performance improvement summary
        nb_current_acc = np.mean(nb_results['current']['y_true'] == nb_results['current']['y_pred'])
        nb_terminal_acc = np.mean(nb_results['terminal']['y_true'] == nb_results['terminal']['y_pred'])
        lstm_current_acc = np.mean(lstm_results['current']['y_true'] == lstm_results['current']['y_pred'])
        lstm_terminal_acc = np.mean(lstm_results['terminal']['y_true'] == lstm_results['terminal']['y_pred'])
        
        print(f"\n=== PERFORMANCE IMPROVEMENT SUMMARY ===")
        print(f"Current Movement Classification:")
        print(f"  Naive Bayes: {nb_current_acc:.4f}")
        print(f"  {model_name}: {lstm_current_acc:.4f}")
        print(f"  Improvement: {lstm_current_acc - nb_current_acc:+.4f} ({((lstm_current_acc - nb_current_acc) / nb_current_acc * 100):+.1f}%)")
        
        print(f"\nTerminal Movement Classification:")
        print(f"  Naive Bayes: {nb_terminal_acc:.4f}")
        print(f"  {model_name}: {lstm_terminal_acc:.4f}")
        print(f"  Improvement: {lstm_terminal_acc - nb_terminal_acc:+.4f} ({((lstm_terminal_acc - nb_terminal_acc) / nb_terminal_acc * 100):+.1f}%)")
        
        print(f"\nCombined Task (Both Predictions Correct):")
        print(f"  Naive Bayes: {combined_results['nb_combined_accuracy']:.4f}")
        print(f"  {model_name}: {combined_results['lstm_combined_accuracy']:.4f}")
        combined_improvement = combined_results['lstm_combined_accuracy'] - combined_results['nb_combined_accuracy']
        combined_rel_improvement = (combined_improvement / combined_results['nb_combined_accuracy'] * 100)
        print(f"  Improvement: {combined_improvement:+.4f} ({combined_rel_improvement:+.1f}%)")
        
        if optimize_hyperparameters:
            print(f"\n=== HYPERPARAMETER OPTIMIZATION EFFECTIVENESS ===")
            print(f"The hyperparameters were specifically optimized to maximize combined accuracy,")
            print(f"where both current and terminal predictions must be correct.")
            print(f"Expected combined accuracy from CV: {best_hyperparams['mean_combined_accuracy']:.4f}")
            print(f"Actual combined accuracy on full data: {combined_results['lstm_combined_accuracy']:.4f}")
            
        print(f"\n=== TASK DIFFICULTY ANALYSIS ===")
        print(f"The combined task (predicting both current and terminal movements correctly)")
        print(f"is more challenging than individual tasks, as expected.")
        print(f"Combined accuracy represents the model's ability to understand the complete")
        print(f"movement sequence context rather than just individual components.")
        
        if optimize_hyperparameters:
            print(f"\nBy optimizing hyperparameters specifically for combined accuracy,")
            print(f"we ensure the model learns to balance both tasks effectively,")
            print(f"rather than excelling at one task at the expense of the other.")
        
        return combined_data, nb_results, lstm_results, tuning_results, best_hyperparams, combined_results

# Usage examples
if __name__ == "__main__":
    data_dir = "/Users/sabrasisler/Desktop/NPTL/Data/t12.2025.03.04/formatted_for_upload"
    save_dir = "/Users/sabrasisler/Desktop/NPTL/nptlAnalysis/t12.2025.03.04/python/sequential_analysis"
    
    pipeline = SequentialMovementPipeline(data_dir)
    
    # Example 1: Full analysis with hyperparameter optimization (default)
    print("Running full analysis with hyperparameter optimization...")
    combined_data, nb_results, lstm_results, tuning_results, best_hyperparams, combined_results = \
        pipeline.run_complete_analysis(save_dir, optimize_hyperparameters=True)
