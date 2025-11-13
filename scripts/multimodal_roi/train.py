"""
Training Pipeline for Multi-modal ROI Feature Extraction + XGBoost
多模態 ROI 特徵提取 + XGBoost 訓練 Pipeline
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import warnings
warnings.filterwarnings('ignore')

from config import *
from resnet3d_mini import MultiModalFeatureExtractor
from dataset import create_dataloaders


class FeatureExtractionTrainer:
    """
    Trainer for Multi-modal Feature Extraction
    
    Training Strategy:
    1. Train 3 Mini-CNNs to extract meaningful features from ROI patches
    2. Extract features for all subjects
    3. Train XGBoost classifier on extracted features
    """
    
    def __init__(
        self,
        model,
        dataloaders,
        device='cuda',
        output_dir=None
    ):
        self.model = model.to(device)
        self.dataloaders = dataloaders
        self.device = device
        self.output_dir = Path(output_dir) if output_dir else OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize optimizer and scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
        )
        
        # Loss function (for feature learning)
        self.criterion = nn.CrossEntropyLoss()
        
        # Add a temporary classifier head for feature learning
        self.temp_classifier = nn.Linear(TOTAL_FEATURE_DIM, 3).to(device)
        
        # Tensorboard
        self.writer = SummaryWriter(log_dir=self.output_dir / 'logs')
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        self.best_val_loss = float('inf')
        self.patience_counter = 0
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        self.temp_classifier.train()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.dataloaders['train'], desc=f'Epoch {epoch+1}/{NUM_EPOCHS}')
        
        for batch in pbar:
            # Get data
            patches = batch['patches']
            labels = batch['label'].to(self.device)
            
            t1_patches = patches['T1'].to(self.device)
            t2_patches = patches['T2_FLAIR'].to(self.device)
            dwi_patches = patches['DWI'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            features = self.model(t1_patches, t2_patches, dwi_patches)
            outputs = self.temp_classifier(features)
            
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        epoch_loss = running_loss / len(self.dataloaders['train'])
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        self.temp_classifier.eval()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(self.dataloaders['val'], desc='Validating'):
                # Get data
                patches = batch['patches']
                labels = batch['label'].to(self.device)
                
                t1_patches = patches['T1'].to(self.device)
                t2_patches = patches['T2_FLAIR'].to(self.device)
                dwi_patches = patches['DWI'].to(self.device)
                
                # Forward pass
                features = self.model(t1_patches, t2_patches, dwi_patches)
                outputs = self.temp_classifier(features)
                
                loss = self.criterion(outputs, labels)
                
                # Statistics
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_loss = running_loss / len(self.dataloaders['val'])
        val_acc = 100. * correct / total
        
        return val_loss, val_acc
    
    def train(self):
        """Full training loop"""
        print("\n" + "="*80)
        print("Training Multi-modal Feature Extractor")
        print("="*80)
        
        for epoch in range(NUM_EPOCHS):
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_acc = self.validate()
            
            # Update scheduler
            self.scheduler.step(val_loss)
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Tensorboard
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('Accuracy/train', train_acc, epoch)
            self.writer.add_scalar('Accuracy/val', val_acc, epoch)
            
            # Print epoch summary
            print(f'\nEpoch {epoch+1}/{NUM_EPOCHS}:')
            print(f'  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
            print(f'  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%')
            print("\n")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint('best_feature_extractor.pth')
                print(f'  [OK] Best model saved!')
            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f'\n[WARN] Early stopping triggered after {epoch+1} epochs')
                break
        
        print("\n[OK] Feature extractor training completed!")
        
        # Save final model
        self.save_checkpoint('final_feature_extractor.pth')
        
        # Save training history
        history_df = pd.DataFrame(self.history)
        history_df.to_csv(self.output_dir / 'training_history.csv', index=False)
        
        self.writer.close()
    
    def save_checkpoint(self, filename):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history
        }
        
        torch.save(checkpoint, MODEL_DIR / filename)
    
    def load_checkpoint(self, filename):
        """Load model checkpoint"""
        checkpoint = torch.load(MODEL_DIR / filename, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        self.history = checkpoint['history']
        
        print(f"[OK] Checkpoint loaded: {filename}")


def extract_features_for_xgboost(model, dataloader, device='cuda'):
    """
    Extract features from all subjects for XGBoost training
    
    Returns:
    --------
    features : np.ndarray
        Feature matrix of shape (N, 22104)
    labels : np.ndarray
        Labels of shape (N,)
    subject_ids : list
        List of subject IDs
    """
    model.eval()
    
    all_features = []
    all_labels = []
    all_subject_ids = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Extracting features'):
            # Get data
            patches = batch['patches']
            labels = batch['label'].numpy()
            subject_ids = batch['subject_id']
            
            t1_patches = patches['T1'].to(device)
            t2_patches = patches['T2_FLAIR'].to(device)
            dwi_patches = patches['DWI'].to(device)
            
            # Extract features
            features = model(t1_patches, t2_patches, dwi_patches)
            features = features.cpu().numpy()
            
            all_features.append(features)
            all_labels.append(labels)
            all_subject_ids.extend(subject_ids)
    
    features = np.vstack(all_features)
    labels = np.concatenate(all_labels)
    
    return features, labels, all_subject_ids


def train_xgboost_classifier(X_train, y_train, X_val, y_val):
    """
    Train XGBoost classifier on extracted features
    
    Parameters:
    -----------
    X_train, y_train : np.ndarray
        Training features and labels
    X_val, y_val : np.ndarray
        Validation features and labels
    
    Returns:
    --------
    model : xgb.XGBClassifier
        Trained XGBoost model
    """
    print("\n" + "="*80)
    print("Training XGBoost Classifier")
    print("="*80)
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Feature dimension: {X_train.shape[1]}")
    
    # Create XGBoost model
    model = xgb.XGBClassifier(**XGBOOST_CONFIG)
    
    # Train with early stopping
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=True
    )
    
    # Evaluate
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    
    train_acc = accuracy_score(y_train, train_pred)
    val_acc = accuracy_score(y_val, val_pred)
    
    print(f"\n[OK] XGBoost training completed!")
    print(f"   Train Accuracy: {train_acc:.4f}")
    print(f"   Val Accuracy:   {val_acc:.4f}")
    
    # Print classification report
    print("\nValidation Classification Report:")
    print(classification_report(y_val, val_pred, target_names=['NC', 'MCI', 'AD']))
    
    return model


def main():
    """Main training pipeline"""
    print("="*80)
    print("Multi-modal ROI Feature Extraction + XGBoost Pipeline")
    print("="*80)
    print(f"Device: {DEVICE}")
    print(f"Data root: {DATA_ROOT}")
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    dataloaders = create_dataloaders(
        data_root=DATA_ROOT,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        use_cache=True
    )
    
    # Create model
    print("\nInitializing model...")
    model = MultiModalFeatureExtractor(
        feature_dim=FEATURE_DIM_PER_ROI,
        initial_filters=RESNET_CONFIG['initial_filters']
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Train feature extractor
    trainer = FeatureExtractionTrainer(
        model=model,
        dataloaders=dataloaders,
        device=DEVICE,
        output_dir=OUTPUT_DIR
    )
    
    trainer.train()
    
    # Load best model
    trainer.load_checkpoint('best_feature_extractor.pth')
    
    # Extract features for XGBoost
    print("\nExtracting features for XGBoost...")
    
    X_train, y_train, train_ids = extract_features_for_xgboost(
        model, dataloaders['train'], device=DEVICE
    )
    X_val, y_val, val_ids = extract_features_for_xgboost(
        model, dataloaders['val'], device=DEVICE
    )
    X_test, y_test, test_ids = extract_features_for_xgboost(
        model, dataloaders['test'], device=DEVICE
    )
    
    print(f"\n[OK] Feature extraction completed!")
    print(f"   Train: {X_train.shape}")
    print(f"   Val:   {X_val.shape}")
    print(f"   Test:  {X_test.shape}")
    
    # Train XGBoost
    xgb_model = train_xgboost_classifier(X_train, y_train, X_val, y_val)
    
    # Test evaluation
    test_pred = xgb_model.predict(X_test)
    test_acc = accuracy_score(y_test, test_pred)
    
    print(f"\n{'='*80}")
    print("Final Test Results")
    print("="*80)
    print(f"Test Accuracy: {test_acc:.4f}")
    print("\nTest Classification Report:")
    print(classification_report(y_test, test_pred, target_names=['NC', 'MCI', 'AD']))
    
    # Save XGBoost model
    joblib.dump(xgb_model, MODEL_DIR / 'xgboost_classifier.pkl')
    print(f"\n[OK] XGBoost model saved to: {MODEL_DIR / 'xgboost_classifier.pkl'}")
    
    # Feature importance analysis
    print("\nAnalyzing feature importance...")
    feature_importance = xgb_model.feature_importances_
    
    # Save feature importance
    importance_df = pd.DataFrame({
        'feature_idx': range(len(feature_importance)),
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    importance_df.to_csv(OUTPUT_DIR / 'feature_importance.csv', index=False)
    
    print(f"\nTop 10 most important features:")
    print(importance_df.head(10))
    
    print("\n" + "="*80)
    print("Pipeline completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
