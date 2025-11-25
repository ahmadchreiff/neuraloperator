import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent  # Go up to neuraloperator/
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from fnnoperator.Error_Analysis import compute_spectral_error

class FNNTrainer:
    def __init__(
        self,
        model: nn.Module,
        X_train: torch.Tensor = None,
        Y_train: torch.Tensor = None,
        X_val: torch.Tensor = None,
        Y_val: torch.Tensor = None,
        X_test: torch.Tensor = None,
        Y_test: torch.Tensor = None,
        loss_fn: nn.Module = None,
        optimizer: torch.optim.Optimizer = None,
        scheduler: torch.optim.lr_scheduler._LRScheduler = None,
        device: torch.device = None,
        n_epochs: int = 100,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 128,
        gradient_accumulation_steps: int = 1,
        use_amp: bool = False,
        verbose: bool = True,
    ):
        self.model = model
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_val = X_val
        self.Y_val = Y_val
        self.X_test = X_test
        self.Y_test = Y_test
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.use_amp = use_amp
        
        # Set device first (needed for scaler initialization)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.verbose = verbose
        
        self.optimizer = optimizer or torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scaler = torch.cuda.amp.GradScaler() if use_amp and self.device.type == "cuda" else None
        self.scheduler = scheduler or torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=n_epochs)
        self.loss_fn = loss_fn or nn.MSELoss()

        # Move the model to the device
        self.model = self.model.to(self.device)
        
        # Print device information
        if self.verbose:
            print(f"Device: {self.device}")
            if torch.cuda.is_available():
                print(f"  [GPU] {torch.cuda.get_device_name(0)}")
                print(f"  [CUDA] Version: {torch.version.cuda}")
                print(f"  [Memory] {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            else:
                print("  [WARNING] Using CPU (CUDA not available)")
        
        if self.X_train is None or self.Y_train is None:
            raise ValueError("X_train and Y_train must be provided")

        # Helper function to convert to float32 tensor efficiently
        def to_float32_tensor(data):
            if isinstance(data, torch.Tensor):
                return data.to(dtype=torch.float32)
            else:
                return torch.tensor(data, dtype=torch.float32)
        
        train_dataset = TensorDataset(to_float32_tensor(self.X_train), to_float32_tensor(self.Y_train))

        self.train_loader = DataLoader (train_dataset, batch_size=batch_size, shuffle=True)
        
        if self.X_val is not None and self.Y_val is not None:
            val_dataset = TensorDataset(to_float32_tensor(self.X_val), to_float32_tensor(self.Y_val))
            self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        else:
            self.val_loader = None

        if self.X_test is not None and self.Y_test is not None:
            test_dataset = TensorDataset(to_float32_tensor(self.X_test), to_float32_tensor(self.Y_test))
            self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        else:
            self.test_loader = None
    
    def _compute_loss(self, y_pred, y_target, x_input=None):
        """
        Compute loss, handling physics-informed losses that need input.
        
        Args:
            y_pred: Model predictions
            y_target: Target values
            x_input: Input values (needed for physics-informed losses)
        
        Returns:
            Loss value
        """
        # Check if loss function needs u_input (PhysicsInformedLoss or CombinedLoss containing it)
        if 'PhysicsInformedLoss' in str(type(self.loss_fn)):
            # Direct PhysicsInformedLoss
            return self.loss_fn(y_pred, y_target, u_input=x_input)
        elif hasattr(self.loss_fn, 'losses'):
            # CombinedLoss - check if any loss is PhysicsInformedLoss
            has_physics = any(
                'PhysicsInformedLoss' in str(type(loss)) for loss in self.loss_fn.losses
            )
            if has_physics and x_input is not None:
                return self.loss_fn(y_pred, y_target, u_input=x_input)
        
        # Default: standard loss computation
        return self.loss_fn(y_pred, y_target)

    def train_epoch(self):
        self.model.train() # set the model to training mode
        running_batch_loss = 0.0
        self.optimizer.zero_grad() # zero gradients at the start of epoch

        for batch_idx, (X_batch, Y_batch) in enumerate(self.train_loader):
            X_batch = X_batch.to(self.device) # move the batch to the same device as the model
            Y_batch = Y_batch.to(self.device)
            
            # Mixed precision training
            if self.use_amp and self.scaler is not None:
                with torch.cuda.amp.autocast():
                    y_pred = self.model(X_batch) # forward pass
                    loss = self._compute_loss(y_pred, Y_batch, X_batch) / self.gradient_accumulation_steps # scale loss for accumulation
                
                self.scaler.scale(loss).backward() # backward pass with scaling
            else:
                y_pred = self.model(X_batch) # forward pass
                loss = self._compute_loss(y_pred, Y_batch, X_batch) / self.gradient_accumulation_steps # scale loss for accumulation
                loss.backward() # backward pass

            # Update weights only after accumulating gradients
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if self.use_amp and self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step() # update the weights
                self.optimizer.zero_grad() # zero gradients for next accumulation

            # Since the loss functions compute the average loss over the batch, we need to multiply the loss by the batch size to get the total loss of all samples in the batch
            # Then we will compute total over all batches and compute the average loss across all batches
            running_batch_loss += loss.item() * X_batch.size(0) * self.gradient_accumulation_steps

        return running_batch_loss / len(self.train_loader.dataset) # average over number of samples in the dataset

    def validate(self):
        if self.val_loader is None:
            return None, None, None
        
        self.model.eval() # set the model to evaluation mode
        val_loss = 0.0
        mse_loss = 0.0
        all_preds = []
        all_targets = []
        mse_fn = nn.MSELoss()

        with torch.no_grad(): # disable gradient computation since we are not training
            for X_batch, Y_batch in self.val_loader:
                X_batch = X_batch.to(self.device)
                Y_batch = Y_batch.to(self.device)

                if self.use_amp and self.scaler is not None:
                    with torch.cuda.amp.autocast():
                        y_pred = self.model(X_batch)
                        loss = self._compute_loss(y_pred, Y_batch, X_batch)
                        # Also compute MSE separately for monitoring
                        mse = mse_fn(y_pred, Y_batch)
                else:
                    y_pred = self.model(X_batch)
                    loss = self._compute_loss(y_pred, Y_batch, X_batch)
                    # Also compute MSE separately for monitoring
                    mse = mse_fn(y_pred, Y_batch)
                
                val_loss += loss.item() * X_batch.size(0)
                mse_loss += mse.item() * X_batch.size(0)
                
                # Collect predictions and targets for spectral error computation
                all_preds.append(y_pred.detach().cpu())
                all_targets.append(Y_batch.detach().cpu())

        val_loss_avg = val_loss / len(self.val_loader.dataset)
        mse_loss_avg = mse_loss / len(self.val_loader.dataset)
        
        # Compute spectral errors
        try:
            # Concatenate all batches
            preds_concat = torch.cat(all_preds, dim=0).numpy()
            targets_concat = torch.cat(all_targets, dim=0).numpy()
            
            # Handle different output shapes: (batch, channels, H, W) or (batch, time, channels, H, W)
            # Remove channel dimension and handle time steps
            if len(preds_concat.shape) == 4:  # (batch, channels, H, W)
                preds_2d = preds_concat[:, 0, :, :]  # Remove channel dim
                targets_2d = targets_concat[:, 0, :, :]
            elif len(preds_concat.shape) == 5:  # (batch, time, channels, H, W)
                # Use final time step for spectral error
                preds_2d = preds_concat[:, -1, 0, :, :]
                targets_2d = targets_concat[:, -1, 0, :, :]
            else:
                preds_2d = preds_concat
                targets_2d = targets_concat
            
            # Compute spectral errors
            spectral_results = compute_spectral_error(preds_2d, targets_2d, normalize=True)
            # Use relative spectral error: ||FFT_pred - FFT_true||_2 / ||FFT_true||_2
            spectral_error = spectral_results['relative_spectral_error']
        except Exception as e:
            # If spectral error computation fails, return None
            if self.verbose:
                print(f"Warning: Could not compute spectral error: {e}")
            spectral_error = None
        
        return val_loss_avg, mse_loss_avg, spectral_error
    
    def test(self):
        if self.test_loader is None:
            return None, None, None, None
        
        self.model.eval()
        test_loss = 0.0
        test_mse = 0.0
        test_preds = [] # predictions for each batch
        all_preds = []
        all_targets = []
        mse_fn = nn.MSELoss()

        with torch.no_grad():
            for X_batch, Y_batch in self.test_loader:
                X_batch = X_batch.to(self.device)
                Y_batch = Y_batch.to(self.device)

                y_pred = self.model(X_batch)
                loss = self._compute_loss(y_pred, Y_batch, X_batch)
                mse = mse_fn(y_pred, Y_batch)
                test_loss += loss.item() * X_batch.size(0)
                test_mse += mse.item() * X_batch.size(0)
                test_preds.append(y_pred.detach().cpu().numpy())
                
                # Collect for spectral error computation
                all_preds.append(y_pred.detach().cpu())
                all_targets.append(Y_batch.detach().cpu())

        test_loss_avg = test_loss / len(self.test_loader.dataset)
        test_mse_avg = test_mse / len(self.test_loader.dataset)
        
        # Compute spectral errors
        try:
            # Concatenate all batches
            preds_concat = torch.cat(all_preds, dim=0).numpy()
            targets_concat = torch.cat(all_targets, dim=0).numpy()
            
            # Handle different output shapes
            if len(preds_concat.shape) == 4:  # (batch, channels, H, W)
                preds_2d = preds_concat[:, 0, :, :]
                targets_2d = targets_concat[:, 0, :, :]
            elif len(preds_concat.shape) == 5:  # (batch, time, channels, H, W)
                # Use final time step for spectral error
                preds_2d = preds_concat[:, -1, 0, :, :]
                targets_2d = targets_concat[:, -1, 0, :, :]
            else:
                preds_2d = preds_concat
                targets_2d = targets_concat
            
            # Compute spectral errors
            spectral_results = compute_spectral_error(preds_2d, targets_2d, normalize=True)
            # Use relative spectral error: ||FFT_pred - FFT_true||_2 / ||FFT_true||_2
            spectral_error = spectral_results['relative_spectral_error']
        except Exception as e:
            if self.verbose:
                print(f"Warning: Could not compute spectral error: {e}")
            spectral_error = None

        return test_loss_avg, test_mse_avg, spectral_error, test_preds

    def fit(self):
        """
        Train the model for n_epochs with validation monitoring.
        
        Returns:
            dict: Training history with 'train_loss', 'val_loss', 'val_spectral_error' lists.
                Note: Test evaluation should be done separately using test() method.
        """
        history = {'train_loss': [], 'val_loss': [], 'val_mse': [], 'val_spectral_error': []}
        
        for epoch in range(self.n_epochs):
            # Reset epoch stats for combined loss debug (if using combined loss)
            if hasattr(self.loss_fn, 'reset_epoch_stats'):
                self.loss_fn.reset_epoch_stats()
            
            # Train for one epoch
            train_loss = self.train_epoch()
            
            # Validate (if validation data is available)
            val_loss, val_mse, val_spectral = self.validate()
            
            # Update learning rate
            self.scheduler.step()
            
            # Store history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss if val_loss is not None else None)
            history['val_mse'].append(val_mse if val_mse is not None else None)
            history['val_spectral_error'].append(val_spectral if val_spectral is not None else None)
            
            # Print progress (single line, no duplicates)
            if self.verbose:
                val_str = f"{val_loss:.6f}" if val_loss is not None else "N/A"
                val_mse_str = f"{val_mse:.6f}" if val_mse is not None else "N/A"
                val_spec_str = f"{val_spectral:.6f}" if val_spectral is not None else "N/A"
                current_lr = self.optimizer.param_groups[0]['lr']
                # Single print statement - ensure no duplicates
                message = f"Epoch {epoch+1:4d}/{self.n_epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_str} | Val MSE: {val_mse_str} | Val Spectral: {val_spec_str} | LR: {current_lr:.2e}"
                print(message, flush=True)
            
            # Print combined loss debug info if enabled
            if hasattr(self.loss_fn, 'print_epoch_stats'):
                self.loss_fn.print_epoch_stats(epoch + 1)
        
        return history

    def predict(self, X):
        """
        Make predictions on input data.
        
        Args:
            X: Input data. Can be numpy array or torch.Tensor.
               Shape: (n_samples, *input_dims) or (n_samples, channels, *grid)
               Single sample will be automatically batched.
        
        Returns:
            numpy.ndarray: Predictions with shape matching input (without batch dimension if single sample).
        """
        self.model.eval()
        
        # Convert to tensor if needed
        if isinstance(X, torch.Tensor):
            X_tensor = X.to(dtype=torch.float32)
        else:
            X_tensor = torch.tensor(X, dtype=torch.float32)
        
        # Handle single sample: add batch dimension
        was_single_sample = len(X_tensor.shape) < len(self.X_train.shape)
        if was_single_sample:
            X_tensor = X_tensor.unsqueeze(0)
        
        # Move to device
        X_tensor = X_tensor.to(self.device)
        
        # Make predictions
        with torch.no_grad():
            predictions = self.model(X_tensor)
            predictions = predictions.detach().cpu().numpy()
        
        # Remove batch dimension if single sample
        if was_single_sample:
            predictions = predictions.squeeze(0)
        
        return predictions