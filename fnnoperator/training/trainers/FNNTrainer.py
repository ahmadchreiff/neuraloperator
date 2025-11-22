import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

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
                    loss = self.loss_fn(y_pred, Y_batch) / self.gradient_accumulation_steps # scale loss for accumulation
                
                self.scaler.scale(loss).backward() # backward pass with scaling
            else:
                y_pred = self.model(X_batch) # forward pass
                loss = self.loss_fn(y_pred, Y_batch) / self.gradient_accumulation_steps # scale loss for accumulation
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
            return None
        
        self.model.eval() # set the model to evaluation mode
        val_loss = 0.0

        with torch.no_grad(): # disable gradient computation since we are not training
            for X_batch, Y_batch in self.val_loader:
                X_batch = X_batch.to(self.device)
                Y_batch = Y_batch.to(self.device)

                if self.use_amp and self.scaler is not None:
                    with torch.cuda.amp.autocast():
                        y_pred = self.model(X_batch)
                        loss = self.loss_fn(y_pred, Y_batch)
                else:
                    y_pred = self.model(X_batch)
                    loss = self.loss_fn(y_pred, Y_batch)
                
                val_loss += loss.item() * X_batch.size(0)

        return val_loss / len(self.val_loader.dataset) # average over number of samples in the dataset
    
    def test(self):
        if self.test_loader is None:
            return None
        
        self.model.eval()
        test_loss = 0.0
        test_preds = [] # predictions for each batch

        with torch.no_grad():
            for X_batch, Y_batch in self.test_loader:
                X_batch = X_batch.to(self.device)
                Y_batch = Y_batch.to(self.device)

                y_pred = self.model(X_batch)
                loss = self.loss_fn(y_pred, Y_batch)
                test_loss += loss.item() * X_batch.size(0)
                test_preds.append(y_pred.detach().cpu().numpy())

        return test_loss / len(self.test_loader.dataset), test_preds

    def fit(self):
        """
        Train the model for n_epochs with validation monitoring.
        
        Returns:
            dict: Training history with 'train_loss' and 'val_loss' lists.
                Note: Test evaluation should be done separately using test() method.
        """
        history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(self.n_epochs):
            # Train for one epoch
            train_loss = self.train_epoch()
            
            # Validate (if validation data is available)
            val_loss = self.validate()
            
            # Update learning rate
            self.scheduler.step()
            
            # Store history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss if val_loss is not None else None)
            
            # Print progress
            if self.verbose:
                val_str = f"{val_loss:.6f}" if val_loss is not None else "N/A"
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch+1:4d}/{self.n_epochs} | "
                      f"Train Loss: {train_loss:.6f} | "
                      f"Val Loss: {val_str} | "
                      f"LR: {current_lr:.2e}")
        
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