import sys
import time

import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from model import TrashToTreasure
from utils.loss import ReconstructionLoss,UsefulLoss,GapLoss
from utils.eval_metrice import eval_senti
from utils.tools import save_model,_get_class_weights

class Solver(object):
    def __init__(self, config, train_loader, dev_loader, test_loader, is_train=True, model=None, trial=None):
        self.config = config
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.test_loader = test_loader
        self.sigma=config.sigma
        self.beta=config.beta
        self.device = config.device


        self.is_train = is_train
        self.model = model

        self.update_batch = config.update_batch

        # Initialize the model if not provided
        if model is None:
            self.model = TrashToTreasure(config).to(self.device)
        if config.dataset=='Reuters':
            config.learning_rate=5e-4

        self.class_weights = _get_class_weights(self.train_loader,self.device)

        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights).to(self.device)
        self.ReconstructionLoss = ReconstructionLoss().to(self.device)
        self.UsefulLoss = UsefulLoss(self.config,class_weights=self.class_weights).to(self.device)
        self.GapLoss = GapLoss(self.config,class_weights=self.class_weights).to(self.device)

        # Set up optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

        # Learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode='min', patience=config.when, factor=0.5
        )

        # Initialize dynamic values for tracking the best performance
        self.best_valid_acc = 0
        self.best_test_acc = 0
        self.best_epoch = 0
        self.best_results = None
        self.best_truths = None

    def train_and_eval(self):
        model = self.model
        optimizer_main = self.optimizer
        scheduler_main = self.scheduler
        criterion = self.criterion

        def train(model, optimizer, criterion):
            """Training loop for one epoch."""
            model.train()
            num_batches = len(self.train_loader)
            proc_loss, proc_size = 0, 0
            start_time = time.time()
            for batch_idx, (data_list, labels) in enumerate(self.train_loader):
                data_list = [x.to(self.device) for x in data_list]
                labels = labels.to(self.device)
                batch_size = labels.size(0)

                optimizer.zero_grad()

                outputs  = model(data_list)

                original_list = outputs['original_list']
                useful_list = outputs['useful_list']
                trash_list = outputs['trash_list']
                reconstructed_list = outputs['reconstructed_list']
                treasure = outputs['treasure']
                predictions = outputs['prediction']


                cls_loss = criterion(predictions, labels)
                useful_loss = self.UsefulLoss(useful_list,labels)
                rec_loss = self.ReconstructionLoss(reconstructed_list,original_list)
                gap_loss = self.GapLoss(treasure,trash_list,labels)

                total_loss = cls_loss +useful_loss +  self.sigma*rec_loss + self.beta*gap_loss

                total_loss.backward()

                if (batch_idx + 1) % self.update_batch == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.clip)
                    optimizer.step()

                proc_loss += total_loss.item() * batch_size
                proc_size += batch_size

                if batch_idx % self.config.log_interval == 0 and batch_idx > 0:
                    avg_loss = proc_loss / proc_size
                    elapsed_time = time.time() - start_time
                    print(
                        f'Epoch {self.epoch:2d} | Batch {batch_idx:3d}/{num_batches:3d} | Time/Batch(ms) {elapsed_time * 1000 / self.config.log_interval:5.2f} | Train Loss {avg_loss:5.4f}'
                    )
                    proc_loss, proc_size = 0, 0
                    start_time = time.time()
            return  proc_loss / len(self.train_loader.dataset)

        def evaluate(model, criterion, test=False):
            """Evaluation loop."""
            model.eval()
            loader = self.test_loader if test else self.dev_loader
            total_loss = 0.0
            results, truths = [], []

            with torch.no_grad():
                for batch_idx, (data_list, labels) in enumerate(loader):
                    data_list = [x.to(self.device) for x in data_list]
                    labels = labels.to(self.device)
                    batch_size = labels.size(0)

                    # Forward pass
                    outputs = model(data_list)

                    predictions = outputs['prediction']

                    total_loss += criterion(predictions, labels).item() * batch_size

                    results.append(outputs['prediction'])
                    truths.append(labels)

            avg_loss = total_loss / (len(self.test_loader.dataset) if test else len(self.dev_loader.dataset))
            results = torch.cat(results)
            truths = torch.cat(truths)
            return avg_loss, results, truths

        early_stop_counter = 0  # Counter for early stopping

        for epoch in range(1, self.config.num_epochs + 1):
            start = time.time()
            self.epoch = epoch

            # Training phase
            train_loss = train(model, optimizer_main, criterion)

            # Evaluation phase
            val_loss, val_results, val_truths = evaluate(model, criterion, test=False)
            test_loss, results, truths = evaluate(model, criterion, test=True)

            # Scheduler step
            scheduler_main.step(val_loss)

            # Print results
            end = time.time()
            duration = end - start
            print("-" * 50)
            print(
                f'Epoch {epoch:2d} | Time {duration:5.4f} sec | Valid Loss {val_loss:5.4f} | Test Loss {test_loss:5.4f}'
            )
            print("-" * 50)

            print("----------------val-------------------")
            val_acc=eval_senti(val_results, val_truths, False)
            print("----------------test------------------")
            test_acc=eval_senti(results, truths, False)

            # Save model if performance improved
            if test_acc > self.best_test_acc:
                self.best_epoch = epoch
                self.best_test_acc = test_acc
                self.best_results = results  # Store results of the best model
                self.best_truths = truths  # Store truths of the best model
                print("----------------the best model saved----------------")
                eval_senti(results, truths, False)
                print(f"Saved model at pre_trained_models/best_model.pt!")
                save_model(self.config, model)
                early_stop_counter = 0
            else:
                early_stop_counter += 1

            if early_stop_counter >= self.config.patience:
                print("Early stopping triggered.")
                break

        # Output best results
        print(f'Best epoch: {self.best_epoch}')
        if self.best_results is not None and self.best_truths is not None:
                eval_senti(self.best_results, self.best_truths, False)
        sys.stdout.flush()
        return self.best_test_acc
