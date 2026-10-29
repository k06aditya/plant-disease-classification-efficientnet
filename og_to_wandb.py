import wandb

# 1. Start a new WandB run
run = wandb.init(project="plant-disease-efficientnet", job_type="log_epoch_history")

# 2. Complete epoch-by-epoch training history
epochs_data = [
    {"epoch": 1, "train_accuracy": 0.8504, "train_loss": 0.5975, "val_accuracy": 0.9317, "val_loss": 0.2557},
    {"epoch": 2, "train_accuracy": 0.9231, "train_loss": 0.2649, "val_accuracy": 0.9452, "val_loss": 0.1864},
    {"epoch": 3, "train_accuracy": 0.9337, "train_loss": 0.2138, "val_accuracy": 0.9514, "val_loss": 0.1576},
    {"epoch": 4, "train_accuracy": 0.9411, "train_loss": 0.1870, "val_accuracy": 0.9570, "val_loss": 0.1391},
    {"epoch": 5, "train_accuracy": 0.9447, "train_loss": 0.1726, "val_accuracy": 0.9579, "val_loss": 0.1327},
    {"epoch": 6, "train_accuracy": 0.9465, "train_loss": 0.1648, "val_accuracy": 0.9601, "val_loss": 0.1265},
    {"epoch": 7, "train_accuracy": 0.9503, "train_loss": 0.1530, "val_accuracy": 0.9612, "val_loss": 0.1201},
    {"epoch": 8, "train_accuracy": 0.9514, "train_loss": 0.1486, "val_accuracy": 0.9660, "val_loss": 0.1086},
    {"epoch": 9, "train_accuracy": 0.9518, "train_loss": 0.1435, "val_accuracy": 0.9628, "val_loss": 0.1128},
    {"epoch": 10, "train_accuracy": 0.9529, "train_loss": 0.1393, "val_accuracy": 0.9639, "val_loss": 0.1111}
]

# 3. Log each epoch sequentially so WandB draws line charts
for data in epochs_data:
    wandb.log(data)

# 4. Attach your model artifact
artifact = wandb.Artifact(name="best_plant_disease_model", type="model")
artifact.add_file("plant_disease_model.keras")
wandb.log_artifact(artifact)

# 5. Finish run
wandb.finish()
print("Successfully logged full epoch curves and model artifact to WandB!")