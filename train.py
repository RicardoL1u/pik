from pik.utils.trainer import Trainer, parse_arguments
import os
import wandb

if __name__ == "__main__":
    args = parse_arguments()
    
    # Ensure data files exist
    assert os.path.exists(args.hidden_states_filename)
    assert os.path.exists(args.text_generations_filename)
    
    if args.use_wandb:
        wandb.init(project="pik", name=args.wandb_run_name)
    
    trainer = Trainer(args)

    training_loss, train_metrics, val_metrics, test_metrics = trainer.trainning_loop()