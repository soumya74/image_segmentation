from tensorboard.backend.event_processing import event_accumulator
import pandas as pd

# Point this to the folder containing your events.out.tfevents file
log_dir = "C:\\Users\\soumy\\Downloads\\events.out.tfevents.1790485870.a86e24e6878c.3728.0"

# Initialize and load the binary event data
ea = event_accumulator.EventAccumulator(log_dir)
ea.Reload()

# Check available tags and extract the loss data into a Pandas DataFrame
if 'Loss/train' in ea.scalars.Keys():
    loss_events = ea.scalars.Items('Loss/train')
    
    # Create a clean DataFrame with the Step number and Loss value
    df = pd.DataFrame([(e.step, e.value) for e in loss_events], columns=['Step', 'Loss'])
    
    print(df.head())

    # Prints (rows, columns) -> (2, 3)
    print(df.shape) 

    # Prints total number of elements (rows * columns) -> 6
    print(df.size)
    
    # Optional: Save it as a readable CSV
    # df.to_csv("training_loss.csv", index=False)
else:
    print("No 'Loss/train' scalar found in this event file.")