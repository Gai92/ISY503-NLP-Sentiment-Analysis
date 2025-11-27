import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from train import train_lstm

def quick_test():
    print("Quick training test with small data...")
    
    # just test if training runs
    # it will use real data but only for a quick test
    
    try:
        # this will train on full data but we can stop it early
        model, history = train_lstm()
        print("Training works!")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    quick_test()
