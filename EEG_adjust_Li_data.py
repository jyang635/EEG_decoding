import os
import numpy as np
import argparse
import pickle



### Run this script to adjust the EEG data if you download the ones from Li's website
def main():
    parser = argparse.ArgumentParser(description='Conformer+deconv')
    
    # Add your command-line arguments
    parser.add_argument('--eeg_folder', type=str, default='/home/yjk122/IP_temp/EEG_Image_decode/Preprocessed_data_250Hz')
    parser.add_argument('--subject_id', type=str, default='sub-01', help='Subject ID to analyze')
    args = parser.parse_args()
    
    eeg_parent_dir=args.eeg_folder
    eeg_parent_dir = os.path.join(eeg_parent_dir, args.subject_id)
    print(f"EEG data folder: {eeg_parent_dir}")
    eeg_data_train = np.load(os.path.join(eeg_parent_dir, 'preprocessed_eeg_training.npy'), allow_pickle=True)
    # time_indices = np.where((eeg_data['times'] >= start_time) & (eeg_data['times'] <= end_time))[0]

    eeg_data_test = np.load(os.path.join(eeg_parent_dir, 'preprocessed_eeg_test.npy'), allow_pickle=True)

            

    eeg_data_train['times']=eeg_data_train['times'][50:]
    eeg_data_test['times']=eeg_data_test['times'][50:]
    # Save the preprocessed EEG data back to the original directories
    print("Saving updated EEG data...")

    # Save training data
    train_save_path = os.path.join(eeg_parent_dir, 'preprocessed_eeg_training.npy')
    with open(train_save_path, "wb") as f:
        pickle.dump(eeg_data_train, f, protocol=4)
    print(f"Training data saved to {train_save_path}")

    # Save test data
    test_save_path = os.path.join(eeg_parent_dir, 'preprocessed_eeg_test.npy')
    with open(test_save_path, "wb") as f:
        pickle.dump(eeg_data_test, f, protocol=4)
    print(f"Test data saved to {test_save_path}")

if __name__ == "__main__":
    main()