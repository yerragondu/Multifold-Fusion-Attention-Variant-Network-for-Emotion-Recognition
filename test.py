from scripts.models.v0 import EEGA
import torch
from scripts.models.v0 import AA
from scripts.models.v0 import VA
import torch.nn as nn
import pickle
import os
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
import pandas as pd
from train import *

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []

    for sub in range(1, 43):
        print("subject", sub)
        file_path = "Data/"
        data_eeg, data_aud, data_vis = load_data('EEG'), load_data('Audio'), load_data('Vision')
        model_aud, model_eeg, model_vis = load_pretrained_model()
        register_hook(model_aud, model_eeg, model_vis)
        _, testloader = prepare_dataloader()
        model_aud.model.to(device), model_eeg.model.to(device), model_vis.model.to(device)
        model = MultimodalEmotionRecognitionWithAttention(num_classes=5)
        state_dict = torch.load(f"ckpt/multimodal/model_subject_{sub:02d}.pth", weights_only=True)
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        model.to(device)

        model_aud.model.eval()
        model_eeg.model.eval()
        model_vis.model.eval()
        model.eval()

        test_correct = 0
        test_total = 0
        test = {
            'vis_correct': 0,
            'aud_correct': 0,
            'eeg_correct': 0,
            'multimodal_correct': 0,
            'total': 0
        }

        with torch.no_grad():  # Disable gradient computation for evaluation
            for batch_idx, (b_vis, b_lbl, b_eeg, b_aud) in enumerate(testloader, start=1):
                b_vis, b_lbl, b_eeg, b_aud = [b.to(device) for b in (b_vis, b_lbl, b_eeg, b_aud)]
                hooked_data = {
                    'eeg_input': [],
                    'aud_input': [],
                    'vis_input': []
                }
                o_vis = model_vis.model(b_vis)
                o_eeg = model_eeg.model(b_eeg)
                o_aud = model_aud.model(b_aud)

                _, predicted_vis = torch.max(o_vis.logits, 1)
                _, predicted_eeg = torch.max(o_eeg, 1)
                _, predicted_aud = torch.max(o_aud.logits, 1)
                test['vis_correct'] += (predicted_vis == b_lbl).sum().item()
                test['aud_correct'] += (predicted_aud == b_lbl).sum().item()
                test['eeg_correct'] += (predicted_eeg == b_lbl).sum().item()

                vis_input = torch.cat([x.to(device) for x in hooked_data['vis_input']], dim=0)
                aud_input = torch.cat([x.to(device) for x in hooked_data['aud_input']], dim=0)
                eeg_input = torch.cat([x.to(device) for x in hooked_data['eeg_input']], dim=0)
                output = model(vis_input, aud_input, eeg_input)
                _, predicted = torch.max(output, 1)
                test['multimodal_correct'] += (predicted == b_lbl).sum().item()
                test['total'] += b_lbl.size(0)

        # Calculate accuracies
        vis_accuracy = test['vis_correct'] / test['total'] * 100
        aud_accuracy = test['aud_correct'] / test['total'] * 100
        eeg_accuracy = test['eeg_correct'] / test['total'] * 100
        multimodal_accuracy = test['multimodal_correct'] / test['total'] * 100

        print(f"Vision Accuracy: {vis_accuracy:.2f}%")
        print(f"Audio Accuracy: {aud_accuracy:.2f}%")
        print(f"EEG Accuracy: {eeg_accuracy:.2f}%")
        print(f"Multimodal Accuracy: {multimodal_accuracy:.2f}%")

        # Append the results for the current subject
        results.append({
            'Subject': sub,
            'Vision Accuracy (%)': vis_accuracy,
            'Audio Accuracy (%)': aud_accuracy,
            'EEG Accuracy (%)': eeg_accuracy,
            'Multimodal Accuracy (%)': multimodal_accuracy
        })
    # Create a DataFrame and save it to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv('model_test_accuracies.csv', index=False)