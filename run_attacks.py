import os
import argparse
from tqdm import tqdm
import cv2
from datetime import datetime
from configs import load_config
from attacker import PhishpediaAttacker
import json
import traceback
import sys
import numpy as np
import torch


def convert_to_serializable(obj):
    """Convert numpy and torch types to Python native types for JSON serialization"""
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy().tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    return obj


def run_attacks():
    parser = argparse.ArgumentParser(description='Run adversarial attacks on Phishpedia')
    parser.add_argument('--data_dir', default='datasets/adv_test', help='Directory containing test data')
    parser.add_argument('--output_dir', default='attack_results', help='Directory to save results')
    parser.add_argument('--attack_type', default='fgsm', choices=['fgsm', 'pgd'], help='Type of attack to run')
    parser.add_argument('--target_type', default='detector', choices=['detector', 'classifier'], 
                       help='Which component to attack')
    parser.add_argument('--num_steps', type=int, default=5, help='Number of attack steps')
    parser.add_argument('--epsilon', type=float, default=0.1, help='Maximum perturbation size')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    args = parser.parse_args()

    # Create output directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, f'{args.attack_type}_{args.target_type}_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    print(f"Created output directory: {output_dir}")

    # Set up logging
    if args.debug:
        log_file = os.path.join(output_dir, 'debug.log')
        sys.stdout = open(log_file, 'w')
        sys.stderr = sys.stdout

    try:
        # Load Phishpedia models and configurations
        print("Loading models and configurations...")
        ELE_MODEL, SIAMESE_THRE, SIAMESE_MODEL, LOGO_FEATS, LOGO_FILES, DOMAIN_MAP_PATH = load_config()
        print(f'Loaded reference list with {len(LOGO_FEATS)} logos')

        # Initialize attacker
        print("Initializing attacker...")
        attacker = PhishpediaAttacker(
            rcnn_model=ELE_MODEL,
            siamese_model=SIAMESE_MODEL,
            domain_map_path=DOMAIN_MAP_PATH,
            logo_feats=LOGO_FEATS,
            logo_files=LOGO_FILES,
            epsilon=args.epsilon
        )

        # Initialize result tracking
        results = []
        attack_success = 0
        total_samples = 0
        
        # Get list of valid folders
        folders = [f for f in os.listdir(args.data_dir) 
                  if os.path.isdir(os.path.join(args.data_dir, f))]
        print(f"Found {len(folders)} samples to process")

        # Process each sample in the dataset
        for folder in tqdm(folders):
            print(f"\n{'='*50}")
            print(f"Processing folder: {folder}")
            folder_path = os.path.join(args.data_dir, folder)

            # Get screenshot and URL
            screenshot_path = os.path.join(folder_path, "shot.png")
            info_path = os.path.join(folder_path, "info.txt")
            
            if not os.path.exists(screenshot_path):
                print(f"Screenshot not found: {screenshot_path}")
                continue
            if not os.path.exists(info_path):
                print(f"Info file not found: {info_path}")
                continue

            # Read URL
            try:
                with open(info_path, 'r') as f:
                    url = f.read().strip()
                print(f"URL: {url}")
            except Exception as e:
                print(f"Error reading info.txt for {folder}: {e}")
                print(traceback.format_exc())
                continue

            try:
                print(f"Starting attack...")
                # Run attack
                if args.attack_type == 'fgsm':
                    adv_image, attack_info = attacker.fgsm_attack(
                        image=screenshot_path,
                        url=url,
                        target_type=args.target_type,
                        num_steps=args.num_steps
                    )
                else:  # pgd
                    adv_image, attack_info = attacker.pgd_attack(
                        image=screenshot_path,
                        url=url,
                        target_type=args.target_type,
                        num_steps=args.num_steps
                    )

                print("Attack completed, saving results...")

                # Create sample output directory
                sample_output_dir = os.path.join(output_dir, folder)
                os.makedirs(sample_output_dir, exist_ok=True)

                # Save adversarial image
                adv_image_path = os.path.join(sample_output_dir, "adversarial.png")
                cv2.imwrite(adv_image_path, adv_image)
                print(f"Saved adversarial image to: {adv_image_path}")

                # Save original image for comparison
                orig_image_path = os.path.join(sample_output_dir, "original.png")
                orig_image = cv2.imread(screenshot_path)
                cv2.imwrite(orig_image_path, orig_image)
                print(f"Saved original image to: {orig_image_path}")

                # Update statistics
                total_samples += 1
                if attack_info['success']:
                    attack_success += 1
                    print("Attack successful!")

                # Save attack information
                result_dict = {
                    'folder': folder,
                    'url': url,
                    'success': convert_to_serializable(attack_info['success']),
                    'num_steps': convert_to_serializable(attack_info['num_steps_taken'])
                }

                # Save detailed attack info
                if args.target_type == 'detector':
                    if attack_info['original_detection'] is not None:
                        result_dict['original_num_logos'] = convert_to_serializable(
                            len(attack_info['original_detection']))
                    if attack_info['adversarial_detection'] is not None:
                        result_dict['adversarial_num_logos'] = convert_to_serializable(
                            len(attack_info['adversarial_detection']))
                else:  # matcher
                    if attack_info['original_matching'] is not None:
                        # Unpack tuple values (brand, domain, coord, confidence)
                        result_dict['original_brand'] = convert_to_serializable(
                            attack_info['original_matching'][0])  # matched_target
                        result_dict['original_domain'] = convert_to_serializable(
                            attack_info['original_matching'][1])  # matched_domain
                        result_dict['original_confidence'] = convert_to_serializable(
                            attack_info['original_matching'][3])  # siamese_conf
                    if attack_info['adversarial_matching'] is not None:
                        # Unpack tuple values (brand, domain, coord, confidence)
                        result_dict['adversarial_brand'] = convert_to_serializable(
                            attack_info['adversarial_matching'][0])  # matched_target
                        result_dict['adversarial_domain'] = convert_to_serializable(
                            attack_info['adversarial_matching'][1])  # matched_domain
                        result_dict['adversarial_confidence'] = convert_to_serializable(
                            attack_info['adversarial_matching'][3])  # siamese_conf

                # Save attack details for this sample
                details_path = os.path.join(sample_output_dir, "attack_details.json")
                with open(details_path, 'w') as f:
                    json.dump(result_dict, f, indent=2)
                print(f"Saved attack details to: {details_path}")

                # Add to results list
                results.append(result_dict)

            except Exception as e:
                print(f"Error processing {folder}: {str(e)}")
                print(traceback.format_exc())
                continue

        # Save final results
        if total_samples > 0:
            success_rate = (attack_success / total_samples) * 100
            
            # Save summary
            summary_file = os.path.join(output_dir, 'summary.txt')
            with open(summary_file, 'w') as f:
                f.write(f'Attack Type: {args.attack_type}\n')
                f.write(f'Target Type: {args.target_type}\n')
                f.write(f'Number of Steps: {args.num_steps}\n')
                f.write(f'Epsilon: {args.epsilon}\n')
                f.write(f'Total Samples: {total_samples}\n')
                f.write(f'Successful Attacks: {attack_success}\n')
                f.write(f'Success Rate: {success_rate:.2f}%\n')
            print(f"Saved summary to: {summary_file}")

            # Save detailed results
            results_file = os.path.join(output_dir, 'detailed_results.json')
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Saved detailed results to: {results_file}")

            print(f'\nAttack completed! Results saved to {output_dir}')
            print(f'Success Rate: {success_rate:.2f}% ({attack_success}/{total_samples})')
        else:
            print("No samples were successfully processed!")

    except Exception as e:
        print(f"Critical error: {str(e)}")
        print(traceback.format_exc())
        return

    finally:
        if args.debug:
            sys.stdout.close()
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__


if __name__ == '__main__':
    run_attacks()