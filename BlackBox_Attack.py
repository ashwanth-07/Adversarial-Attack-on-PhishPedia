import os
import cv2
import torch
import numpy as np
from PIL import Image
import random
from torchvision import transforms
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from phishpedia import PhishpediaWrapper
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import re

class PhishAttacker:
    def __init__(self, phishpedia_wrapper, output_dir, num_variations=5):
        """
        Initialize the phishing detector attack framework
        
        Args:
            phishpedia_wrapper: Instance of PhishpediaWrapper to attack
            output_dir: Directory to save attack results
            num_variations: Number of attack variations to try per type
        """
        self.model = phishpedia_wrapper
        self.output_dir = output_dir
        self.num_variations = num_variations
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        self.results_dir = os.path.join(output_dir, 'attack_results')
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Initialize results tracking
        self.results_df = pd.DataFrame(columns=[
            'folder', 'url', 'attack_type', 'variation_id', 
            'original_phish_category', 'attacked_phish_category',
            'original_target', 'attacked_target',
            'psnr', 'ssim', 'l2_distance',
            'attack_success', 'logo_recog_time', 'logo_match_time'
        ])
        
    def attack_dataset(self, dataset_dir):
        """Run attacks on all images in the dataset directory"""
        _forbidden_suffixes = r"\.(mp3|wav|wma|ogg|mkv|zip|tar|xz|rar|z|deb|bin|iso|csv|tsv|dat|txt|css|log|xml|sql|mdb|apk|bat|exe|jar|wsf|fnt|fon|otf|ttf|ai|bmp|gif|ico|jp(e)?g|png|ps|psd|svg|tif|tiff|cer|rss|key|odp|pps|ppt|pptx|c|class|cpp|cs|h|java|sh|swift|vb|odf|xlr|xls|xlsx|bak|cab|cfg|cpl|cur|dll|dmp|drv|icns|ini|lnk|msi|sys|tmp|3g2|3gp|avi|flv|h264|m4v|mov|mp4|mp(e)?g|rm|swf|vob|wmv|doc(x)?|odt|rtf|tex|wks|wps|wpd)$"
        
        print("\n=== Starting Attack Campaign ===")
        print(f"Dataset directory: {dataset_dir}")
        print(f"Number of variations per attack: {self.num_variations}")
        
        total_folders = len(os.listdir(dataset_dir))
        print(f"Total folders to process: {total_folders}")
        
        for folder in tqdm(os.listdir(dataset_dir)):
            folder_path = os.path.join(dataset_dir, folder)
            if not os.path.isdir(folder_path):
                continue
                
            # Get paths
            screenshot_path = os.path.join(folder_path, "shot.png")
            html_path = os.path.join(folder_path, "html.txt")
            info_path = os.path.join(folder_path, 'info.txt')
            
            # Check if required files exist
            if not os.path.exists(screenshot_path):
                continue
            if not os.path.exists(html_path):
                html_path = os.path.join(folder_path, "index.html")
            if not os.path.exists(info_path):
                continue
                
            # Read URL
            with open(info_path, 'r') as file:
                url = file.read().strip()
                
            # Skip forbidden file types
            if re.search(_forbidden_suffixes, url, re.IGNORECASE):
                continue
                
            # Run attacks
            try:
                self.attack_single_image(folder, url, screenshot_path, html_path)
            except Exception as e:
                print(f"Error processing {folder}: {str(e)}")
                continue
                
        # Save final results
        today = datetime.now().strftime('%Y%m%d')
        self.results_df.to_csv(os.path.join(self.output_dir, f'{today}_attack_results.csv'), index=False)
        self._generate_summary_plots()
        
    def attack_single_image(self, folder, url, screenshot_path, html_path):
        """Run all attacks on a single screenshot"""
        try:
            print(f"\n=== Processing {folder} ===")
            print(f"URL: {url}")
            print(f"Screenshot: {screenshot_path}")
            
            # Get original prediction
            print("Getting original prediction...")
            orig_phish_category, orig_target, orig_domain, orig_plotvis, orig_conf, \
                orig_boxes, orig_logo_time, orig_match_time = self.model.test_orig_phishpedia(url, screenshot_path, html_path)
            
            print(f"Original prediction: {'Phishing' if orig_phish_category else 'Benign'}")
            if orig_phish_category:
                print(f"Target brand: {orig_target}")
                print(f"Confidence: {orig_conf:.4f}")
            
            # Save original visualization if phishing
            if orig_phish_category:
                os.makedirs(os.path.join(self.results_dir, folder), exist_ok=True)
                if orig_plotvis is not None:
                    save_path = os.path.join(self.results_dir, folder, "original.png")
                    cv2.imwrite(save_path, orig_plotvis)
                    print(f"Saved original visualization to {save_path}")
            
            # Load and verify original image
            print("Loading original image...")
            orig_img = cv2.imread(screenshot_path)
            if orig_img is None:
                print(f"Error: Could not load image from {screenshot_path}")
                return
                
            # Check image dimensions
            if orig_img.shape[0] == 0 or orig_img.shape[1] == 0:
                print(f"Error: Invalid image dimensions for {screenshot_path}")
                return
                
            print(f"Image loaded successfully. Dimensions: {orig_img.shape}")

        except e:
            print(e)
            
        
        # Run each attack type
        attack_types = ['noise']#['jpeg', 'noise', 'spatial', 'color']
        for attack_type in attack_types:
            for variation_id in range(self.num_variations):
                # Apply attack
                print("Applying attack transformation...")
                attacked_img = self._apply_attack(orig_img.copy(), attack_type, variation_id)
                
                # Save attacked image temporarily
                temp_path = os.path.join(self.results_dir, f'temp_{attack_type}_{variation_id}.png')
                cv2.imwrite(temp_path, attacked_img)
                print(f"Saved attacked image to: {temp_path}")
                
                # Get prediction on attacked image
                print("Getting prediction on attacked image...")
                attacked_phish_category, attacked_target, attacked_domain, attacked_plotvis, \
                    attacked_conf, attacked_boxes, logo_time, match_time = self.model.test_orig_phishpedia(url, temp_path, html_path)
                
                print(f"Attack prediction: {'Phishing' if attacked_phish_category else 'Benign'}")
                if attacked_phish_category:
                    print(f"Attacked target brand: {attacked_target}")
                    print(f"Attacked confidence: {attacked_conf:.4f}")
                
                
                # Calculate metrics
                metrics = self._calculate_metrics(orig_img, attacked_img)
                
                # Determine if attack was successful
                attack_success = orig_phish_category == 1 and attacked_phish_category == 0
                
                # Save successful attack visualization
                if attack_success:
                    save_dir = os.path.join(self.results_dir, folder)
                    os.makedirs(save_dir, exist_ok=True)
                    cv2.imwrite(os.path.join(save_dir, f"{attack_type}_variation_{variation_id}.png"), attacked_plotvis)
                
                # Record results
                self.results_df = pd.concat([self.results_df, pd.DataFrame([{
                    'folder': folder,
                    'url': url,
                    'attack_type': attack_type,
                    'variation_id': variation_id,
                    'original_phish_category': orig_phish_category,
                    'attacked_phish_category': attacked_phish_category,
                    'original_target': orig_target,
                    'attacked_target': attacked_target,
                    'psnr': metrics['psnr'],
                    'ssim': metrics['ssim'],
                    'l2_distance': metrics['l2'],
                    'attack_success': attack_success,
                    'logo_recog_time': logo_time,
                    'logo_match_time': match_time
                }])], ignore_index=True)
                
                # Clean up temp file
                os.remove(temp_path)
    
    def _apply_attack(self, image, attack_type, variation_id):
        """Apply specified attack type to image"""
        if attack_type == 'jpeg':
            return self._jpeg_compression_attack(image, variation_id)
        elif attack_type == 'noise':
            return self._noise_attack(image, variation_id)
        elif attack_type == 'spatial':
            return self._spatial_attack(image, variation_id)
        elif attack_type == 'color':
            return self._color_attack(image, variation_id)
        else:
            raise ValueError(f'Unknown attack type: {attack_type}')
    
    def _jpeg_compression_attack(self, image, variation_id):
        """Apply JPEG compression attack"""
        quality = random.randint(10, 30)  # Random quality between 10-30
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, enc_img = cv2.imencode('.jpg', image, encode_param)
        dec_img = cv2.imdecode(enc_img, 1)
        return dec_img
    
    def _noise_attack(self, image, variation_id):
        """Apply random noise attack"""
        noise = np.random.normal(0, 1, image.shape).astype(np.uint8)
        noisy_img = cv2.add(image, noise)
        return noisy_img
    
    def _spatial_attack(self, image, variation_id):
        """Apply spatial transformation attack"""
        try:
            # Convert to PIL for easier transformations
            image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # Get image dimensions
            w, h = image_pil.size
            
            # Random spatial parameters
            angle = random.uniform(-10, 10)
            scale = random.uniform(0.9, 1.1)
            
            # Calculate translate as fraction of image size (must be between 0 and 1)
            translate = (
                random.uniform(-0.1, 0.1),  # 10% of image width max
                random.uniform(-0.1, 0.1)   # 10% of image height max
            )
            
            # Apply transformations
            transform = transforms.Compose([
                transforms.RandomAffine(
                    degrees=angle,
                    translate=translate,
                    scale=(scale, scale),
                    fillcolor=(255, 255, 255)
                )
            ])
            
            transformed = transform(image_pil)
            return cv2.cvtColor(np.array(transformed), cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"Error in spatial attack: {str(e)}")
            # Return original image if transformation fails
            return image.copy()
    
    def _color_attack(self, image, variation_id):
        """Apply color manipulation attack"""
        # Random color adjustment parameters
        brightness = random.uniform(0.8, 1.2)
        contrast = random.uniform(0.8, 1.2)
        hue = random.uniform(-0.1, 0.1)
        
        # Convert to HSV for easier color manipulation
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        
        # Apply adjustments
        hsv[..., 0] = np.clip(hsv[..., 0] + hue * 180, 0, 180)  # Hue
        hsv[..., 1] = np.clip(hsv[..., 1] * contrast, 0, 255)   # Saturation
        hsv[..., 2] = np.clip(hsv[..., 2] * brightness, 0, 255) # Value
        
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    def _calculate_metrics(self, original, attacked):
        """Calculate similarity metrics between original and attacked images"""
        return {
            'psnr': psnr(original, attacked),
            'ssim': ssim(original, attacked, channel_axis=2),
            'l2': np.sqrt(np.mean((original - attacked) ** 2))
        }
    
    def _generate_summary_plots(self):
        """Generate summary plots of attack results"""
        # Calculate success rates by attack type
        success_rates = self.results_df[self.results_df['original_phish_category'] == 1].groupby('attack_type')['attack_success'].mean()
        
        # Plot success rates
        plt.figure(figsize=(10, 6))
        success_rates.plot(kind='bar')
        plt.title('Attack Success Rates by Attack Type')
        plt.xlabel('Attack Type')
        plt.ylabel('Success Rate')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'success_rates.png'))
        plt.close()
        
        # Plot metric distributions
        metrics = ['psnr', 'ssim', 'l2_distance']
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for i, metric in enumerate(metrics):
            sns.boxplot(data=self.results_df, x='attack_type', y=metric, ax=axes[i])
            axes[i].set_title(f'{metric.upper()} by Attack Type')
            axes[i].tick_params(axis='x', rotation=45)
            
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'metric_distributions.png'))
        plt.close()

def main():
    # Example usage
    import argparse
    from datetime import datetime
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True, type=str, help="Input dataset folder")
    parser.add_argument("--output", default=f'attack_results_{datetime.now().strftime("%Y%m%d")}', 
                       help="Output directory for results")
    parser.add_argument("--variations", type=int, default=5, 
                       help="Number of variations to try per attack type")
    args = parser.parse_args()
    
    # Initialize models
    phishpedia = PhishpediaWrapper()
    attacker = PhishAttacker(phishpedia, args.output, args.variations)
    
    # Run attacks
    attacker.attack_dataset(args.folder)

if __name__ == "__main__":
    main()