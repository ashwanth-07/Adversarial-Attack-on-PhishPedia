import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
import tldextract
from typing import Tuple, List, Optional, Union
from logo_matching import check_domain_brand_inconsistency
import traceback
import json
from torchvision import transforms
from utils import brand_converter, resolution_alignment, l2_norm
import os
import pickle
import copy
from detectron2.structures import Instances, BoxMode

class PhishpediaAttacker:
    def __init__(
        self,
        rcnn_model,
        siamese_model: nn.Module,
        domain_map_path: str,
        logo_feats: np.ndarray,
        logo_files: np.ndarray,
        epsilon: float = 0.3,  # Increased epsilon
        momentum: float = 0.9,  # Added momentum
    ):
        self.rcnn_model = rcnn_model
        self.siamese_model = siamese_model
        self.domain_map_path = domain_map_path
        self.logo_feats = logo_feats
        self.logo_files = logo_files
        self.epsilon = epsilon
        self.momentum = momentum

    def compute_detection_loss(self, outputs: dict, input_tensor: torch.Tensor) -> torch.Tensor:
        """Enhanced loss function that targets both confidence scores and features"""
        try:
            instances = outputs['instances']
            scores = instances.scores
            boxes = instances.pred_boxes.tensor
            
            # Confidence loss
            confidence_loss = -torch.log(1 - scores + 1e-10).mean()
            
            # Feature disruption loss
            h, w = input_tensor.shape[2:]
            box_regions = torch.zeros((len(boxes), h, w), device=input_tensor.device)
            
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box.int()
                box_regions[i, y1:y2, x1:x2] = 1
                
            # Normalize box regions
            box_regions = box_regions / (box_regions.sum() + 1e-10)
            
            # Compute feature loss on box regions
            feature_loss = (input_tensor.squeeze(0).mean(0) * box_regions).sum()
            
            # Combined loss
            loss = confidence_loss + 0.1 * feature_loss
            
            print(f"Confidence loss: {confidence_loss.item()}, Feature loss: {feature_loss.item()}")
            return loss
            
        except Exception as e:
            print(f"Error in compute_detection_loss: {str(e)}")
            print(traceback.format_exc())
            raise

    def fgsm_attack(
        self, 
        image: str,
        url: str,
        target_type: str = "detector",
        num_steps: int = 10,  # Increased number of steps
        step_size: Optional[float] = None,
    ) -> Tuple[np.ndarray, dict]:
        """Enhanced FGSM attack with momentum for both detector and classifier"""
        try:
            if step_size is None:
                step_size = self.epsilon / (num_steps * 2)  # Smaller step size

            # Prepare images
            print(f"Preparing images for {target_type} attack...")
            original_image = self.prepare_image_for_rcnn(image)
            original_tensor = self.prepare_image_for_attack(original_image)
            
            # Parse URL
            if isinstance(url, (dict, str)):
                if isinstance(url, dict) and 'url' in url:
                    url = url['url']
                elif isinstance(url, str):
                    try:
                        url_dict = json.loads(url)
                        if isinstance(url_dict, dict) and 'url' in url_dict:
                            url = url_dict['url']
                    except:
                        pass
            print(f"Using URL: {url}")

            attack_info = {
                'success': False,
                'num_steps_taken': 0,
                'original_detection': None,
                'adversarial_detection': None,
                'original_matching': None,
                'adversarial_matching': None,
                'confidence_history': []  # Track confidence over steps
            }

            # Get original predictions
            print("Getting original predictions...")
            outputs = self.rcnn_model(original_image)
            attack_info['original_detection'] = outputs['instances'] if outputs else None
            
            if target_type == "classifier" and (not outputs or len(outputs['instances']) == 0):
                print("No logos detected to attack classifier")
                return original_image, attack_info

            # For classifier attack, get original matches
            original_matching = None
            if target_type == "classifier":
                original_boxes = outputs['instances'].pred_boxes.tensor
                # Get original matching results
                original_matching = check_domain_brand_inconsistency(
                    logo_boxes=original_boxes.cpu().numpy(),
                    domain_map_path=self.domain_map_path,
                    model=self.siamese_model,
                    logo_feat_list=self.logo_feats,
                    file_name_list=self.logo_files,
                    url=url,
                    shot_path=image,
                    ts=0.85,
                    topk=1
                )
                attack_info['original_matching'] = original_matching
                if original_matching and original_matching[3] is not None:
                    print(f"Original confidence: {original_matching[3]:.4f}")
                    attack_info['confidence_history'].append(original_matching[3])

            print(f"Found {len(outputs['instances'])} logos in original image")
            adv_tensor = original_tensor.clone()
            best_adv_tensor = None
            best_confidence = float('inf')
            
            # Initialize momentum
            grad_momentum = torch.zeros_like(adv_tensor)
            
            print("Starting attack iterations...")
            for step in range(num_steps):
                print(f"\nStep {step + 1}/{num_steps}")
                adv_tensor.requires_grad = True
                
                # Convert to RCNN format
                adv_image = self.tensor_to_rcnn_format(adv_tensor)
                
                # Get predictions
                outputs = self.rcnn_model(adv_image)
                
                if target_type == "detector":
                    if not outputs or len(outputs['instances']) == 0:
                        print("No detections found, attack succeeded!")
                        attack_info['success'] = True
                        best_adv_tensor = adv_tensor.clone()
                        break
                    
                    # Compute detector loss
                    loss = self.compute_detection_loss(outputs, adv_tensor)
                    current_score = len(outputs['instances'])
                    
                else:  # classifier attack
                    if not outputs or len(outputs['instances']) == 0:
                        continue
                        
                    # Compute classifier loss
                    boxes = outputs['instances'].pred_boxes.tensor
                    loss = self.compute_classifier_loss(adv_tensor, boxes)
                    
                    # Check current attack effectiveness
                    current_matching = check_domain_brand_inconsistency(
                        logo_boxes=boxes.cpu().numpy(),
                        domain_map_path=self.domain_map_path,
                        model=self.siamese_model,
                        logo_feat_list=self.logo_feats,
                        file_name_list=self.logo_files,
                        url=url,
                        shot_path=image,
                        ts=0.85,
                        topk=1
                    )
                    
                    current_confidence = current_matching[3] if current_matching else 0
                    attack_info['confidence_history'].append(current_confidence)
                    print(f"Current confidence: {current_confidence:.4f}")
                    
                    if current_matching and original_matching:
                        confidence_drop = original_matching[3] - current_confidence
                        print(f"Confidence drop: {confidence_drop:.4f}")
                    
                    current_score = current_confidence

                print(f"Step {step + 1} loss: {loss.item()}")

                # Update best adversarial example if current is better
                if current_score < best_confidence:
                    best_confidence = current_score
                    best_adv_tensor = adv_tensor.clone()

                # Compute gradients
                loss.backward()
                
                # Update image with momentum
                with torch.no_grad():
                    grad = adv_tensor.grad
                    if grad is not None:
                        # Update momentum
                        grad_momentum = self.momentum * grad_momentum + grad / torch.norm(grad, p=1)
                        # Update image
                        adv_tensor.data = adv_tensor.data + step_size * grad_momentum.sign()
                        # Project perturbation
                        delta = torch.clamp(adv_tensor.data - original_tensor.data,
                                          -self.epsilon, self.epsilon)
                        adv_tensor.data = torch.clamp(original_tensor.data + delta, 0, 1)
                    else:
                        print("No gradient computed!")
                
                attack_info['num_steps_taken'] = step + 1
                adv_tensor = adv_tensor.detach()

            print("\nCreating final adversarial image...")
            final_adv_tensor = best_adv_tensor if best_adv_tensor is not None else adv_tensor
            final_adv_image = self.tensor_to_rcnn_format(final_adv_tensor)

            print("Getting final predictions...")
            with torch.no_grad():
                final_outputs = self.rcnn_model(final_adv_image)
                attack_info['adversarial_detection'] = final_outputs['instances'] if final_outputs else None
                
                if target_type == "detector":
                    num_final_detections = len(final_outputs['instances']) if final_outputs else 0
                    initial_detections = len(attack_info['original_detection'])
                    attack_info['success'] = num_final_detections < initial_detections
                    print(f"Final number of detections: {num_final_detections}")
                else:  # classifier
                    if final_outputs and len(final_outputs['instances']) > 0:
                        boxes = final_outputs['instances'].pred_boxes.tensor
                        final_matching = check_domain_brand_inconsistency(
                            logo_boxes=boxes.cpu().numpy(),
                            domain_map_path=self.domain_map_path,
                            model=self.siamese_model,
                            logo_feat_list=self.logo_feats,
                            file_name_list=self.logo_files,
                            url=url,
                            shot_path=image,
                            ts=0.85,
                            topk=1
                        )
                        attack_info['adversarial_matching'] = final_matching
                        
                        # Print final confidence and total drop
                        if final_matching and original_matching:
                            final_confidence = final_matching[3]
                            initial_confidence = original_matching[3]
                            confidence_drop = initial_confidence - final_confidence
                            print(f"Initial confidence: {initial_confidence:.4f}")
                            print(f"Final confidence: {final_confidence:.4f}")
                            print(f"Total confidence drop: {confidence_drop:.4f}")
                        
                        # Success if brand prediction changed or confidence significantly decreased
                        attack_info['success'] = (
                            not final_matching or
                            final_matching[0] != original_matching[0] or
                            (final_matching[3] < original_matching[3] * 0.5)  # More aggressive threshold
                        )

            print(f"Attack {'succeeded' if attack_info['success'] else 'failed'}")
            return final_adv_image, attack_info

        except Exception as e:
            print(f"Error in fgsm_attack: {str(e)}")
            print(traceback.format_exc())
            raise

    def prepare_image_for_rcnn(self, image_path: str) -> np.ndarray:
        """Prepare image in format expected by RCNN"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not read image at {image_path}")
            print(f"Image shape: {image.shape}")
            return image
        except Exception as e:
            print(f"Error in prepare_image_for_rcnn: {str(e)}")
            print(traceback.format_exc())
            raise

    def prepare_image_for_attack(self, image: np.ndarray) -> torch.Tensor:
        """Convert BGR image to normalized RGB tensor"""
        try:
            print(f"Input image shape: {image.shape}")
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(rgb_image).float() / 255.0
            tensor = tensor.permute(2, 0, 1)
            if tensor.dim() == 3:
                tensor = tensor.unsqueeze(0)
            print(f"Final tensor shape: {tensor.shape}")
            return tensor
        except Exception as e:
            print(f"Error in prepare_image_for_attack: {str(e)}")
            print(traceback.format_exc())
            raise

    def tensor_to_rcnn_format(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert normalized RGB tensor to BGR image for RCNN"""
        try:
            print(f"Input tensor shape: {tensor.shape}")
            image = tensor.detach().squeeze(0).cpu().numpy()
            image = np.transpose(image, (1, 2, 0))
            image = (image * 255).astype(np.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            print(f"Final image shape: {image.shape}")
            return image
        except Exception as e:
            print(f"Error in tensor_to_rcnn_format: {str(e)}")
            print(traceback.format_exc())
            raise


    def compute_classifier_loss(self, image_tensor: torch.Tensor, detected_boxes: torch.Tensor) -> torch.Tensor:
        """Compute loss for attacking the Siamese classifier"""
        try:
            img_transforms = transforms.Compose([
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
            
            # Extract logo regions using detected boxes
            logo_tensors = []
            for box in detected_boxes:
                x1, y1, x2, y2 = map(int, box.tolist())
                # Ensure valid box coordinates
                x1, x2 = max(0, x1), min(image_tensor.shape[3], x2)
                y1, y2 = max(0, y1), min(image_tensor.shape[2], y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                    
                logo_region = image_tensor[:, :, y1:y2, x1:x2]
                
                # Resize to match expected input size
                logo_region = torch.nn.functional.interpolate(
                    logo_region, 
                    size=(128, 128),  # Match the size used in get_embedding
                    mode='bilinear', 
                    align_corners=False
                )
                
                # Apply normalization
                logo_region = img_transforms(logo_region.squeeze(0))
                logo_region = logo_region.unsqueeze(0)
                logo_tensors.append(logo_region)
                
            if not logo_tensors:
                return torch.tensor(0.0, device=image_tensor.device, requires_grad=True)
                
            # Stack all logo tensors
            logo_batch = torch.cat(logo_tensors, dim=0)
                
            # Get features for detected logos using Siamese model
            logo_features = []
            for logo in logo_batch:
                logo_input = logo.unsqueeze(0)
                feat = self.siamese_model.features(logo_input)  # Get features directly like in get_embedding
                feat = l2_norm(feat).squeeze(0)  # L2-normalize like in get_embedding
                logo_features.append(feat)
                
            # Stack features
            logo_features = torch.stack(logo_features)
            
            # Convert reference features to tensor
            ref_feats = torch.tensor(self.logo_feats, device=image_tensor.device)
            
            print(f"Logo features shape: {logo_features.shape}")
            print(f"Reference features shape: {ref_feats.shape}")
            
            # Compute cosine similarities
            similarities = torch.mm(logo_features, ref_feats.t())
            
            # Sort similarities for each logo
            sorted_sims, _ = torch.sort(similarities, dim=1, descending=True)
            
            # Loss: maximize similarity with incorrect matches (top 3-10)
            # while minimizing similarity with correct match (top-1)
            incorrect_matches = sorted_sims[:, 3:10]  # Use positions 3-10 as incorrect matches
            correct_match = sorted_sims[:, 0:1]  # Assumed correct match is top-1
            
            loss = -torch.mean(incorrect_matches) + torch.mean(correct_match)
            
            return loss
                
        except Exception as e:
            print(f"Error in compute_classifier_loss: {str(e)}")
            print(traceback.format_exc())
            raise
            
            # Sort similarities for each logo
            sorted_sims, _ = torch.sort(similarities, dim=1, descending=True)
            
            # Loss: maximize similarity with incorrect matches (top 3-10)
            # while minimizing similarity with correct match (top-1)
            incorrect_matches = sorted_sims[:, 3:10]  # Use positions 3-10 as incorrect matches
            correct_match = sorted_sims[:, 0:1]  # Assumed correct match is top-1
            
            loss = -torch.mean(incorrect_matches) + torch.mean(correct_match)
            
            return loss
                
        except Exception as e:
            print(f"Error in compute_classifier_loss: {str(e)}")
            print(traceback.format_exc())
            raise
            
        except Exception as e:
            print(f"Error in compute_classifier_loss: {str(e)}")
            print(traceback.format_exc())
            raise

    def pgd_attack(
        self,
        image: str,
        url: str,
        target_type: str = "detector",
        num_steps: int = 100,  # Increased number of steps
        step_size: Optional[float] = None,
        random_start: bool = True
    ) -> Tuple[np.ndarray, dict]:
        """
        Performs a more aggressive PGD attack on either the detector or classifier.
        
        Args:
            image: Path to the input image
            url: URL associated with the image 
            target_type: Type of attack target ("detector" or "classifier")
            num_steps: Number of PGD iterations
            step_size: Step size for each iteration (default: epsilon/3)
            random_start: Whether to initialize with random perturbation

        Returns:
            Tuple containing (adversarial image, attack info dictionary)
        """
        try:
            if step_size is None:
                step_size = self.epsilon / 3  # Larger step size

            # Prepare images
            print(f"Preparing images for {target_type} attack...")
            original_image = self.prepare_image_for_rcnn(image)
            original_tensor = self.prepare_image_for_attack(original_image)
            
            # Parse URL
            if isinstance(url, (dict, str)):
                if isinstance(url, dict) and 'url' in url:
                    url = url['url']
                elif isinstance(url, str):
                    try:
                        url_dict = json.loads(url)
                        if isinstance(url_dict, dict) and 'url' in url_dict:
                            url = url_dict['url']
                    except:
                        pass
            print(f"Using URL: {url}")

            attack_info = {
                'success': False,
                'num_steps_taken': 0,
                'original_detection': None,
                'adversarial_detection': None,
                'original_matching': None,
                'adversarial_matching': None,
                'confidence_history': []  # Track confidence over steps
            }

            # Get original predictions
            print("Getting original predictions...")
            outputs = self.rcnn_model(original_image)
            attack_info['original_detection'] = outputs['instances'] if outputs else None
            
            if target_type == "classifier" and (not outputs or len(outputs['instances']) == 0):
                print("No logos detected to attack classifier")
                return original_image, attack_info

            # For classifier attack, get original matches
            if target_type == "classifier":
                original_boxes = outputs['instances'].pred_boxes.tensor
                # Get original matching results
                original_matching = check_domain_brand_inconsistency(
                    logo_boxes=original_boxes.cpu().numpy(),
                    domain_map_path=self.domain_map_path,
                    model=self.siamese_model,
                    logo_feat_list=self.logo_feats,
                    file_name_list=self.logo_files,
                    url=url,
                    shot_path=image,
                    ts=0.85,
                    topk=1
                )
                attack_info['original_matching'] = original_matching
                if original_matching and original_matching[3] is not None:
                    print(f"Original confidence: {original_matching[3]:.4f}")
                    attack_info['confidence_history'].append(original_matching[3])

            # Initialize adversarial example
            adv_tensor = original_tensor.clone()
            
            if random_start:
                print("Applying random initialization...")
                # More aggressive random initialization
                random_noise = torch.zeros_like(adv_tensor).uniform_(-self.epsilon, self.epsilon)
                adv_tensor = torch.clamp(adv_tensor + random_noise, 0, 1)

            best_adv_tensor = None
            best_confidence = float('inf') if target_type == "classifier" else float('inf')
            num_no_improvement = 0  # Counter for steps without improvement
            
            print(f"Starting PGD iterations for {target_type} attack...")
            for step in range(num_steps):
                print(f"\nStep {step + 1}/{num_steps}")
                adv_tensor.requires_grad = True
                
                # Convert to RCNN format
                adv_image = self.tensor_to_rcnn_format(adv_tensor)
                
                # Get predictions
                outputs = self.rcnn_model(adv_image)
                
                if target_type == "detector":
                    if not outputs or len(outputs['instances']) == 0:
                        print("No detections found, attack succeeded!")
                        attack_info['success'] = True
                        best_adv_tensor = adv_tensor.clone()
                        break
                    
                    # Compute detector loss
                    loss = self.compute_detection_loss(outputs, adv_tensor)
                    current_score = len(outputs['instances'])
                    
                else:  # classifier attack
                    if not outputs or len(outputs['instances']) == 0:
                        continue
                        
                    # Compute classifier loss
                    boxes = outputs['instances'].pred_boxes.tensor
                    loss = self.compute_classifier_loss(adv_tensor, boxes)
                    
                    # Check current attack effectiveness
                    current_matching = check_domain_brand_inconsistency(
                        logo_boxes=boxes.cpu().numpy(),
                        domain_map_path=self.domain_map_path,
                        model=self.siamese_model,
                        logo_feat_list=self.logo_feats,
                        file_name_list=self.logo_files,
                        url=url,
                        shot_path=image,
                        ts=0.85,
                        topk=1
                    )
                    
                    current_confidence = current_matching[3] if current_matching else 0
                    attack_info['confidence_history'].append(current_confidence)
                    print(f"Current confidence: {current_confidence:.4f}")
                    
                    if current_matching and original_matching:
                        confidence_drop = original_matching[3] - current_confidence
                        print(f"Confidence drop: {confidence_drop:.4f}")
                    
                    current_score = current_confidence
                
                print(f"Step {step + 1} loss: {loss.item()}")
                
                # Update best adversarial example if current is better
                if current_score < best_confidence:
                    best_confidence = current_score
                    best_adv_tensor = adv_tensor.clone()
                    num_no_improvement = 0
                else:
                    num_no_improvement += 1
                
                # Early stopping if no improvement for many steps
                if num_no_improvement > 20:  # Increase patience
                    print("No improvement for many steps, stopping attack")
                    break
                
                # Compute gradients
                loss.backward()
                
                # Update image with stronger update
                with torch.no_grad():
                    grad = adv_tensor.grad
                    if grad is not None:
                        # Normalized gradient step
                        grad_norm = torch.norm(grad, p=float('inf'))
                        normalized_grad = grad / (grad_norm + 1e-10)
                        
                        # More aggressive step
                        adv_tensor.data = adv_tensor.data + step_size * 1.5 * normalized_grad.sign()
                        
                        # Project back to epsilon ball
                        delta = torch.clamp(adv_tensor.data - original_tensor.data, 
                                          -self.epsilon, self.epsilon)
                        adv_tensor.data = torch.clamp(original_tensor.data + delta, 0, 1)
                    else:
                        print("No gradient computed!")
                
                attack_info['num_steps_taken'] = step + 1
                adv_tensor = adv_tensor.detach()

            print("\nCreating final adversarial image...")
            final_adv_tensor = best_adv_tensor if best_adv_tensor is not None else adv_tensor
            final_adv_image = self.tensor_to_rcnn_format(final_adv_tensor)

            # Get final predictions and determine success
            print("Getting final predictions...")
            with torch.no_grad():
                final_outputs = self.rcnn_model(final_adv_image)
                attack_info['adversarial_detection'] = final_outputs['instances'] if final_outputs else None
                
                if target_type == "detector":
                    num_final_detections = len(final_outputs['instances']) if final_outputs else 0
                    initial_detections = len(attack_info['original_detection'])
                    attack_info['success'] = num_final_detections < initial_detections
                    print(f"Final number of detections: {num_final_detections}")
                else:  # classifier
                    if final_outputs and len(final_outputs['instances']) > 0:
                        boxes = final_outputs['instances'].pred_boxes.tensor
                        final_matching = check_domain_brand_inconsistency(
                            logo_boxes=boxes.cpu().numpy(),
                            domain_map_path=self.domain_map_path,
                            model=self.siamese_model,
                            logo_feat_list=self.logo_feats,
                            file_name_list=self.logo_files,
                            url=url,
                            shot_path=image,
                            ts=0.85,
                            topk=1
                        )
                        attack_info['adversarial_matching'] = final_matching
                        
                        # Print final confidence and total drop
                        if final_matching and original_matching:
                            final_confidence = final_matching[3]
                            initial_confidence = original_matching[3]
                            confidence_drop = initial_confidence - final_confidence
                            print(f"Initial confidence: {initial_confidence:.4f}")
                            print(f"Final confidence: {final_confidence:.4f}")
                            print(f"Total confidence drop: {confidence_drop:.4f}")
                        
                        # Success if brand prediction changed or confidence significantly decreased
                        attack_info['success'] = (
                            not final_matching or
                            final_matching[0] != original_matching[0] or
                            (final_matching[3] < original_matching[3] * 0.5)  # More aggressive threshold
                        )

            print(f"Attack {'succeeded' if attack_info['success'] else 'failed'}")
            return final_adv_image, attack_info

        except Exception as e:
            print(f"Error in pgd_attack: {str(e)}")
            print(traceback.format_exc())
            raise