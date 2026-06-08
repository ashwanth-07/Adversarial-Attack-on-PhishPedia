import os
import json
import tempfile
import traceback
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from logo_matching import check_domain_brand_inconsistency
from utils import l2_norm


class PhishpediaAttacker:
    """Adversarial attacks (FGSM / PGD) against the two-stage Phishpedia pipeline.

    Two fixes over the previous implementation are central here:

    1. The matcher is evaluated on the *adversarial* image, not the clean file
       on disk. ``check_domain_brand_inconsistency`` -> ``pred_brand`` reopens
       ``shot_path`` from disk, so previously every "current confidence" was the
       clean image's confidence and the perturbation never affected the metric.
       We now write the perturbed image to a temp file and feed *that* path.

    2. The detector loss is computed through a real differentiable forward pass.
       ``DefaultPredictor`` takes a NumPy array and runs under ``torch.no_grad``,
       so no gradient reached the input. We call the underlying detectron2 model
       directly with a grad-bearing tensor instead.

    Both losses are written so that *lower is better* (weaker detection / weaker
    brand match), and the optimizer performs gradient **descent** accordingly.
    """

    def __init__(
        self,
        rcnn_model,
        siamese_model,
        domain_map_path: str,
        logo_feats: np.ndarray,
        logo_files: np.ndarray,
        epsilon: float = 0.3,
        momentum: float = 0.9,
    ):
        self.rcnn_model = rcnn_model            # detectron2 DefaultPredictor (used for eval)
        self.siamese_model = siamese_model
        self.domain_map_path = domain_map_path
        self.logo_feats = logo_feats
        self.logo_files = logo_files
        self.epsilon = epsilon
        self.momentum = momentum

        # Underlying differentiable detector + its test-time preprocessing.
        self.detector_model = getattr(rcnn_model, "model", None)
        self.input_format = getattr(rcnn_model, "input_format", "BGR")

        # ResizeShortestEdge parameters used at test time (replicated below so the
        # differentiable forward matches the DefaultPredictor inference pipeline).
        self._min_size, self._max_size = 800, 1333
        aug = getattr(rcnn_model, "aug", None)
        if aug is not None:
            se = getattr(aug, "short_edge_length", None)
            if isinstance(se, (list, tuple)) and len(se) > 0:
                self._min_size = int(se[0])
            elif isinstance(se, int):
                self._min_size = int(se)
            self._max_size = int(getattr(aug, "max_size", self._max_size))

        # Run everything on the model's device.
        try:
            self.device = next(self.siamese_model.parameters()).device
        except StopIteration:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------ #
    # image <-> tensor helpers
    # ------------------------------------------------------------------ #
    def prepare_image_for_rcnn(self, image_path: str) -> np.ndarray:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image at {image_path}")
        if image.shape[-1] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        return image

    def prepare_image_for_attack(self, image: np.ndarray) -> torch.Tensor:
        """BGR uint8 (H, W, C) -> RGB float [0, 1] tensor (1, C, H, W)."""
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).float() / 255.0
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device)

    def tensor_to_rcnn_format(self, tensor: torch.Tensor) -> np.ndarray:
        """RGB float [0, 1] (1, C, H, W) -> BGR uint8 (H, W, C) for the predictor."""
        image = tensor.detach().clamp(0, 1).squeeze(0).cpu().numpy()
        image = np.transpose(image, (1, 2, 0))
        image = (image * 255.0).round().astype(np.uint8)
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    def _write_temp_image(self, tensor: torch.Tensor) -> str:
        """Persist the adversarial image so the matcher reads the *perturbed*
        pixels (it loads ``shot_path`` from disk)."""
        img = self.tensor_to_rcnn_format(tensor)
        fd, path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        cv2.imwrite(path, img)
        return path

    @staticmethod
    def _parse_url(url):
        if isinstance(url, dict):
            return url.get("url", url)
        if isinstance(url, str):
            try:
                parsed = json.loads(url)
                if isinstance(parsed, dict) and "url" in parsed:
                    return parsed["url"]
            except (ValueError, TypeError):
                pass
        return url

    # ------------------------------------------------------------------ #
    # differentiable detector
    # ------------------------------------------------------------------ #
    def _resize_shortest_edge(self, img: torch.Tensor) -> torch.Tensor:
        """Differentiable replica of detectron2's ResizeShortestEdge (test mode)."""
        _, h, w = img.shape
        size = float(self._min_size)
        scale = size / min(h, w)
        if h < w:
            newh, neww = size, scale * w
        else:
            newh, neww = scale * h, size
        if max(newh, neww) > self._max_size:
            s = self._max_size / max(newh, neww)
            newh, neww = newh * s, neww * s
        newh, neww = int(newh + 0.5), int(neww + 0.5)
        return F.interpolate(
            img.unsqueeze(0), size=(newh, neww), mode="bilinear", align_corners=False
        ).squeeze(0)

    def detector_forward(self, adv_tensor: torch.Tensor) -> dict:
        """Forward pass that keeps a gradient path from the detector's outputs
        back to ``adv_tensor`` (unlike DefaultPredictor, which detaches)."""
        if self.detector_model is None:
            raise RuntimeError(
                "Underlying detectron2 model unavailable; cannot run a "
                "differentiable detector attack."
            )
        _, _, H, W = adv_tensor.shape
        img = adv_tensor.squeeze(0) * 255.0          # (3, H, W) RGB, 0-255
        if self.input_format == "BGR":
            img = img.flip(0)                         # RGB -> BGR
        img = self._resize_shortest_edge(img)
        inputs = {"image": img.to(self.device), "height": H, "width": W}
        return self.detector_model([inputs])[0]

    # ------------------------------------------------------------------ #
    # losses (lower == better for the attacker)
    # ------------------------------------------------------------------ #
    def compute_detection_loss(self, adv_tensor: torch.Tensor):
        """Detection-strength loss. Minimizing it drives box confidences toward 0.

        Returns (loss, mean_score, num_boxes). loss is None when nothing is
        detected (the detector is already evaded)."""
        outputs = self.detector_forward(adv_tensor)
        instances = outputs["instances"]
        if len(instances) == 0:
            return None, 0.0, 0
        scores = instances.scores
        loss = -torch.log(1.0 - scores + 1e-6).mean()   # >= 0, minimized as scores -> 0
        return loss, float(scores.mean().item()), len(instances)

    def compute_classifier_loss(self, adv_tensor: torch.Tensor, boxes: torch.Tensor):
        """Siamese-matching loss. Minimizing it pushes the top (true) match
        similarity down and distractor similarities up -> brand evasion."""
        _, _, Ht, Wt = adv_tensor.shape
        mean = torch.tensor([0.5, 0.5, 0.5], device=adv_tensor.device).view(1, 3, 1, 1)
        std = torch.tensor([0.5, 0.5, 0.5], device=adv_tensor.device).view(1, 3, 1, 1)

        logos = []
        for box in boxes:
            x1, y1, x2, y2 = (int(v) for v in box.tolist())
            x1, x2 = max(0, x1), min(Wt, x2)
            y1, y2 = max(0, y1), min(Ht, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = adv_tensor[:, :, y1:y2, x1:x2]            # (1, 3, h, w)
            _, _, ch, cw = crop.shape
            m = max(ch, cw)
            # Pad to square with white (1.0), mirroring get_embedding's ImageOps.expand.
            pad = ((m - cw) // 2, (m - cw) - (m - cw) // 2,
                   (m - ch) // 2, (m - ch) - (m - ch) // 2)
            crop = F.pad(crop, pad, mode="constant", value=1.0)
            crop = F.interpolate(crop, size=(128, 128), mode="bilinear", align_corners=False)
            crop = (crop - mean) / std
            logos.append(crop)

        if not logos:
            return None

        batch = torch.cat(logos, dim=0)
        feats = l2_norm(self.siamese_model.features(batch))          # (N, 2048)
        ref = torch.as_tensor(self.logo_feats, dtype=feats.dtype, device=feats.device)
        sims = feats @ ref.t()                                       # (N, R)
        sorted_sims, _ = torch.sort(sims, dim=1, descending=True)
        correct = sorted_sims[:, 0:1]            # strongest (assumed true) match
        incorrect = sorted_sims[:, 3:10]         # distractors
        return correct.mean() - incorrect.mean()

    # ------------------------------------------------------------------ #
    # matcher evaluation (always on the adversarial image)
    # ------------------------------------------------------------------ #
    def _match(self, tensor: torch.Tensor, boxes: np.ndarray, url: str):
        tmp = self._write_temp_image(tensor)
        try:
            return check_domain_brand_inconsistency(
                logo_boxes=boxes,
                domain_map_path=self.domain_map_path,
                model=self.siamese_model,
                logo_feat_list=self.logo_feats,
                file_name_list=self.logo_files,
                url=url,
                shot_path=tmp,
                ts=0.85,
                topk=1,
            )
        finally:
            os.remove(tmp)

    # ------------------------------------------------------------------ #
    # shared attack loop
    # ------------------------------------------------------------------ #
    def _run_attack(
        self,
        image: str,
        url: str,
        target_type: str,
        num_steps: int,
        step_size: float,
        mode: str,
        random_start: bool = False,
        patience: Optional[int] = None,
    ) -> Tuple[np.ndarray, dict]:
        original_image = self.prepare_image_for_rcnn(image)
        original_tensor = self.prepare_image_for_attack(original_image)
        url = self._parse_url(url)
        print(f"Using URL: {url}")

        attack_info = {
            "success": False,
            "num_steps_taken": 0,
            "original_detection": None,
            "adversarial_detection": None,
            "original_matching": None,
            "adversarial_matching": None,
            "confidence_history": [],
        }

        outputs = self.rcnn_model(original_image)
        attack_info["original_detection"] = outputs["instances"] if outputs else None

        if target_type == "classifier":
            if not outputs or len(outputs["instances"]) == 0:
                print("No logos detected to attack the classifier")
                return original_image, attack_info
            orig_boxes = outputs["instances"].pred_boxes.tensor.detach().cpu().numpy()
            # Baseline match is measured on the clean image (correct for the baseline).
            original_matching = self._match(original_tensor, orig_boxes, url)
            attack_info["original_matching"] = original_matching
            if original_matching and original_matching[3] is not None:
                attack_info["confidence_history"].append(float(original_matching[3]))
                print(f"Original confidence: {original_matching[3]:.4f}")

        adv_tensor = original_tensor.clone()
        if random_start:
            noise = torch.empty_like(adv_tensor).uniform_(-self.epsilon, self.epsilon)
            adv_tensor = torch.clamp(adv_tensor + noise, 0, 1)

        grad_momentum = torch.zeros_like(adv_tensor)
        best_tensor, best_score = None, float("inf")
        no_improve = 0

        for step in range(num_steps):
            print(f"\nStep {step + 1}/{num_steps}")
            adv_tensor.requires_grad_(True)

            if target_type == "detector":
                loss, mean_score, n = self.compute_detection_loss(adv_tensor)
                if loss is None:
                    print("No detections found, attack succeeded!")
                    attack_info["success"] = True
                    best_tensor = adv_tensor.detach().clone()
                    attack_info["num_steps_taken"] = step + 1
                    break
                score = mean_score
            else:
                adv_np = self.tensor_to_rcnn_format(adv_tensor)
                cur = self.rcnn_model(adv_np)
                if not cur or len(cur["instances"]) == 0:
                    print("No logos detected on adversarial image, attack succeeded!")
                    attack_info["success"] = True
                    best_tensor = adv_tensor.detach().clone()
                    attack_info["num_steps_taken"] = step + 1
                    break
                boxes = cur["instances"].pred_boxes.tensor
                loss = self.compute_classifier_loss(adv_tensor, boxes)
                if loss is None:
                    adv_tensor = adv_tensor.detach()
                    continue
                cur_match = self._match(adv_tensor, boxes.detach().cpu().numpy(), url)
                cur_conf = cur_match[3] if cur_match and cur_match[3] is not None else 0.0
                attack_info["confidence_history"].append(float(cur_conf))
                print(f"Current confidence: {cur_conf:.4f}")
                score = float(cur_conf)

            print(f"Step {step + 1} loss: {loss.item():.6f}")

            if score < best_score:
                best_score, best_tensor, no_improve = score, adv_tensor.detach().clone(), 0
            else:
                no_improve += 1

            self.siamese_model.zero_grad(set_to_none=True)
            if self.detector_model is not None:
                self.detector_model.zero_grad(set_to_none=True)
            loss.backward()

            with torch.no_grad():
                grad = adv_tensor.grad
                if grad is None:
                    print("No gradient computed!")
                    break
                if mode == "fgsm":
                    grad_momentum = self.momentum * grad_momentum + grad / (
                        grad.abs().mean() + 1e-12
                    )
                    step_dir = grad_momentum.sign()
                else:  # pgd
                    step_dir = grad.sign()
                # Gradient DESCENT: minimize the loss (suppress detection / matching).
                adv_tensor = adv_tensor - step_size * step_dir
                delta = torch.clamp(adv_tensor - original_tensor, -self.epsilon, self.epsilon)
                adv_tensor = torch.clamp(original_tensor + delta, 0, 1)

            adv_tensor = adv_tensor.detach()
            attack_info["num_steps_taken"] = step + 1

            if patience is not None and no_improve > patience:
                print("No improvement for many steps, stopping early")
                break

        final_tensor = best_tensor if best_tensor is not None else adv_tensor
        final_image = self.tensor_to_rcnn_format(final_tensor)

        with torch.no_grad():
            final_outputs = self.rcnn_model(final_image)
        attack_info["adversarial_detection"] = (
            final_outputs["instances"] if final_outputs else None
        )

        if target_type == "detector":
            n_final = len(final_outputs["instances"]) if final_outputs else 0
            n_init = (
                len(attack_info["original_detection"])
                if attack_info["original_detection"] is not None
                else 0
            )
            attack_info["success"] = n_final < n_init
            print(f"Final number of detections: {n_final} (was {n_init})")
        else:
            if final_outputs and len(final_outputs["instances"]) > 0:
                boxes = final_outputs["instances"].pred_boxes.tensor.detach().cpu().numpy()
                final_match = self._match(final_tensor, boxes, url)
                attack_info["adversarial_matching"] = final_match
                om = attack_info["original_matching"]
                attack_info["success"] = (
                    not final_match
                    or final_match[0] is None
                    or (om is not None and final_match[0] != om[0])
                    or (
                        om is not None
                        and om[3]
                        and final_match[3] is not None
                        and final_match[3] < om[3] * 0.5
                    )
                )
                if final_match and om and om[3] and final_match[3] is not None:
                    print(f"Initial confidence: {om[3]:.4f}")
                    print(f"Final confidence:   {final_match[3]:.4f}")
            else:
                # No logo detected at all -> nothing for the matcher to flag.
                attack_info["success"] = True

        print(f"Attack {'succeeded' if attack_info['success'] else 'failed'}")
        return final_image, attack_info

    # ------------------------------------------------------------------ #
    # public API (signatures unchanged so run_attacks.py keeps working)
    # ------------------------------------------------------------------ #
    def fgsm_attack(
        self,
        image: str,
        url: str,
        target_type: str = "detector",
        num_steps: int = 10,
        step_size: Optional[float] = None,
    ) -> Tuple[np.ndarray, dict]:
        """Iterative, momentum FGSM (MI-FGSM). ``num_steps=1`` gives plain FGSM."""
        try:
            if step_size is None:
                step_size = self.epsilon / max(1, num_steps * 2)
            return self._run_attack(
                image, url, target_type, num_steps, step_size, mode="fgsm"
            )
        except Exception as e:
            print(f"Error in fgsm_attack: {e}")
            print(traceback.format_exc())
            raise

    def pgd_attack(
        self,
        image: str,
        url: str,
        target_type: str = "detector",
        num_steps: int = 100,
        step_size: Optional[float] = None,
        random_start: bool = True,
    ) -> Tuple[np.ndarray, dict]:
        """Projected Gradient Descent with random start and early stopping."""
        try:
            if step_size is None:
                step_size = self.epsilon / 3
            return self._run_attack(
                image,
                url,
                target_type,
                num_steps,
                step_size,
                mode="pgd",
                random_start=random_start,
                patience=20,
            )
        except Exception as e:
            print(f"Error in pgd_attack: {e}")
            print(traceback.format_exc())
            raise
