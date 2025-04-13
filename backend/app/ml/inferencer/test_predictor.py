import os
import matplotlib.pyplot as plt
from app.ml.inferencer.predictor import ChangeDetectionPredictor
from app.ml.inferencer.evaluation import calculate_metrics, plot_metrics

def test_model(model_path, before_image_path, after_image_path, ground_truth_path=None, output_dir='results'):
    """Test the model on a single image pair"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize predictor
    predictor = ChangeDetectionPredictor(model_path=model_path)
    
    # Process images
    result = predictor.process_image_pair(
        before_image_path, 
        after_image_path, 
        output_dir
    )
    
    print(f"Change percentage: {result['analysis']['change_percentage']:.2f}%")
    print(f"Number of change regions: {result['analysis']['num_change_regions']}")
    
    # Evaluate with ground truth if available
    if ground_truth_path and os.path.exists(ground_truth_path):
        import rasterio
        import numpy as np
        
        # Load ground truth mask
        with rasterio.open(ground_truth_path) as src:
            gt_mask = src.read(1)
            gt_mask = (gt_mask > 0).astype(np.uint8)
        
        # Get prediction
        pred_mask = result['prediction']['binary_mask']
        
        # Resize gt_mask if needed
        if gt_mask.shape != pred_mask.shape:
            import cv2
            gt_mask = cv2.resize(gt_mask, (pred_mask.shape[1], pred_mask.shape[0]), 
                               interpolation=cv2.INTER_NEAREST)
        
        # Calculate metrics
        metrics = calculate_metrics(pred_mask, gt_mask)
        
        # Print metrics
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Kappa coefficient: {metrics['kappa']:.4f}")
        print(f"FI error: {metrics['fi_error']:.4f}")
        
        # Plot metrics
        plot_metrics(metrics, os.path.join(output_dir, 'evaluation_metrics.png'))
    
    return result

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test change detection model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model file")
    parser.add_argument("--before", type=str, required=True, help="Path to before image")
    parser.add_argument("--after", type=str, required=True, help="Path to after image")
    parser.add_argument("--ground_truth", type=str, help="Path to ground truth mask (optional)")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory")
    
    args = parser.parse_args()
    
    test_model(
        args.model_path,
        args.before,
        args.after,
        args.ground_truth,
        args.output_dir
    )