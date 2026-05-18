import os
import sys
import time
import json
import argparse
from scanner import DepthEngine

def main():
    parser = argparse.ArgumentParser(description="Batch process image sets into 3D meshes")
    parser.add_argument("--step", type=int, default=4, help="Subsampling density")
    parser.add_argument("--scale", type=float, default=0.55, help="Depth scale")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    sets_dir = os.path.join(data_dir, "test_sets")
    model_path = os.path.join(base_dir, "midas_small.tflite")

    if not os.path.exists(sets_dir):
        print(f"Error: Test sets directory not found at {sets_dir}")
        return

    engine = DepthEngine(model_path)
    test_sets = sorted([d for d in os.listdir(sets_dir) if os.path.isdir(os.path.join(sets_dir, d))])
    
    results = []
    print(f"{'Set Name':<20} | {'Status':<10} | {'Time (s)':<10}")
    print("-" * 45)

    for set_name in test_sets:
        set_path = os.path.join(sets_dir, set_name)
        images = sorted([os.path.join(set_path, f) for f in os.listdir(set_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        
        if not images:
            continue
            
        output_mesh = os.path.join(data_dir, f"mesh_{set_name}.glb")
        
        start_time = time.time()
        try:
            engine.generate_fused_mesh(images[:4], output_mesh, step=args.step, scale=args.scale)
            elapsed = round(time.time() - start_time, 2)
            print(f"{set_name:<20} | Success    | {elapsed:<10}")
            results.append({"set": set_name, "status": "success", "time": elapsed})
        except Exception as e:
            print(f"{set_name:<20} | Error      | {str(e)[:20]}")
            results.append({"set": set_name, "status": "error", "error": str(e)})

    # Save summary
    with open(os.path.join(data_dir, "batch_results.json"), "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
