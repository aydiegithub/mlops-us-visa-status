"""
Entry point to run the complete ML pipeline for US visa approval prediction.
Tracks total runtime and suppresses warnings for clean output.
"""

from usa_visa.pipeline.training_pipeline import TrainPipeline
import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    import time
    from usa_visa.pipeline.training_pipeline import TrainPipeline

    start_time = time.time()
    print("🚀 Running US Visa Approval ML Pipeline...\n")

    pipeline = TrainPipeline()
    pipeline.run_pipeline()

    end_time = time.time()
    elapsed_time = end_time - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    print(f"\n✅ Pipeline completed in {int(minutes)} min {int(seconds)} sec.")
