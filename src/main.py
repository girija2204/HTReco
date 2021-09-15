import os
import argparse
import sys
from configparser import ConfigParser

from src.dataloader.datastore import Datastore
from src.htr_service import HTRService
from src import constants

config = ConfigParser()
config.read(os.path.join(constants.ROOT_DIR, 'config.ini'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--transform", action="store_true", default=False)
    parser.add_argument("--source", type=str, required=False)
    parser.add_argument("--architecture", type=str, required=False, default="flor")
    parser.add_argument("--train", action="store_true", default=False)
    parser.add_argument("--predict_and_evaluate", action="store_true", default=False)
    parser.add_argument("--predict", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--ground_truth_path", type=str, required=False)
    parser.add_argument("--images_path", type=str, required=False)
    parser.add_argument("--partition_criteria_path", type=str, required=False)
    parser.add_argument("--dataset_path", type=str, required=False)
    parser.add_argument("--image_format", type=str, required=False)
    parser.add_argument("--target", type=str, required=False)
    parser.add_argument("--segmentation_level", type=str, required=False)

    args = parser.parse_args()

    if args.transform:
        images_path = args.images_path if args.images_path is not None else config['images_path']
        ground_truth_path = args.ground_truth_path if args.ground_truth_path is not None else config['ground_truth_path']
        partition_criteria_path = args.partition_criteria_path if args.partition_criteria_path is not None else config['partition_criteria_path']
        if images_path is None or ground_truth_path is None or partition_criteria_path is None:
            print(f"error")
            sys.exit("Please provide images_path/ground_truth_path/partition_criteria_path")
        dataset_path = args.dataset_path
        print(f"{args.source} dataset will be transformed...")
        datastore = Datastore(datasource=args.source, ground_truth_path=ground_truth_path,
                              images_path=images_path, partition_criteria_path=partition_criteria_path,
                              dataset_path=dataset_path, format=constants.STORAGE_FORMAT, img_format=args.image_format)
        datastore.read_raw_dataset()
    elif args.train:
        print(f"Data Source: {args.source}\nTask: Train\nStatus: started...")
        htrService = HTRService(architecture=args.architecture, datasource=args.source,
                                segmentation_level=args.segmentation_level)
        model_history = htrService.train()
        print(f"Task: Train - Status: ended - Please find the training results at {htrService.train_output_file}.")
    elif args.predict_and_evaluate:
        print(f"Data Source: {args.source} - Task: Prediction & Evaluation - Status: started...")
        htrService = HTRService(architecture=args.architecture, datasource=args.source,
                                segmentation_level=args.segmentation_level)
        htrService.predict_and_evaluate()
        print(f"Task: Prediction & Evaluation - Status: ended - Please find the prediction results at "
              f"{htrService.prediction_output_file} and evaluation results at {htrService.evaluation_output_file}.")
    elif args.predict:
        data_source = args.target if args.target is not None else constants.TARGET_FOLDER_PATH
        results_folder = constants.RESULTS_FOLDER_PATH
        print(f"Data Source: {data_source} - Task: Prediction - Status: started...")
        htrService = HTRService(architecture=args.architecture, datasource=args.source,
                                segmentation_level=args.segmentation_level)
        output_file = htrService.predict(data_source, results_folder)
        print(f"Task: Prediction - Status: ended - Please find the prediction results at {output_file}.")
    elif args.benchmark:
        print(f"Running the benchmark service...")
        batch_sizes = [16, 32, 64, 128, 256, 512]
        num_steps = [100, 200, -1]
        architecture = "flor"
        source = "iam"
        for batch_size in batch_sizes:
            for num_step in num_steps:
                htrService = HTRService(architecture=architecture, datasource=source,
                                        segmentation_level=args.segmentation_level)
                benchmark = htrService.benchmark(batch_size, num_step)
                print(f"Benchmark: {benchmark}")
