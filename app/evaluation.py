import argparse
import yaml
import subprocess
import os
import ntpath
import sys

def main(args):
      
    if sys.version_info[0] != 2:
        raise Exception("Must be using Python 2")

    ## DO NOT CHANGE! this code
    # configuration YAML formatted file, default configuration is ./config/
    with open(args.config, 'r') as ymlfile:
        cfg = yaml.load(ymlfile) # please use cfg['config_name'] to use
    
    sys.path.insert(0, cfg['vist_eval_dir'])
    from evaluate import Evaluate
    
    stdoutOrigin=sys.stdout 
    sys.stdout = open(cfg['evaluation_result']['dir']+'/output.log', "w")
      
    
    evaluation_id = ntpath.basename(args.config)
    evaluation_id = os.path.splitext(evaluation_id)[0]
    result_path = os.path.join(cfg['evaluation_result']['dir'], cfg['test_result']['experiment_id'], cfg['test_result']['model_id'], evaluation_id)
    
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    
    story_result = os.path.join(cfg['test_result']['dir'], cfg['test_result']['inference_id'], cfg['test_result']['experiment_id'], cfg['test_result']['model_id'], cfg['test_result']['file'])
    subprocess.call(["java", "-jar", cfg['evaluation_dir_class']+ "runnable_jar/EvalVIST.jar", "-testFile", story_result, "-gsFile", cfg['sis_test']])


    score_evaluation = Evaluate(prediction_file = story_result, \
        image_dir = cfg['dataset']['directory']['image'], \
            annotation_dir = cfg['dataset']['directory']['annotation'], \
            vist_api_dir = cfg['vist_api_dir'])
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='specify the file configuration, YAML formatted')

    args = parser.parse_args()
    main(args)
