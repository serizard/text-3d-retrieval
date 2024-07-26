from utils.refine import TextRefiner
import argparse
from configparser import ConfigParser
from utils.misc import make_default_config, load_config, dump_config
 

def process_input(user_input):
    '''
    user_input 받아서 사전에 정의된 clip 모델 바탕으로 text feature로 변환하기
    
    by 도형
    '''

def retrieve_3d(text_feature, k=5):
    '''
    text feature 받아서 3D object retrieval

    by 세민
    '''



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process user input')
    parser.add_argument('--access_token', type=str, help='Hugging Face access token', required=True)
    parser.add_argument('--init_config', type=bool, help='Initialize default configuration')
    args = parser.parse_args()

    if args.init_config:
        make_default_config()
    
    config = load_config()

    refiner = TextRefiner(access_token=args.access_token)

    user_input = input("Enter a user description of shape to retrieve: ")
    structured_text = refiner.refine(user_input)

    text_feature = process_input(structured_text)