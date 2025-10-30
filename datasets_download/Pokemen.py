import requests
import os
import json
from PIL import Image
from io import BytesIO
import torch
from torchvision import transforms
from tqdm import tqdm

class PokemonDatasetCreator:
    def __init__(self, output_dir="./datasets/pokemon_diffusion_data"):
        self.output_dir = output_dir
        self.images_dir = os.path.join(output_dir, "images")
        self.metadata_file = os.path.join(output_dir, "metadata.jsonl")
        os.makedirs(self.images_dir, exist_ok=True)
        
        # Diffusion模型需要的预处理
        self.preprocess = transforms.Compose([
            transforms.Resize(512),
            transforms.CenterCrop(512),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
    
    def get_pokemon_data(self, start_id=1, end_id=1010):  # 包含所有世代的宝可梦
        """从PokeAPI获取宝可梦数据"""
        base_url = "https://pokeapi.co/api/v2/pokemon/"
        metadata = []
        
        for pokemon_id in tqdm(range(start_id, end_id + 1)):
            try:
                # 获取基本信息
                response = requests.get(f"{base_url}{pokemon_id}", timeout=10)
                '''
                (Pdb) data.keys()
                dict_keys(['abilities', 'base_experience', 'cries', 'forms',    'game_indices', 
                            'height',   'held_items',       'id',   'is_default', 
                            'location_area_encounters',     'moves', 'name',    'order', 'past_abilities', 
                            'past_types', 'species',        'sprites', 'stats', 'types', 'weight'])
                '''
                data = response.json()
                
                # 获取物种信息（用于描述）
                species_response = requests.get(data['species']['url'], timeout=10) # data['species'].keys = name url
                '''
                (Pdb) species_data.keys()
                dict_keys(['base_happiness',    'capture_rate',     'color',    'egg_groups', 
                            'evolution_chain',  'evolves_from_species',     'flavor_text_entries', 'form_descriptions', 
                            'forms_switchable', 'gender_rate', 'genera',    'generation', 'growth_rate', 'habitat', 
                            'has_gender_differences', 'hatch_counter', 'id', 'is_baby', 'is_legendary', 'is_mythical', 
                            'name',             'names',  'order', 'pal_park_encounters', 'pokedex_numbers', 'shape', 'varieties'])
                '''
                species_data = species_response.json()
                
                # 获取英文描述
                '''
                (Pdb) species_data['flavor_text_entries'][0]
                {'flavor_text': 'A strange seed was\nplanted on its\nback at birth.\x0cThe plant sprouts\nand grows with\nthis POKéMON.', 
                'language': {   'name': 'en',   'url': 'https://pokeapi.co/api/v2/language/9/'}, 
                'version':  {   'name': 'red',  'url': 'https://pokeapi.co/api/v2/version/1/'}}
                '''
                flavor_texts = [entry['flavor_text'] for entry in species_data['flavor_text_entries'] 
                               if entry['language']['name'] == 'en']
                description = flavor_texts[0].replace('\n', ' ').replace('\f', ' ') if flavor_texts else ""
                
                # 获取高质量图片（官方艺术图）
                image_url = data['sprites']['other']['official-artwork']['front_default']
                if not image_url:  # 备用方案
                    image_url = data['sprites']['other']['home']['front_default']
                
                if image_url:
                    # 下载并处理图片
                    img_response = requests.get(image_url)
                    img = Image.open(BytesIO(img_response.content)).convert("RGB")
                    
                    # 应用预处理
                    processed_img = self.preprocess(img)
                    
                    # 保存图片和元数据
                    pokemon_name = data['name']
                    filename = f"{pokemon_id:04d}_{pokemon_name}.pt"
                    torch.save(processed_img, os.path.join(self.images_dir, filename))
                    
                    # 创建元数据
                    types = [t['type']['name'] for t in data['types']]
                    metadata.append({
                        "file_name": filename,
                        "text": f"A pokemon called {pokemon_name}, {description}",
                        "types": types,
                        "height": data['height'],
                        "weight": data['weight']
                    })
                    
            except Exception as e:
                print(f"Error processing Pokemon #{pokemon_id}: {e}")
                continue
        
        # 保存元数据
        with open(self.metadata_file, 'w') as f:
            for item in metadata:
                f.write(json.dumps(item) + '\n')
        
        print(f"下载完成！共处理 {len(metadata)} 只宝可梦")
        return metadata

# 使用示例
if __name__ == "__main__":
    creator = PokemonDatasetCreator()
    metadata = creator.get_pokemon_data(1, 151)  # 只下载第一代