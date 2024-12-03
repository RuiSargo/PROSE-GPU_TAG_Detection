import os
import cairosvg

# Caminho para a pasta de entrada e saída
base_folder = './ArucoMarkers'  # Substitua pelo caminho real do seu dataset
output_folder = './converted_images'

# Criar a pasta de saída se não existir
os.makedirs(output_folder, exist_ok=True)

# Percorrer todas as subpastas
for subfolder in os.listdir(base_folder):
    subfolder_path = os.path.join(base_folder, subfolder)
    if os.path.isdir(subfolder_path):  # Verifica se é uma subpasta
        # Criar uma pasta correspondente na saída
        subfolder_output = os.path.join(output_folder, subfolder)
        os.makedirs(subfolder_output, exist_ok=True)

        # Processar todos os arquivos .svg na subpasta
        for svg_file in os.listdir(subfolder_path):
            if svg_file.endswith('.svg'):
                svg_path = os.path.join(subfolder_path, svg_file)
                png_path = os.path.join(subfolder_output, svg_file.replace('.svg', '.png'))
                
                # Converter SVG para PNG
                cairosvg.svg2png(url=svg_path, write_to=png_path)
                print(f"Convertido: {svg_path} -> {png_path}")

print("Conversão de SVG para PNG concluída!")
