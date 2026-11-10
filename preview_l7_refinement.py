"""Render original/refined geometry at identical views and diagnostic colors."""
import argparse
from pathlib import Path

from PIL import Image, ImageDraw
from plyfile import PlyData
import torch
import torchvision

from arguments import ModelParams, PipelineParams
from gaussian_renderer import render
from scene import Scene, GaussianModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    model = ModelParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument('--original', type=Path, required=True)
    parser.add_argument('--refined', type=Path, required=True)
    args = parser.parse_args()
    args.model_path = str(args.refined.parent.parent.parent)
    dataset, pipe = model.extract(args), pipeline.extract(args)
    output = Path(dataset.model_path) / 'review'
    output.mkdir(exist_ok=True)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=35000, shuffle=False)
    figure = Image.new('RGB', (1320, 1280), '#17191c')
    draw = ImageDraw.Draw(figure)
    draw.text((20, 15), 'Same camera / fixed L7 color / SH disabled', fill='white')
    with torch.no_grad():
        for column, (label, path) in enumerate((('Original v4', args.original), ('Refined v4', args.refined))):
            gaussians.load_ply(str(path))
            vertex = PlyData.read(str(path))['vertex'].data
            colors = (gaussians.get_features[:, 0, :] * 0.28209479177387814 + 0.5).clamp(0, 1)
            mask = torch.tensor(vertex['layer'] == 7, device='cuda')
            colors[mask] = torch.tensor([0.84, 0.70, 0.47], device='cuda')
            for row, index in enumerate((0, 7)):
                view = scene.getTestCameras()[index]
                view.image_height = view.image_width = 1024
                image = render(view, gaussians, pipe, torch.tensor([0.08, 0.08, 0.08], device='cuda'),
                               override_color=colors, antialiasing=dataset.antialiasing)['render'].clamp(0, 1)
                torchvision.utils.save_image(image, output / f'{label.split()[0].lower()}_{index:02d}.png')
                pixels = image.permute(1, 2, 0).cpu().numpy()
                crop = Image.fromarray((pixels[210:800, 180:830] * 255).round().astype('uint8'))
                left, top = column * 660 + 5, row * 620 + 60
                draw.text((left + 10, top - 20), f'{label} / view {index}', fill='white')
                figure.paste(crop, (left, top))
    figure.save(output / 'geometry_comparison.png')


if __name__ == '__main__':
    main()
