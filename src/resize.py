import os
from PIL import Image

def resize_image(image, size):
    return image.resize(size, Image.ANTIALIAS)

def resize_images(image_dir, output_dir, size):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    images=os.listdir(image_dir)
    num_images=len(images)

    for i, image in enumerate(images):
        with open(os.path.join(image_dir, image),'r+b') as f:
            with Image.open(f) as img:
                img = resize_image(img, size)
                img.save(os.path.join(output_dir,image), img.format)

        if (i+1) % 100 == 0:
            print("[{}/{}] Resized the images and saved into '{}'."
                  .format(i + 1, num_images, output_dir))



def main(args):
    image_dir = args['image_dir']
    output_dir = os.path.join(args['data_dir'],args['output_dir'])
    image_size = [args['image_size'], args['image_size']]
    if not os.path.exists(args['data_dir']):
        os.mkdir(args['data_dir'])
        os.mkdir(output_dir)
    resize_images(image_dir, output_dir, image_size)

args = {
    'image_dir' : '../raw_data_dir/val2014',
    'data_dir':'../data_dir',
    'output_dir' : 'resized_val2014',
    'image_size' : 256
}
main(args)
