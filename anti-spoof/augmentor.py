import Augmentor


def RandomAugment(folder, IP=False, Graph=False, Erase=False):
    if IP==False and Graph==False and Erase==False:
        return None
    p = Augmentor.Pipeline(folder)

    if IP:
        p.random_color(0.5, min_factor=0.4, max_factor=1.6)
        p.random_brightness(0.5, min_factor=0.4, max_factor=1.6)
        p.random_contrast(0.5, min_factor=0.4, max_factor=1.2)

    if Graph:
        #p.rotate(probability=0.7, max_left_rotation=20, max_right_rotation=20)
        #p.zoom(probability=0.5, min_factor=0.8, max_factor=1.2)
        p.random_distortion(probability=0.5, grid_width=4, grid_height=4, magnitude=4)
        #p.skew_left_right(probability=0.5, magnitude=0.15)

    if Erase:
        p.random_erasing(1.0,rectangle_area=0.5)
    return p


if __name__=='__main__':
    folder = '29xxx'
    p = RandomAugment(folder, Graph=True)
    p.sample(1000)
