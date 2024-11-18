from typing import Tuple, Callable, List
import torch
from torch.nn import functional
import torchvision.transforms as transforms
import numpy as np
from experiments.image_processing.get_matrix import make_filter2d, make_derivatives2d
from PIL import Image
from os import listdir
from random import shuffle
from scipy.sparse import linalg


def check_and_extract_number_of_datapoints(number_of_datapoints_per_dataset: dict) -> Tuple[int, int, int, int]:
    if (('prior' not in number_of_datapoints_per_dataset)
            or ('train' not in number_of_datapoints_per_dataset)
            or ('test' not in number_of_datapoints_per_dataset)
            or ('validation' not in number_of_datapoints_per_dataset)):
        raise ValueError("Missing number of datapoints.")
    else:
        return (number_of_datapoints_per_dataset['prior'],
                number_of_datapoints_per_dataset['train'],
                number_of_datapoints_per_dataset['test'],
                number_of_datapoints_per_dataset['validation'])


def pil_to_tensor(img: Image, device: str) -> torch.Tensor:
    return transforms.ToTensor()(img).to(device)


def get_blurring_kernel() -> torch.Tensor:
    kernel = torch.tensor([[1, 4, 7, 4, 1],
                           [4, 16, 26, 16, 4],
                           [7, 26, 41, 26, 7],
                           [4, 16, 26, 16, 4],
                           [1, 4, 7, 4, 1]]) / 273
    kernel = kernel.reshape((1, 1, 5, 5))
    return kernel


def get_finite_difference_kernels() -> Tuple[torch.Tensor, torch.Tensor]:
    diff_kernel_height = torch.tensor([[0, 0, 0],
                                       [0, -1., 0],
                                       [0, 1, 0]]).reshape((1, 1, 3, 3))
    diff_kernel_width = torch.tensor([[0, 0, 0],
                                      [0, -1., 1],
                                      [0, 0, 0]]).reshape((1, 1, 3, 3))
    return diff_kernel_width, diff_kernel_height


def get_shape_of_images() -> Tuple[int, int, int, int]:
    img_height = 250
    img_width = int(0.75 * img_height)
    return 1, 1, img_height, img_width  # Note that this automatically returns a tuple


def get_image_height_and_width() -> Tuple[int, int]:
    shape = get_shape_of_images()
    return shape[2], shape[3]


def get_epsilon() -> float:
    return 0.01


def get_distribution_of_regularization_parameter() -> torch.distributions.uniform.Uniform:
    return torch.distributions.uniform.Uniform(low=5e-2, high=5e-1)


def get_largest_possible_regularization_parameter() -> float:
    dist = get_distribution_of_regularization_parameter()
    return dist.high.item()


def get_loss_function_of_algorithm(blurring_kernel: torch.Tensor) -> Tuple[Callable, Callable]:

    shape_of_image = get_shape_of_images()
    diff_kernel_width, diff_kernel_height = get_finite_difference_kernels()

    def blur_tensor(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        return functional.conv2d(input=torch.nn.ReflectionPad2d(kernel.shape[-1] // 2)(x), weight=kernel)

    def img_derivatives(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return blur_tensor(x=x, kernel=diff_kernel_height), blur_tensor(x=x, kernel=diff_kernel_width)

    def quadratic(x: torch.Tensor, blurred_img: torch.Tensor) -> torch.Tensor:
        cur_blurred_img = blur_tensor(x=x.reshape(shape_of_image), kernel=blurring_kernel)
        return 0.5 * torch.linalg.norm(cur_blurred_img - blurred_img) ** 2

    def regularizer(x: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        epsilon = get_epsilon()
        d_h, d_w = img_derivatives(x=x.reshape(shape_of_image))
        return mu * torch.sum(torch.sqrt(d_h ** 2 + d_w ** 2 + epsilon ** 2))

    # Define loss function as:
    # (1/2) * || Kx - b ||^2 + mu * sum_ij sqrt((D_hx)_ij ** 2 + (D_wx)_ij ** 2 + eps ** 2)
    def loss_function(x, parameter):
        return quadratic(x, blurred_img=parameter['img']) + regularizer(x, mu=parameter['mu'])

    return loss_function, blur_tensor


def get_smoothness_parameter() -> float:

    img_height, img_width = get_image_height_and_width()
    blurring_kernel = get_blurring_kernel().reshape((5, 5))
    epsilon = get_epsilon()
    largest_possible_regularization_parameter = get_largest_possible_regularization_parameter()

    # Get Hessian of problem (0.5 * ||Kx-b||**2 + 0.5 * mu * ||Dx||**2)
    # This is given by K.T@K + mu * D.T@D
    K = make_filter2d(width=img_width, height=img_height, filter_to_apply=blurring_kernel)
    D = make_derivatives2d(width=img_width, height=img_height)
    H = K.T @ K + (largest_possible_regularization_parameter / epsilon) * D.T @ D
    w, _ = linalg.eigsh(H)
    smoothness_parameter = torch.max(torch.tensor(w))
    return smoothness_parameter.item()


def load_and_transform_image(path: str, device: str) -> torch.Tensor:
    img_height, img_width = get_image_height_and_width()
    to_grayscale = transforms.Grayscale()
    resizing = transforms.Resize((img_height, img_width))
    return pil_to_tensor(resizing(to_grayscale(Image.open(path))), device)


def load_images(path_to_images: str, device: str) -> List[torch.Tensor]:

    # (!) Naming convention: images start with 'img'
    all_images_in_folder = [x for x in listdir(path_to_images) if x[0:3] == 'img']
    imgs = [load_and_transform_image(path=path_to_images + img_name, device=device)
            for img_name in all_images_in_folder]
    return imgs


def split_images_into_separate_sets(imgs: List[torch.Tensor]):
    if len(imgs) <= 4:
        raise Exception("Too few images provided for splitting.")

    shuffle(imgs)   # Note that this is inplace
    indices = np.linspace(0, len(imgs), 5).astype(int)
    return (imgs[indices[0]:indices[1]],
            imgs[indices[1]:indices[2]],
            imgs[indices[2]:indices[3]],
            imgs[indices[3]:indices[4]])


def get_noise_distribution() -> torch.distributions.Distribution:
    return torch.distributions.normal.Normal(loc=0, scale=25/256)


def clip_to_interval_zero_one(image: torch.Tensor) -> torch.Tensor:
    return torch.maximum(torch.tensor(0.0), torch.minimum(torch.tensor(1.0), image))


def add_noise_and_blurr(images: List[torch.Tensor],
                        blurring_function: Callable) -> List[torch.Tensor]:
    blurring_kernel = get_blurring_kernel()
    noise_distribution = get_noise_distribution()
    shape_of_image = get_shape_of_images()
    return [clip_to_interval_zero_one(blurring_function(x.reshape(shape_of_image), kernel=blurring_kernel)
                                      + noise_distribution.sample(shape_of_image))
            for x in images]


def get_blurred_images(images: List[torch.Tensor],
                       blurring_function: Callable
                       ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:

    images_prior, images_train, images_test, images_validation = split_images_into_separate_sets(images)

    blurred_images_prior = add_noise_and_blurr(images_prior, blurring_function=blurring_function)
    blurred_images_train = add_noise_and_blurr(images_train, blurring_function=blurring_function)
    blurred_images_test = add_noise_and_blurr(images_test, blurring_function=blurring_function)
    blurred_images_validation = add_noise_and_blurr(images_validation, blurring_function=blurring_function)
    return blurred_images_prior, blurred_images_train, blurred_images_test, blurred_images_validation


def get_parameters(images: List[torch.Tensor],
                   number_of_datapoints_per_dataset: dict,
                   blurring_function: Callable) -> dict:

    n_prior, n_train, n_test, n_validation = check_and_extract_number_of_datapoints(number_of_datapoints_per_dataset)
    distribution_regularization_parameters = get_distribution_of_regularization_parameter()
    blurred_images_prior, blurred_images_train, blurred_images_test, blurred_images_validation = get_blurred_images(
        images, blurring_function=blurring_function)

    parameters = {
        'prior': [{'img': blurred_images_prior[np.random.choice(np.arange(len(blurred_images_prior)))],
                   'mu': distribution_regularization_parameters.sample((1,))} for _ in range(n_prior)],
        'train': [{'img': blurred_images_train[np.random.choice(np.arange(len(blurred_images_train)))],
                   'mu': distribution_regularization_parameters.sample((1,))} for _ in range(n_train)],
        'test': [{'img': blurred_images_test[np.random.choice(np.arange(len(blurred_images_test)))],
                  'mu': distribution_regularization_parameters.sample((1,))} for _ in range(n_test)],
        'validation': [{'img': blurred_images_validation[np.random.choice(np.arange(len(blurred_images_validation)))],
                        'mu': distribution_regularization_parameters.sample((1,))} for _ in range(n_validation)]
    }
    return parameters


def get_data(path_to_images: str,
             number_of_datapoints_per_dataset: dict,
             device: str) -> Tuple[dict, Callable, float]:

    blurring_kernel = get_blurring_kernel()
    smoothness_parameter = get_smoothness_parameter()
    loss_function_of_algorithm, blur_tensor = get_loss_function_of_algorithm(blurring_kernel)
    images = load_images(path_to_images, device=device)
    parameters = get_parameters(
        images=images, number_of_datapoints_per_dataset=number_of_datapoints_per_dataset, blurring_function=blur_tensor
    )

    return parameters, loss_function_of_algorithm, smoothness_parameter
