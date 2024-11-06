import unittest
from typing import Callable
import torch
from torch.xpu import device

from main import TESTING_LEVEL
from experiments.image_processing.data_generation import (get_blurring_kernel,
                                                          get_finite_difference_kernels,
                                                          get_shape_of_images,
                                                          get_loss_function_of_algorithm,
                                                          get_epsilon,
                                                          get_distribution_of_regularization_parameter,
                                                          get_smoothness_parameter,
                                                          get_largest_possible_regularization_parameter,
                                                          get_image_height_and_width,
                                                          load_and_transform_image,
                                                          load_images,
                                                          split_images_into_separate_sets,
                                                          get_noise_distribution,
                                                          clip_to_interval_zero_one,
                                                          add_noise_and_blurr,
                                                          check_and_extract_number_of_datapoints,
                                                          get_parameters,
                                                          get_blurred_images,
                                                          get_data
                                                          )


class TestDataGeneration(unittest.TestCase):

    def setUp(self):
        self.image_path = '/home/michael/Desktop/Experiments/Images/'
        self.image_name_test = 'img_001.jpg'

    def test_check_and_extract_number_of_datapoints(self):
        with self.assertRaises(ValueError):
            check_and_extract_number_of_datapoints({})
        with self.assertRaises(ValueError):
            check_and_extract_number_of_datapoints({'prior': 1})
        with self.assertRaises(ValueError):
            check_and_extract_number_of_datapoints({'prior': 1, 'train': 1})
        with self.assertRaises(ValueError):
            check_and_extract_number_of_datapoints({'prior': 1, 'train': 1, 'test': 1})
        number_data = {'prior': torch.randint(low=1, high=100, size=(1,)).item(),
                       'train': torch.randint(low=1, high=100, size=(1,)).item(),
                       'test': torch.randint(low=1, high=100, size=(1,)).item(),
                       'validation': torch.randint(low=1, high=100, size=(1,)).item()}
        n_prior, n_train, n_test, n_val = check_and_extract_number_of_datapoints(number_data)
        self.assertEqual(n_prior, number_data['prior'])
        self.assertEqual(n_train, number_data['train'])
        self.assertEqual(n_test, number_data['test'])
        self.assertEqual(n_val, number_data['validation'])

    def test_get_blurring_kernel(self):
        kernel = get_blurring_kernel()
        self.assertIsInstance(kernel, torch.Tensor)
        self.assertEqual(kernel.shape, torch.Size((1, 1, 5, 5)))

    def test_get_finite_difference_kernels(self):
        k_w, k_h = get_finite_difference_kernels()
        self.assertIsInstance(k_w, torch.Tensor)
        self.assertEqual(k_w.shape, torch.Size((1, 1, 3, 3)))
        self.assertIsInstance(k_h, torch.Tensor)
        self.assertEqual(k_h.shape, torch.Size((1, 1, 3, 3)))

    def test_get_shape_of_image(self):
        shape = get_shape_of_images()
        self.assertIsInstance(shape, tuple)
        self.assertEqual(len(shape), 4)
        self.assertEqual(shape[0], 1)
        self.assertEqual(shape[1], 1)

    def test_get_loss_function_of_algorithm(self):
        blurring_kernel = get_blurring_kernel()
        loss_function, blur_tensor = get_loss_function_of_algorithm(blurring_kernel)
        self.assertIsInstance(loss_function, Callable)
        self.assertIsInstance(blur_tensor, Callable)

    def test_get_epsilon(self):
        eps = get_epsilon()
        self.assertIsInstance(eps, float)
        self.assertTrue(eps > 0)

    def test_get_distribution_of_regularization_parameters(self):
        dist = get_distribution_of_regularization_parameter()
        self.assertIsInstance(dist, torch.distributions.uniform.Uniform)

    def test_get_largest_possible_regularization_parameter(self):
        p = get_largest_possible_regularization_parameter()
        self.assertIsInstance(p, float)
        self.assertEqual(p, get_distribution_of_regularization_parameter().high.item())

    # @unittest.skipIf(condition=(TESTING_LEVEL != 'FULL_TEST_WITH_EXPERIMENTS'),
    #                  reason='Too expensive to test all the time.')
    def test_get_smoothness_parameter(self):
        p = get_smoothness_parameter()
        self.assertIsInstance(p, float)
        self.assertTrue(p > 0)

    def test_get_image_height_and_width(self):
        shape = get_shape_of_images()
        height, width = get_image_height_and_width()
        self.assertEqual(height, shape[2])
        self.assertEqual(width, shape[3])

    def test_load_and_transform_image(self):
        img = load_and_transform_image(path=self.image_path + self.image_name_test, device='cpu')
        height, width = get_image_height_and_width()
        self.assertIsInstance(img, torch.Tensor)
        self.assertEqual(img.shape, torch.Size([1, height, width]))

    def test_load_images(self):
        all_images = load_images(path_to_images=self.image_path, device='cpu')
        height, width = get_image_height_and_width()
        self.assertIsInstance(all_images, list)
        for img in all_images:
            self.assertIsInstance(img, torch.Tensor)
            self.assertEqual(img.shape, torch.Size([1, height, width]))

    def test_split_images_into_separate_sets(self):
        all_images = load_images(path_to_images=self.image_path, device='cpu')
        all_images = all_images[0:8]
        imgs_prior, imgs_train, imgs_test, imgs_val = split_images_into_separate_sets(all_images)
        self.assertEqual(len(all_images), len(imgs_prior) + len(imgs_train) + len(imgs_test) + len(imgs_val))

    def test_get_noise_distribution(self):
        dist = get_noise_distribution()
        self.assertIsInstance(dist, torch.distributions.normal.Normal)

    def test_clip_to_interval(self):
        image = torch.randn((10, 20))
        clipped_image = clip_to_interval_zero_one(image)
        self.assertEqual(image.shape, clipped_image.shape)
        self.assertTrue(torch.min(clipped_image) >= 0)
        self.assertTrue(torch.max(clipped_image) <= 1)

    def test_add_noise_and_blurr(self):
        all_images = load_images(path_to_images=self.image_path, device='cpu')
        num_imgs = torch.randint(low=1, high=10, size=(1,)).item()
        if num_imgs > len(all_images):
            num_imgs = len(all_images)
        loss_function, blur_tensor = get_loss_function_of_algorithm(get_blurring_kernel())
        blurred = add_noise_and_blurr(all_images[:num_imgs], blurring_function=blur_tensor)
        self.assertEqual(len(blurred), num_imgs)
        shape = get_shape_of_images()
        for img in blurred:
            self.assertEqual(img.shape, shape)

    def test_get_blurred_images(self):
        all_images = load_images(path_to_images=self.image_path, device='cpu')
        all_images = all_images[0:8]
        loss_function, blur_tensor = get_loss_function_of_algorithm(get_blurring_kernel())
        b_i_p, b_i_tr, b_i_te, b_i_v = get_blurred_images(images=all_images, blurring_function=blur_tensor)
        self.assertIsInstance(b_i_p, list)
        self.assertIsInstance(b_i_tr, list)
        self.assertIsInstance(b_i_te, list)
        self.assertIsInstance(b_i_v, list)

    def test_get_parameter(self):
        all_images = load_images(path_to_images=self.image_path, device='cpu')
        all_images = all_images[0:8]
        loss_function, blur_tensor = get_loss_function_of_algorithm(get_blurring_kernel())
        number_data = {'prior': torch.randint(low=1, high=100, size=(1,)).item(),
                       'train': torch.randint(low=1, high=100, size=(1,)).item(),
                       'test': torch.randint(low=1, high=100, size=(1,)).item(),
                       'validation': torch.randint(low=1, high=100, size=(1,)).item()}
        parameter = get_parameters(
            images=all_images, number_of_datapoints_per_dataset=number_data, blurring_function=blur_tensor
        )

        self.assertIsInstance(parameter, dict)
        self.assertEqual(list(parameter.keys()), list(number_data.keys()))
        for name in parameter.keys():
            self.assertIsInstance(parameter[name], list)
            self.assertEqual(len(parameter[name]), number_data[name])

    def test_get_data(self):
        number_data = {'prior': torch.randint(low=1, high=100, size=(1,)).item(),
                       'train': torch.randint(low=1, high=100, size=(1,)).item(),
                       'test': torch.randint(low=1, high=100, size=(1,)).item(),
                       'validation': torch.randint(low=1, high=100, size=(1,)).item()}
        parameters, loss_function_of_algorithm, smoothness_parameter = get_data(
            path_to_images=self.image_path, number_of_datapoints_per_dataset=number_data, device='cpu')
        self.assertIsInstance(parameters, dict)
        self.assertIsInstance(loss_function_of_algorithm, Callable)
        self.assertIsInstance(smoothness_parameter, float)
