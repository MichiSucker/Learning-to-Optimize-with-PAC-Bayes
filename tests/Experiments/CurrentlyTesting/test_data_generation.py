import unittest
from typing import Callable
import torch
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
        # Check that it raises an error if we do not have all the data sets
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

        # Check that extraction worked as specified.
        n_prior, n_train, n_test, n_val = check_and_extract_number_of_datapoints(number_data)
        self.assertEqual(n_prior, number_data['prior'])
        self.assertEqual(n_train, number_data['train'])
        self.assertEqual(n_test, number_data['test'])
        self.assertEqual(n_val, number_data['validation'])

    def test_get_blurring_kernel(self):
        # Check that kernel is specified as claimed.
        kernel = get_blurring_kernel()
        self.assertIsInstance(kernel, torch.Tensor)
        self.assertEqual(kernel.shape, torch.Size((1, 1, 5, 5)))

    def test_get_finite_difference_kernels(self):
        # Check that difference kernels are specified correclty, and that they get returned in correct order.
        k_w, k_h = get_finite_difference_kernels()
        self.assertIsInstance(k_w, torch.Tensor)
        self.assertEqual(k_w.shape, torch.Size((1, 1, 3, 3)))
        self.assertTrue(k_w[0, 0, 1, 2] == 1)
        self.assertIsInstance(k_h, torch.Tensor)
        self.assertEqual(k_h.shape, torch.Size((1, 1, 3, 3)))
        self.assertTrue(k_h[0, 0, 2, 1] == 1)

    def test_get_shape_of_image(self):
        # Check that the shape of the images fits the one needed for PyTorch-Conv2d layers.
        shape = get_shape_of_images()
        self.assertIsInstance(shape, tuple)
        self.assertEqual(len(shape), 4)
        self.assertEqual(shape[0], 1)
        self.assertEqual(shape[1], 1)

    def test_get_image_height_and_width(self):
        # Check that this returns the same values that are specified in get_shape_of_images().
        shape = get_shape_of_images()
        height, width = get_image_height_and_width()
        self.assertEqual(height, shape[2])
        self.assertEqual(width, shape[3])

    def test_get_epsilon(self):
        # Just check that it's a non-negative float.
        eps = get_epsilon()
        self.assertIsInstance(eps, float)
        self.assertTrue(eps > 0)

    def test_get_distribution_of_regularization_parameters(self):
        # Check that we sample from the right distribution class.
        # Exact values not specified here.
        dist = get_distribution_of_regularization_parameter()
        self.assertIsInstance(dist, torch.distributions.uniform.Uniform)

    def test_get_largest_possible_regularization_parameter(self):
        # Check that this value coincides with the value of the corresponding distribution.
        p = get_largest_possible_regularization_parameter()
        self.assertIsInstance(p, float)
        self.assertEqual(p, get_distribution_of_regularization_parameter().high.item())

    def test_get_loss_function_of_algorithm(self):
        # Check that we get callables.
        blurring_kernel = get_blurring_kernel()
        loss_function, blur_tensor = get_loss_function_of_algorithm(blurring_kernel)
        self.assertIsInstance(loss_function, Callable)
        self.assertIsInstance(blur_tensor, Callable)
        img = load_and_transform_image(path=self.image_path + self.image_name_test, device='cpu')
        val = loss_function(img, parameter={'img': img, 'mu': torch.tensor(1.)})
        self.assertIsInstance(val, torch.Tensor)
        self.assertIsInstance(val.item(), float)

    def test_get_smoothness_parameter(self):
        # This is a weak test: Only check that the smoothness parameter is a non-negative float.
        p = get_smoothness_parameter()
        self.assertIsInstance(p, float)
        self.assertTrue(p > 0)

    def test_load_and_transform_image(self):
        # Check that image gets loaded correctly and gets transformed as specified.
        img = load_and_transform_image(path=self.image_path + self.image_name_test, device='cpu')
        height, width = get_image_height_and_width()
        self.assertIsInstance(img, torch.Tensor)
        self.assertEqual(img.shape, torch.Size([1, height, width]))

    def test_load_images(self):
        # Check that all images get loaded into the specified shape
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
        with self.assertRaises(Exception):
            all_images = all_images[0:2]
            split_images_into_separate_sets(all_images)

    def test_get_noise_distribution(self):
        # Check that the noise-distribution (at least the class) is as specified.
        dist = get_noise_distribution()
        self.assertIsInstance(dist, torch.distributions.normal.Normal)

    def test_clip_to_interval(self):
        # Check that all values lie in [0,1].
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
        for blurred_img, img in zip(blurred, all_images[:num_imgs]):
            # Does not change shape, but does change img.
            self.assertEqual(blurred_img.shape, shape)
            self.assertTrue(torch.linalg.norm(blurred_img - img) > 0)

    def test_get_blurred_images(self):
        all_images = load_images(path_to_images=self.image_path, device='cpu')
        all_images = all_images[0:8]
        loss_function, blur_tensor = get_loss_function_of_algorithm(get_blurring_kernel())
        b_i_p, b_i_tr, b_i_te, b_i_v = get_blurred_images(images=all_images, blurring_function=blur_tensor)
        # Only have to check that we get several lists of images. Blurring itself is already tested.
        self.assertIsInstance(b_i_p, list)
        self.assertIsInstance(b_i_tr, list)
        self.assertIsInstance(b_i_te, list)
        self.assertIsInstance(b_i_v, list)
        self.assertTrue(len(b_i_p) > 0)
        self.assertTrue(len(b_i_tr) > 0)
        self.assertTrue(len(b_i_te) > 0)
        self.assertTrue(len(b_i_v) > 0)

    def test_get_parameter(self):
        # Initialize setting.
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

        # Check that each entry has the correct number of data points
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
        # Only have to check roughly. More details are already checked above.
        self.assertIsInstance(parameters, dict)
        self.assertIsInstance(loss_function_of_algorithm, Callable)
        self.assertIsInstance(smoothness_parameter, float)
