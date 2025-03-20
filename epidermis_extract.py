import csv

import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import optim
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
import cv2
from skimage import segmentation
import numpy as np
import openslide
import matplotlib.pyplot as plt
import os
import pandas as pd
#%%
class infonet(nn.Module):
    def __init__(self,input_dim,nChannel,nConv):
        super(infonet, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, nChannel, kernel_size=3, stride=1, padding=1 )
        self.bn1 = nn.BatchNorm2d(nChannel)
        self.conv2 = nn.ModuleList()
        self.bn2 = nn.ModuleList()
        for i in range(nConv-1):
            self.conv2.append( nn.Conv2d(nChannel, nChannel, kernel_size=3, stride=1, padding=1 ) )
            self.bn2.append( nn.BatchNorm2d(nChannel) )
        self.conv3 = nn.Conv2d(nChannel, nChannel, kernel_size=1, stride=1, padding=0 )
        self.bn3 = nn.BatchNorm2d(nChannel)
        self.nConv = nConv

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu( x )
        x = self.bn1(x)
        for i in range(self.nConv-1):
            x = self.conv2[i](x)
            x = F.relu( x )
            x = self.bn2[i](x)
        x = self.conv3(x)
        x = self.bn3(x)
        return x
class SegModel():
    nChannel = 100
    maxIter = 400
    minLabels = 4
    lr = 0.1
    nConv = 2
    num_superpixels = 10000
    compactness = 100
    visualize = 1
    label_indices = []

    def __init__(self, use_cuda, visualize=1):
        self.use_cuda = use_cuda
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.label_colors = np.random.randint(255, size=(100, 3))  # Randomly generate pixel colors for 100 feature maps
        self.visualize = visualize
        self.image_shape = []

    def build_model(self):
        self.model = infonet(self.data.size(1), nChannel=self.nChannel, nConv=self.nConv)
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        if self.use_cuda:
            self.model.cuda()
        self.model.train()

    def draw_model(self):
        with SummaryWriter(comment="FPN") as w:
            w.add_graph(model=self.model, input_to_model=torch.rand(15, 3, 224, 224))

    def pre_segment_model(self):
        for batch_idx in range(self.maxIter):
            self.optimizer.zero_grad()
            output = self.model(self.data)[0]
            output = output.permute(1, 2, 0).contiguous().view(-1, self.nChannel)
            _, target = torch.max(output, 1)
            self.image_target = target.data.cpu().numpy()
            nLabels = len(np.unique(self.image_target))

            if self.visualize:
                image_target_rgb = np.array([self.label_colors[c % 100] for c in self.image_target])
                image_target_rgb = image_target_rgb.reshape(self.image.shape).astype(np.uint8)
                cv2.imshow("output", image_target_rgb)
                cv2.waitKey(10)

            # Superpixel refinement
            for i in range(len(self.label_indices)):
                labels_per_sp = self.image_target[self.label_indices[i]]
                unique_labels_per_sp = np.unique(labels_per_sp)
                hist = np.zeros(len(unique_labels_per_sp))
                for j in range(len(hist)):
                    hist[j] = len(np.where(labels_per_sp == unique_labels_per_sp[j])[0])
                self.image_target[self.label_indices[i]] = unique_labels_per_sp[np.argmax(hist)]

            target = torch.from_numpy(self.image_target)
            if self.use_cuda:
                target = target.cuda()
            target = Variable(target)
            loss = self.loss_fn(output, target)
            loss.backward()
            self.optimizer.step()

            if nLabels <= self.minLabels:
                break

            # Save output image
            if not self.visualize:
                output = self.model(self.data)[0]
                output = output.permute(1, 2, 0).contiguous().view(-1, self.nChannel)
                _, target = torch.max(output, 1)
                self.image_target = target.data.cpu().numpy()
                image_target_rgb = np.array([self.label_colors[c % 100] for c in self.image_target])
                image_target_rgb = image_target_rgb.reshape(self.image.shape).astype(np.uint8)

        cv2.imwrite("output.bmp", image_target_rgb)

    def post_process_segmentation(self):
        segmented_image = self.image_target.reshape(self.image.shape[0:2]).astype(np.uint8)
        unique_labels = np.unique(self.image_target)

        sample_pixel_count = 50
        image_pixels = np.array(self.image).reshape(-1, 3)
        sampled_pixel_means = []

        for label in unique_labels:
            pixel_locations = np.where(self.image_target == label)[0]
            pixel_locations = pixel_locations[np.random.randint(0, len(pixel_locations), sample_pixel_count)]
            sampled_pixels = image_pixels[pixel_locations]
            sampled_pixel_means.append(np.mean(sampled_pixels, axis=0))

        target_pixel_color = np.array([[77.48, 36.72, 107.56]])
        target_pixel_color = target_pixel_color.repeat(repeats=len(unique_labels), axis=0)
        sampled_pixel_means = np.array(sampled_pixel_means)

        euclidean_distances = np.linalg.norm(target_pixel_color - sampled_pixel_means, axis=1)
        min_distance_index = euclidean_distances.argmin()

        segmented_image[segmented_image != unique_labels[min_distance_index]] = 0
        segmented_image[segmented_image == unique_labels[min_distance_index]] = 255

        kernel = np.ones((3, 3), np.uint8)
        segmented_image = cv2.dilate(segmented_image, kernel)
        segmented_image = cv2.erode(segmented_image, kernel)
        segmented_image = cv2.erode(segmented_image, kernel)
        segmented_image = cv2.dilate(segmented_image, kernel)

        contours, _ = cv2.findContours(segmented_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        for i in range(len(contours)):
            if cv2.contourArea(contours[i]) < 5000:
                segmented_image = cv2.drawContours(segmented_image, [contours[i]], -1, 0, thickness=-1)

        final_segmentation = 255 * np.ones(self.image.shape[0:2]).astype(np.uint8)
        segmented_image = np.array(segmented_image).astype(np.uint8)

        final_segmentation = np.subtract(final_segmentation, segmented_image).astype(np.uint8)
        contours, _ = cv2.findContours(final_segmentation, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        for i in range(len(contours)):
            if cv2.contourArea(contours[i]) < 500:
                final_segmentation = cv2.drawContours(final_segmentation, [contours[i]], -1, 0, thickness=-1)

        segmented_image = 255 * np.ones(self.image.shape[0:2]).astype(np.uint8)
        segmented_image = np.subtract(segmented_image, final_segmentation).astype(np.uint8)

        self.segmented_image = segmented_image
        self.final_segmentation = final_segmentation

    def load_image(self, image):
        self.image = image
        self.data = torch.from_numpy(
            np.array([self.image.transpose((2, 0, 1)).astype('float32') / 255.]))  # Convert to tensor
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data = self.data.to(device)

    def segment_tissue(self):
        labels = segmentation.felzenszwalb(self.image, scale=32, sigma=0.5, min_size=64)
        labels = labels.reshape(self.image.shape[0] * self.image.shape[1])
        unique_labels = np.unique(labels)
        self.label_indices = []
        for i in range(len(unique_labels)):
            self.label_indices.append(np.where(labels == unique_labels[i])[0])

        self.build_model()
        self.pre_segment_model()
        self.post_process_segmentation()
        return self.segmented_image, self.final_segmentation
def cut_pic(slide, image_height=900, image_width=900, screenshot_level=1,distance_threshold=20, target_pixel_color=[76.78, 37.92, 109.32]):
    num_row_images = slide.level_dimensions[screenshot_level][1] // image_height
    num_col_images = slide.level_dimensions[screenshot_level][0] // image_width
    zoom_ratio = slide.level_dimensions[0][1] // slide.level_dimensions[screenshot_level][1]
    vertical_correction_values = np.zeros(num_col_images)
    cropped_image_list = []
    cropped_image_loc = []
    for row in range(num_row_images):
        horizontal_correction_value = 0
        for column in range(num_col_images):
            # print("Total Progress:", current, '/', total, "   ", "Progress:", row * num_col_images + column, "/", num_row_images * num_col_images)
            x = column * image_width * zoom_ratio + int(horizontal_correction_value) * zoom_ratio
            y = row * image_height * zoom_ratio + int(vertical_correction_values[column]) * zoom_ratio
            region = slide.read_region((x, y), screenshot_level, (image_width, image_height))
            region = np.array(region)
            b, g, r, a = cv2.split(region)
            cropped_image = cv2.merge([r, g, b])
            euclidean_distance = np.linalg.norm(target_pixel_color - cropped_image, axis=2)
            min_euclidean_index = np.min(euclidean_distance)

            if min_euclidean_index < distance_threshold:
                similar_pixel_indices = np.argwhere(euclidean_distance < distance_threshold)
                num_similar_pixels = similar_pixel_indices.shape[0]

                if num_similar_pixels > 1000:
                    x_correction_value = np.mean(similar_pixel_indices[:, 1]) - image_width / 2

                    if x_correction_value < 0:  # Only correct forward, not backward, to avoid overlap
                        _ = 1

                    if 0 < x_correction_value < 100:  # Small error, correct and finish
                        region = slide.read_region((int(x + x_correction_value * zoom_ratio), y), screenshot_level,
                                                   (image_width, image_height))
                        region = np.array(region)
                        b, g, r, a = cv2.split(region)
                        cropped_image = cv2.merge([r, g, b])
                        horizontal_correction_value += x_correction_value

                    if x_correction_value >= 100:  # Large error, check for new pixels after correction (ensure edge completeness)
                        x_correction_value += 50
                        old_x_correction_value = x_correction_value
                        cumulative_correction_value = x_correction_value
                        max_iterations = 20

                        while True:
                            region = slide.read_region((int(x + cumulative_correction_value * zoom_ratio), y),
                                                       screenshot_level, (image_width, image_height))
                            region = np.array(region)
                            b, g, r, a = cv2.split(region)
                            cropped_image = cv2.merge([r, g, b])
                            euclidean_distance = np.linalg.norm(target_pixel_color - cropped_image, axis=2)
                            similar_pixel_indices = np.argwhere(euclidean_distance < distance_threshold)
                            new_x_correction_value = np.mean(similar_pixel_indices[:, 1]) - image_width / 2

                            if new_x_correction_value - old_x_correction_value < 0 or max_iterations == 0:  # If change is minimal, consider edge stable
                                horizontal_correction_value += cumulative_correction_value
                                break

                            cumulative_correction_value += new_x_correction_value - old_x_correction_value
                            # print(str(row)+'_'+str(column), np.mean(similar_pixel_indices[:,1]), 'new', new_x_correction_value, 'old', old_x_correction_value)
                            old_x_correction_value = new_x_correction_value
                            max_iterations -= 1

                    y_correction_value = np.mean(similar_pixel_indices[:, 0]) - image_width / 2

                    if y_correction_value > 100:  # Large error, correct and finish
                        region = slide.read_region((int(x), int(y + y_correction_value * zoom_ratio)), screenshot_level,
                                                   (image_width, image_height))
                        region = np.array(region)
                        b, g, r, a = cv2.split(region)
                        cropped_image = cv2.merge([r, g, b])
                        vertical_correction_values[column] += y_correction_value

                    cropped_image_list.append(cropped_image)
                    cropped_image_loc.append([int(x), int(y + y_correction_value * zoom_ratio),screenshot_level,image_width, image_height])
    return cropped_image_list,cropped_image_loc
def distance_to_edge(point, edge_points):
    point_matrix = np.full(edge_points.shape, point)
    distance_array = np.linalg.norm(point_matrix - edge_points, axis=1)
    return distance_array
def draw_edge(image, edge_points, color):
    image = np.array(image)
    image = np.ones(image.shape) * 255
    if len(image.shape) == 2:
        image = image[:, :, np.newaxis]
        image = image.repeat(3, axis=2)
    elif len(image.shape) == 3 and image.shape[2] == 1:
        image = image.repeat(3, axis=2)
    for i in edge_points:
        image[i[1], i[0], :] = color
    plt.imshow(image)
    plt.show()
def check_image(target_light, target_dark):
    contours, hierarchy = cv2.findContours(target_dark, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    filtered_edges = []
    edge_counts = []

    for i in range(len(contours)):
        temp = np.array(contours[i]).reshape(-1, 2)
        temp = temp[(temp[:, 1] >= 10) & (temp[:, 0] >= 10) & (temp[:, 1] <= target_light.shape[0] - 10) & (
                    temp[:, 0] <= target_light.shape[1] - 10)]
        filtered_edges.append(temp)
        edge_counts.append(temp.shape[0])

    if len(edge_counts) == 0 or len(edge_counts) == 1:
        print('No edges found, returning False')
        return False, []
    if np.max(edge_counts) < 200:
        print('Edges too small, returning False')
        return False, []

    selected_edges = []
    for edge_set in filtered_edges:
        distance_array = distance_to_edge([900, 900], edge_set)
        distance_diff = np.diff(distance_array, n=1)

        if abs(distance_array[0] - distance_array[-1]) < 50 and np.max(abs(distance_diff)) < 10:
            print('Closed shape found')
            continue
        selected_edges.append(edge_set)

    if len(selected_edges) == 0:
        print('No suitable edges found')
        return False, []
    return True, selected_edges
def compute_edge_thickness(target_dark, fixed_edge, target_edge):
    euclidean_distances = [[], []]
    draw_edge(target_dark, fixed_edge, [0, 0, 0])
    draw_edge(target_dark, target_edge, [0, 0, 0])

    for i in range(fixed_edge.shape[0]):
        coord = np.full(target_edge.shape, fixed_edge[i])
        temp = np.linalg.norm(coord - target_edge, axis=1)
        euclidean_distances[0].append(np.min(temp))

    for i in range(target_edge.shape[0]):
        coord = np.full(fixed_edge.shape, target_edge[i])
        temp = np.linalg.norm(coord - fixed_edge, axis=1)
        euclidean_distances[1].append(np.min(temp))

    return euclidean_distances

#%%
from tqdm import tqdm
import argparse
parse = argparse.ArgumentParser()
parse.add_argument('-data_dir', type=str, required=True,help='Path to the image dataset.')
parse.add_argument('-save_dir', type=str, required=True,help='Directory where the model configuration and weights will be saved.')
args = parse.parse_args()

save_dir = args.save_dir
data_dir = args.data_dir

if os.path.exists(save_dir) == False:
    os.makedirs(save_dir)

for image_file in tqdm(os.listdir(data_dir)):
    file_path = os.path.join(data_dir,image_file)
    slide = openslide.OpenSlide(file_path)
    image_cut_list,cropped_image_loc_list = cut_pic(slide)
    image_seg_list = []
    image_loc_list = []
    for image_cut,image_loc in zip(image_cut_list,cropped_image_loc_list):
        tissue_segmentation = SegModel(use_cuda=torch.cuda.is_available(), visualize=0)
        tissue_segmentation.load_image(image_cut)
        target_light, target_dark = tissue_segmentation.segment_tissue()
        image_seg_list.append(target_light)
        image_loc_list.append(image_loc)

    save_seg_dir = os.path.join(save_dir,image_file)
    if os.path.exists(save_seg_dir) == False:
        os.makedirs(save_seg_dir)
    f = open(os.path.join(save_seg_dir,'image_info.csv'),'w')
    w = csv.writer(f)
    w.writerow(['index','x', 'y', 'screenshot_level', 'image_width', 'image_height'])
    for index,mask in enumerate(image_seg_list):
        loc = image_loc_list[index]
        x, y, screenshot_level, image_width, image_height = loc
        save_file = os.path.join(save_seg_dir,'{}.png'.format(index))
        cv2.imwrite(filename=save_file, img=mask)
        w.writerow([index, x, y, screenshot_level, image_width, image_height])
    f.close()
