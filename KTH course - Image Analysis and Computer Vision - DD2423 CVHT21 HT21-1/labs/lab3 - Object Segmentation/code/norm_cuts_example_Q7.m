%// 4. NORMALIZED CUT

%--------------------Parameters-------------------------
colour_bandwidth = 15.0;   % color bandwidth
radius = 10;              % maximum neighbourhood distance

%----Control of recursive subdivision-----
ncuts_thresh = 0.5;      % cutting threshold
min_area = 10;           % minimum area of segment
max_depth = 10;           % maximum splitting depth
%-----------------------------------------

scale_factor = 0.4;     % image downscale factor
image_sigma = 2.0;      % image preblurring scale
%-------------------------------------------------------

I = imread('tiger3.jpg');
I = imresize(I, scale_factor);
Iback = I;
d = 2*ceil(image_sigma*2) + 1;
h = fspecial('gaussian', [d d], image_sigma);
I = imfilter(I, h);

segm = norm_cuts_segm(I, colour_bandwidth, radius, ncuts_thresh, min_area, max_depth);
Inew = mean_segments(Iback, segm);
I = overlay_bounds(Iback, segm);
imwrite(Inew,'results/normcuts1_3.png')
imwrite(I,'results/normcuts2_3.png')

figure
subplot(1,2,1); imshow(Inew); title('Segmentation');
subplot(1,2,2); imshow(I); title('Overlaying bounds');
